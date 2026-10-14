# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify native pool sampling in the matching TZRec model image."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from tzrec.constant import Mode
from tzrec.datasets.parquet_dataset import ParquetDataset
from tzrec.datasets.pool_sampler import PoolNegativeSampler
from tzrec.datasets.sampler import BaseSampler
from tzrec.features.feature import create_features
from tzrec.models.dssm import DSSM
from tzrec.protos import data_pb2, feature_pb2, loss_pb2, model_pb2, module_pb2, sampler_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils.state_dict_util import init_parameters


class PoolSamplerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / "train.parquet")
        self.data = pa.table({"u": [float(i) for i in range(9)],
                              "i": list(range(9)), "v": [float(i * 10) for i in range(9)],
                              "tags": [[str(i), "a"] for i in range(9)], "label": [1] * 9})
        pq.write_table(self.data, self.path)
        self.config = sampler_pb2.PoolNegativeSampler(input_path=self.path, num_sample=7,
                pool_size=5, item_id_field="i", attr_fields=["i", "v", "tags"])

    def sampler(self, training=True):
        sampler = BaseSampler.create_class("PoolNegativeSampler")(
            self.config, list(self.data.schema), 2, is_training=training)
        self.assertIsInstance(sampler, PoolNegativeSampler)
        sampler.init_cluster()
        sampler.launch_server()
        sampler.init()
        return sampler

    def batch(self, start, size=2):
        table = self.data.slice(start, size)
        return {name: table[name].combine_chunks() for name in table.column_names}

    def test_registration_warmup_fifo_alignment_and_exclusion(self):
        with patch("tzrec.datasets.sampler.gl.Graph", side_effect=AssertionError("GraphLearn must not run")):
            sampler = self.sampler()
            self.assertEqual(list(sampler._pool), [0, 1, 2, 3])
            result = sampler.get(self.batch(0))
            self.assertEqual(len(result["i"]), 7)
            self.assertTrue(set(result["i"].to_pylist()) <= {2, 3})
            self.assertEqual(result["v"].to_pylist(), [i * 10. for i in result["i"].to_pylist()])
            self.assertEqual(result["tags"].to_pylist(), [[str(i), "a"] for i in result["i"].to_pylist()])
            changed = self.batch(0)
            changed["v"] = pa.array([999., 888.])
            sampler.get(changed)
            self.assertEqual(list(sampler._pool), [0, 1, 2, 3])
            self.assertEqual(sampler._pool[0]["v"], 999.)
            sampler.get(self.batch(4))
            self.assertEqual(list(sampler._pool), [1, 2, 3, 4, 5])
            sampler.get(self.batch(8, 1))
            self.assertEqual(list(sampler._pool), [2, 3, 4, 5, 8])

    def test_evaluation_never_updates_from_validation_rows(self):
        self.config.num_eval_sample = 11
        sampler = self.sampler(training=False)
        snapshot = dict(sampler._pool)
        result = sampler.get(self.batch(8, 1))
        self.assertEqual(len(result["i"]), 11)
        self.assertEqual(snapshot, dict(sampler._pool))
        self.assertNotIn(8, result["i"].to_pylist())

    def test_private_rng_reproducibility_and_no_global_rng_changes(self):
        torch.manual_seed(123)
        sampler = self.sampler(training=False)
        np.random.seed(15)
        state = np.random.get_state()
        first = sampler.get(self.batch(8, 1))["i"].to_pylist()
        second = sampler.get(self.batch(8, 1))["i"].to_pylist()
        self.assertNotEqual(first, second)
        current = np.random.get_state()
        np.testing.assert_array_equal(state[1], current[1])
        self.assertEqual(state[2:], current[2:])
        self.assertEqual(first, self.sampler(training=False).get(self.batch(8, 1))["i"].to_pylist())

    def test_invalid_capacity_source_and_missing_candidate_fail_clearly(self):
        self.config.pool_size = 2
        with self.assertRaisesRegex(ValueError, "pool_size"):
            self.sampler()
        self.config.pool_size = 5
        self.config.input_path = str(Path(self.temp.name) / "missing.parquet")
        with self.assertRaisesRegex(ValueError, "No parquet"):
            self.sampler()
        self.config.input_path = self.path
        sampler = self.sampler()
        with self.assertRaisesRegex(ValueError, "No pool negatives"):
            sampler.get(self.batch(0, 4))
        pq.write_table(self.data.slice(0, 2), self.path)
        with self.assertRaisesRegex(ValueError, "distinct training"):
            self.sampler()

    def test_native_dataset_loss_gradient_and_short_batch(self):
        self.config.attr_fields[:] = ["i", "v"]
        feature_configs = [feature_pb2.FeatureConfig(raw_feature=feature_pb2.RawFeature(
            feature_name=name, expression="item:" + name)) for name in ("u", "i", "v")]
        features = create_features(feature_configs, fg_mode=data_pb2.FG_NORMAL,
                                   neg_fields=list(self.config.attr_fields))
        data_config = data_pb2.DataConfig(batch_size=2, num_workers=0,
                fg_mode=data_pb2.FG_NORMAL, label_fields=["label"], pool_negative_sampler=self.config)
        dataset = ParquetDataset(data_config, features, self.path, mode=Mode.TRAIN)
        dataset.launch_sampler_cluster()
        batches = list(dataset)
        self.assertEqual([len(b.labels["label"]) for b in batches], [2, 2, 2, 2, 1])
        config = model_pb2.ModelConfig(
            feature_groups=[model_pb2.FeatureGroupConfig(group_name=name, feature_names=names,
                group_type=model_pb2.DEEP) for name, names in (("user", ["u"]), ("item", ["i", "v"]))],
            dssm=match_model_pb2.DSSM(
                user_tower=tower_pb2.Tower(input="user", mlp=module_pb2.MLP(hidden_units=[4], use_bn=False)),
                item_tower=tower_pb2.Tower(input="item", mlp=module_pb2.MLP(hidden_units=[4], use_bn=False)),
                output_dim=2, in_batch_negative=False),
            losses=[loss_pb2.LossConfig(softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy())])
        model = DSSM(config, features, ["label"], sampler_type="pool_negative_sampler")
        init_parameters(model, device=torch.device("cpu"))
        model.eval()
        model.init_loss()
        for batch in (batches[0], batches[-1]):
            prediction = model(batch)
            size = len(batch.labels["label"])
            logits = prediction["similarity"]
            self.assertEqual(tuple(logits.shape), (size, 8))
            expected = F.cross_entropy(logits, torch.zeros(size, dtype=torch.long))
            loss = model.loss(prediction, batch)["softmax_cross_entropy"]
            torch.testing.assert_close(loss, expected)
            expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
            actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            torch.testing.assert_close(actual_grad, expected_grad)
            loss.backward()
        evaluation = ParquetDataset(data_config, features, self.path, mode=Mode.EVAL)
        evaluation.launch_sampler_cluster(cluster=dataset.get_sampler_cluster())
        list(evaluation)
        self.assertEqual(list(evaluation._sampler._pool), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
