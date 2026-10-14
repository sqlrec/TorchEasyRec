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

"""Bounded feature pools populated from training positives without GraphLearn."""

import glob
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import get_worker_info

from tzrec.datasets.sampler import BaseSampler
from tzrec.protos import sampler_pb2
from tzrec.utils.filesystem_util import url_to_fs


class PoolNegativeSampler(BaseSampler):
    """Sample cached raw item features, with one FIFO pool per worker.

    Args:
        config: Pool configuration with an existing training parquet source.
        fields: Input field types before feature encoding.
        batch_size: Maximum positive batch size.
        is_training: Whether current positives may update the pool.
        multival_sep: Unused; Arrow list features retain their original types.
    """

    def __init__(
        self,
        config: sampler_pb2.PoolNegativeSampler,
        fields: List[pa.Field],
        batch_size: int,
        is_training: bool = True,
        multival_sep: str = chr(29),
    ) -> None:
        self._g = None
        self._cluster = None
        self._config = config
        self._batch_size = batch_size
        self._is_training = is_training
        self._num_sample = config.num_sample
        if not is_training and config.num_eval_sample:
            self._num_sample = config.num_eval_sample
        self._fields = {f.name: f.type for f in fields}
        self._attrs = list(config.attr_fields)
        self._id = config.item_id_field
        self._pool: OrderedDict = OrderedDict()
        self._rng = None
        if self._num_sample < 1 or config.pool_size <= batch_size:
            raise ValueError(
                "Positive sample count and pool_size > batch_size required"
            )
        if not self._attrs or len(set(self._attrs)) != len(self._attrs):
            raise ValueError("Pool attributes must be nonempty and unique")
        if self._id not in self._attrs or set(self._attrs) - self._fields.keys():
            raise ValueError(
                "Pool attributes must include the item ID and exist in inputs"
            )
        id_type = self._fields[self._id]
        if not (pa.types.is_string(id_type) or id_type in (pa.int32(), pa.int64())):
            raise ValueError("Pool item ID must be a scalar string, int32 or int64")

    def init_cluster(
        self,
        num_client_per_rank: int = 1,
        client_id_bias: int = 0,
        cluster: Optional[Dict[str, Union[int, str]]] = None,
    ) -> None:
        """Use local worker state; no graph service or cross-worker cache exists."""
        self._cluster = {"pool": 1}

    def launch_server(self) -> None:
        """Keep the dataset lifecycle without starting a graph server."""
        pass

    def init(self, client_id: int = -1) -> None:
        """Seed a private RNG and stream enough training items for the first batch."""
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else max(client_id, 0)
        seed = [torch.initial_seed(), int(os.environ.get("RANK", "0")), worker_id]
        self._rng = np.random.default_rng(np.random.SeedSequence(seed))
        self._pool.clear()
        paths = sorted(
            {
                path
                for pattern in self._config.input_path.split(",")
                for path in glob.glob(pattern)
            }
        )
        if not paths:
            raise ValueError("No parquet training files found for pool warmup")
        for path in paths:
            fs, _ = url_to_fs(path)
            with pq.ParquetFile(path, filesystem=fs) as source:
                for batch in source.iter_batches(
                    batch_size=self._batch_size, columns=self._attrs
                ):
                    self._update(dict(zip(batch.schema.names, batch.columns)))
                    if len(self._pool) > self._batch_size:
                        return
        raise ValueError(
            "Pool warmup needs more distinct training items than batch_size"
        )

    def _update(self, data: Dict[str, pa.Array]) -> None:
        """Update item attributes without moving existing IDs; evict oldest IDs."""
        table = pa.table({name: data[name] for name in self._attrs})
        for row in table.to_pylist():
            item = row[self._id]
            if item is None or item == "":
                raise ValueError("Pool item IDs must not be null or empty")
            self._pool[item] = row
            if len(self._pool) > self._config.pool_size:
                self._pool.popitem(last=False)

    def get(self, input_data: Dict[str, pa.Array]) -> Dict[str, pa.Array]:
        """Sample shared negatives, then update the pool only during training.

        Args:
            input_data: Current positive batch before sampled columns are appended.

        Returns:
            Typed item feature arrays, with num_sample rows and possible duplicates.
        """
        positive_ids = set(input_data[self._id].to_pylist())
        candidates = [item for item in self._pool if item not in positive_ids]
        if not candidates:
            raise ValueError("No pool negatives remain after batch-positive exclusion")
        indices = self._rng.integers(len(candidates), size=self._num_sample)
        rows = [self._pool[candidates[i]] for i in indices]
        result = {
            name: pa.array([row[name] for row in rows], type=self._fields[name])
            for name in self._attrs
        }
        if self._is_training:
            self._update(input_data)
        return result

    @property
    def estimated_sample_num(self) -> int:
        """Return the number of shared negative item rows added to each batch."""
        return self._num_sample
