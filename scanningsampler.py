from torch._six import int_classes as _int_classes
from torch import Tensor

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, Sampler


# batch = list(BatchSampler(SequentialSampler(range(20)), batch_size=5, drop_last=False))
# print(batch)


# def _int_classes(args):
#     pass


# def batch(args):
#     pass


class ScanningBatchSampler():
    # r"""Wraps another sampler to yield a mini-batch of indices.
    # Args:
    #     sampler (Sampler or Iterable): Base sampler. Can be any iterable object
    #     batch_size (int): Size of mini-batch.
    #     drop_last (bool): If ``True``, the sampler will drop the last batch if
    #         its size would be less than ``batch_size``
    # Example:
    #     >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
    #     [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    #     >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
    #     [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # """

    def __init__(self, sampler, batch_size):
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        # if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
        #         batch_size <= 0:
        #     raise ValueError("batch_size should be a positive integer value, "
        #                      "but got batch_size={}".format(batch_size))
        # if not isinstance(drop_last, bool):
        #     raise ValueError("drop_last should be a boolean value, but got "
        #                      "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        # self.drop_last = drop_last
        # self.generator = generator

    def __iter__(self):
        batch = []
        # iter(batch)
        idx = 0
        while idx < len(self.sampler):
            batch.append(idx)
            # print(idx)
            idx += 1
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                idx = idx - self.batch_size + 1

        # def __next__(self):
        #     pass

        # for idx in self.sampler:
        #     batch.append(idx)
        #     if len(batch) == self.batch_size:
        #         yield batch
        #
        #         batch = []
        #         # i += 1
        #         idx_sample = idx
        #         - self.batch_size + 1

        # if len(batch) > 0:
        #     yield batch

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

        return len(self.sampler) - self.batch_size + 1  # type: ignore
        # else:
        #     return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore


#batch = list(ScanningBatchSampler(SequentialSampler(range(20)), batch_size=5))
#print(batch)

