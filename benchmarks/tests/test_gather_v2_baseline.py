"""Check page resolution and non-FP8 filtering independently of ACL execution."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.ascend import gather_v2_baseline as baseline


class GatherV2BaselineTest(unittest.TestCase):
    def test_live_metadata_empty_request_and_unaligned_starts(self):
        table = torch.tensor([[4, 1, 3], [2, 0, 4], [1, 3, 2]], dtype=torch.int32)
        cu = torch.tensor([0, 3, 3, 8], dtype=torch.int32)
        starts = torch.tensor([3, 0, 1], dtype=torch.int32)
        rows = torch.arange(8)
        for use_starts in (None, starts):
            for current in (table, table.flip(-1)):
                actual = baseline.physical_slots(current, cu, use_starts, rows, 4)
                expected = []
                for b in range(3):
                    for j in range(int(cu[b + 1] - cu[b])):
                        pos = j + (int(use_starts[b]) if use_starts is not None else 0)
                        expected.append(int(current[b, pos // 4]) * 4 + pos % 4)
                self.assertEqual(actual.tolist(), expected)

    def test_fp8_does_not_load_acl_library(self):
        case = SimpleNamespace(params={"cache": ((1, 64, 576), "float8_e4m3fn")})
        with patch.object(baseline, "_library", side_effect=AssertionError("must not load")):
            self.assertEqual(baseline.make_implementations("PagedKVCacheGatherFwdOp", case), {})

    def test_full_recomputes_indices_core_reuses_them(self):
        inputs = (torch.empty(3, 2), torch.arange(32, dtype=torch.float32).view(4, 4, 2),
                  torch.tensor([[2, 0]], dtype=torch.int32),
                  torch.tensor([0, 3], dtype=torch.int32), None, None)

        class FakeGather:
            provenance = {"api": "test"}

            def __call__(self, cache, slots, out):
                out.copy_(cache[slots])
                return out

        with patch.object(baseline, "GatherV2", FakeGather):
            full = baseline._Run(inputs, include_index_generation=True)
            core = baseline._Run(inputs, include_index_generation=False)
            torch.testing.assert_close(full(), core())
            inputs[2][0, 0] = 1
            torch.testing.assert_close(full(), inputs[1].flatten(0, 1)[4:7])
            torch.testing.assert_close(core(), inputs[1].flatten(0, 1)[8:11])
            full.reset()
            self.assertTrue(bool(full.dst.isnan().all()))


if __name__ == "__main__":
    unittest.main()
