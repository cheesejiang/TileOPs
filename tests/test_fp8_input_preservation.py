"""FP8 preservation checks detect byte changes without FP8 comparison kernels."""

import unittest

import torch

from workloads.numerics import _require_unchanged, _snapshot


class FP8InputPreservationTest(unittest.TestCase):
    def test_all_bit_patterns_and_strided_views(self):
        raw = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
        tensor = raw.view(torch.float8_e4m3fn)
        for view in (tensor, tensor[:, ::2], tensor.t()):
            _require_unchanged([view], _snapshot([view]))

    def test_mutation_is_rejected_including_signed_zero_and_nan(self):
        for before, after in ((0, 128), (127, 255), (56, 57)):
            raw = torch.tensor([before], dtype=torch.uint8)
            tensor = raw.view(torch.float8_e4m3fn)
            pristine = _snapshot([tensor])
            raw[0] = after
            with self.assertRaisesRegex(AssertionError, "overwrote the shared inputs"):
                _require_unchanged([tensor], pristine)


if __name__ == "__main__":
    unittest.main()
