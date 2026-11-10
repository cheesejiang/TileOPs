"""Ascend950 row batches, generic tails, FP8 ties and untouched cache rows."""

import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None, reason="requires Ascend and torch_npu"
)


def _selected_device():
    from workloads.device import run_device

    device = torch.device(run_device())
    if device.type != "npu":
        device = torch.device(f"npu:{torch.npu.current_device()}")
    torch.npu.set_device(device)
    return device


@pytest.mark.parametrize(
    "n,h,d,npages,page,dtype,fp8",
    [
        pytest.param(7, 2, 64, 5, 7, torch.float16, False, marks=pytest.mark.smoke),
        pytest.param(17, 4, 128, 7, 7, torch.bfloat16, True, marks=pytest.mark.smoke),
        pytest.param(9, 2, 65, 5, 7, torch.float16, False, marks=pytest.mark.smoke),
        pytest.param(9, 2, 65, 5, 7, torch.bfloat16, True, marks=pytest.mark.smoke),
        pytest.param(3, 17, 128, 3, 7, torch.float16, True, marks=pytest.mark.smoke),
        pytest.param(0, 2, 64, 5, 7, torch.bfloat16, False, marks=pytest.mark.smoke),
        pytest.param(64, 8, 128, 4096, 16, torch.bfloat16, False, marks=pytest.mark.full),
        pytest.param(64, 8, 128, 4096, 16, torch.float16, False, marks=pytest.mark.full),
        pytest.param(8192, 8, 128, 4096, 16, torch.bfloat16, False, marks=pytest.mark.full),
        pytest.param(4096, 4, 128, 4096, 16, torch.bfloat16, True, marks=pytest.mark.full),
        pytest.param(256, 8, 64, 4096, 16, torch.bfloat16, False, marks=pytest.mark.full),
        pytest.param(1024, 4, 128, 512, 32, torch.float16, True, marks=pytest.mark.full),
    ],
)
def test_scatter(n, h, d, npages, page, dtype, fp8):
    import torch_npu  # noqa: F401

    from tileops.attention import PagedKVCacheWriteFwdOp

    if not torch.npu.is_available():
        pytest.skip("no NPU available")
    device = _selected_device()
    rng = torch.Generator().manual_seed(20261110)
    k = torch.randn((n, h, d), generator=rng).to(dtype)
    v = torch.randn((n, h, d), generator=rng).to(dtype)
    slots = torch.randperm(npages * page, generator=rng)[:n]
    slots[1::5] = -1
    cache_dtype = torch.float8_e4m3fn if fp8 else dtype
    kp = torch.full((npages, page, h, d), 0.375).to(cache_dtype)
    vp = torch.full((npages, page, h, d), -0.625).to(cache_dtype)
    ks = torch.tensor([1.25]) if fp8 else None
    vs = torch.tensor([0.75]) if fp8 else None
    valid = slots >= 0
    expected = []
    for source, pool, scale in ((k, kp, ks), (v, vp, vs)):
        result = pool.clone()
        stored = source[valid]
        if fp8:
            stored = (stored.float() / scale[0]).clamp(-448, 448).to(cache_dtype)
        # CPU indexing of FP8 is not implemented in every PyTorch release.
        result.view(torch.uint8).view(npages * page, -1)[slots[valid]] = (
            stored.contiguous()
            .view(torch.uint8)
            .reshape(int(valid.sum()), h * d * pool.element_size())
        )
        expected.append(result)
    args = [x.to(device) if x is not None else None for x in (k, v, kp, vp, slots, ks, vs)]
    assert PagedKVCacheWriteFwdOp(target="ascend")(*args) is None
    torch.npu.synchronize(device)
    for actual, reference in zip(args[2:4], expected, strict=True):
        assert torch.equal(actual.cpu().view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 8])
def test_fp8_special_values(dtype, rows):
    import torch_npu  # noqa: F401

    from tileops._kernels.ascend.attention.paged_cache_write import build_batched

    if not torch.npu.is_available():
        pytest.skip("no NPU available")
    device = _selected_device()
    # Exactly representable values include E4M3 half-way ties and both signed NaNs.
    values = torch.tensor(
        [
            0.0, -0.0, 1.0625, 1.1875, -1.0625, 448, 449, -449,
            float("inf"), -float("inf"), float("nan"), -float("nan"),
            2**-10, 3 * 2**-10, 2**-9, -2**-10,
        ],
        dtype=dtype,
    ).repeat(rows * 32).reshape(rows, 512)
    pages = torch.full((rows + 2, 512), 0.375).to(torch.float8_e4m3fn)
    expected = pages.clone().view(torch.uint8)
    expected[1 : rows + 1] = (
        values.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    k, v = values.to(device), values.flip(1).to(device)
    kp, vp = pages.to(device), pages.to(device)
    slots = torch.arange(1, rows + 1, dtype=torch.int64, device=device)
    scale = torch.ones(1, device=device)
    program = build_batched(512, rows + 2, str(dtype).removeprefix("torch."), True, rows, 1)
    program(k, v, kp.view(torch.uint8), vp.view(torch.uint8), slots, scale, scale)
    torch.npu.synchronize(device)
    assert torch.equal(kp.cpu().view(torch.uint8), expected)
    expected[1 : rows + 1] = (
        values.flip(1).float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    assert torch.equal(vp.cpu().view(torch.uint8), expected)
