from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# precision-dump experiment switch (fixed at process start). When set, the
# DSV4 KV nope segment is stored as bf16 instead of fp8+ue8m0 scale, making the
# store->load roundtrip lossless. Layout constants below branch on it and every
# consumer (pool / quant / SetKAndS) imports them so the layout stays coherent.
BF16_KV = envs.SGLANG_DSV4_BF16_KV.get()

DIM_NOPE = 448
DIM_ROPE = 64
DIM_TOTAL = DIM_NOPE + DIM_ROPE  # 512
TILE_SIZE = 64  # one nope scale tile = 64 fp8 values
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE  # 7

if BF16_KV:
    # per-token: 448 bf16 nope + 64 bf16 rope = 512 contiguous bf16 (1024 bytes).
    NOPE_ROPE_BYTES = DIM_TOTAL * 2  # 1024
    PADDED_SCALE_PER_TOKEN = 0  # no scale segment
else:
    # per-token: 448 fp8 nope + 64 bf16 rope (= 576 bytes) + 7 ue8m0 scales
    #            padded to 8 bytes. per-page padded up to a multiple of 576.
    NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2  # 576
    PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1  # 8


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Read the DeepSeek v4 paged KV cache for a list of token IDs.

    Args:
        quant_k_cache: (num_pages, bytes_per_page) uint8/bf16.
        page_table_1_flattened: (num_tokens,) int — token IDs into the cache.
        page_size: number of tokens per page.
        out: optional (num_tokens, 1, DIM_TOTAL) bf16 destination.

    Returns:
        (num_tokens, 1, DIM_TOTAL) bfloat16.
    """
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)

    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]

    if out is None:
        out = torch.empty(
            (num_tokens, 1, DIM_TOTAL),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, DIM_TOTAL)
        assert out.dtype == torch.bfloat16

    if BF16_KV:
        buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
        _dequantize_k_cache_paged_bf16_kernel[(num_tokens,)](
            out,
            buf_bf16,
            page_table_1_flattened,
            out.stride(0),
            BYTES_PER_PAGE=bytes_per_page,
            PAGE_SIZE=page_size,
            DIM_TOTAL=DIM_TOTAL,
            NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        )
        return out

    s_offset_bytes = page_size * NOPE_ROPE_BYTES
    buf_fp8 = quant_k_cache_u8.view(fp8_dtype).reshape(-1)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    _dequantize_k_cache_paged_fp8_kernel[(num_tokens,)](
        out,
        buf_fp8,
        buf_bf16,
        buf_uint8,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=s_offset_bytes,
    )
    return out


@triton.jit
def _dequantize_k_cache_paged_bf16_kernel(
    output_ptr,
    buf_bf16_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_TOTAL: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
):
    # One program per token: copy the 512 contiguous bf16 of that token's slot.
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    token_byte_base = page_idx * BYTES_PER_PAGE + in_page * NOPE_ROPE_BYTES
    token_bf16_base = token_byte_base // 2

    offs = tl.arange(0, DIM_TOTAL)
    vals = tl.load(buf_bf16_ptr + token_bf16_base + offs)
    tl.store(output_ptr + token_id * output_stride_0 + offs, vals)


@triton.jit
def _dequantize_k_cache_paged_fp8_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # One program per token: load page_table[token_id] once and emit all
    # NUM_SCALE_TILES nope tiles + rope tail via tl.static_range.
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)


def dequantize_k_cache_paged_ref(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Pure-torch reference for :func:`dequantize_k_cache_paged`."""
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    u8 = quant_k_cache.view(torch.uint8)
    bytes_per_page = u8.shape[-1]
    device = quant_k_cache.device

    loc = page_table_1_flattened.to(torch.int64)
    page_idx = loc // page_size
    in_page = loc % page_size

    if BF16_KV:
        flat_bf16 = u8.view(torch.bfloat16).reshape(-1)
        token_bf16_base = (page_idx * bytes_per_page + in_page * NOPE_ROPE_BYTES) // 2
        idx = token_bf16_base[:, None] + torch.arange(DIM_TOTAL, device=device)[None, :]
        out = torch.empty(
            (loc.shape[0], 1, DIM_TOTAL), dtype=torch.bfloat16, device=device
        )
        out[:, 0, :] = flat_bf16[idx]
        return out

    s_offset_bytes = page_size * NOPE_ROPE_BYTES
    flat_u8 = u8.reshape(-1)
    flat_fp8 = u8.view(fp8_dtype).reshape(-1)
    flat_bf16 = u8.view(torch.bfloat16).reshape(-1)

    page_byte_base = page_idx * bytes_per_page
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + s_offset_bytes + in_page * PADDED_SCALE_PER_TOKEN
    )

    nope_byte = (
        token_data_base[:, None] + torch.arange(DIM_NOPE, device=device)[None, :]
    )
    nope_fp8 = flat_fp8[nope_byte].to(torch.float32)
    scale_byte = (
        token_scale_base[:, None]
        + torch.arange(NUM_SCALE_TILES, device=device)[None, :]
    )
    scale_u8 = flat_u8[scale_byte].to(torch.int32)
    scale_pow2 = torch.exp2((scale_u8 - 127).to(torch.float32))
    scale_pow2 = torch.where(
        scale_pow2 < (2.0**-126), torch.zeros_like(scale_pow2), scale_pow2
    )
    scale_full = scale_pow2.repeat_interleave(TILE_SIZE, dim=1)
    nope = nope_fp8 * scale_full

    rope_bf16_base = (token_data_base + DIM_NOPE) // 2
    rope_idx = rope_bf16_base[:, None] + torch.arange(DIM_ROPE, device=device)[None, :]
    rope = flat_bf16[rope_idx]

    out = torch.empty(
        (loc.shape[0], 1, DIM_TOTAL), dtype=torch.bfloat16, device=device
    )
    out[:, 0, :DIM_NOPE] = nope.to(torch.bfloat16)
    out[:, 0, DIM_NOPE:] = rope
    return out


if __name__ == "__main__":
    assert torch.cuda.is_available(), "this self-test needs a CUDA device"
    torch.manual_seed(0)
    device = "cuda"

    page_size = 64
    num_pages = 8
    num_tokens = 333
    raw_bytes = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = (
        (raw_bytes + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
    ) * NOPE_ROPE_BYTES

    quant_k_cache = torch.randint(
        0, 256, (num_pages, bytes_per_page), dtype=torch.uint8, device=device
    )
    page_table = torch.randint(
        0, num_pages * page_size, (num_tokens,), dtype=torch.int32, device=device
    )

    out_kernel = dequantize_k_cache_paged(quant_k_cache, page_table, page_size)
    out_ref = dequantize_k_cache_paged_ref(quant_k_cache, page_table, page_size)

    torch.testing.assert_close(out_kernel, out_ref, atol=0, rtol=0, equal_nan=True)
    print(
        f"OK ({'bf16' if BF16_KV else 'fp8'}): kernel matches torch ref for "
        f"{num_tokens} tokens (page_size={page_size}, bytes_per_page={bytes_per_page})"
    )
