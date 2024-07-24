"""
Stream K GEMM with Block Pointers
=====================
Stream K is a approach that aims to resovle quantization inefficiency in GPU work load for GEMM problems.
This script implements a Stream K GEMM with block pointers to achieve better hardware utilization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def calculate_pids(tile_id,
                 # Matrix dimensions
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 # Meta-parameters
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                 GROUP_SIZE_M: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_SIZE_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(GROUP_SIZE_M, grid_m - group_id * GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


# Multiply-accumulate loop in GEMM Stream K tiles
@triton.jit
def perform_multiply_accumulate(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,  #
        # Stream-K parameters
        iters_per_tile, start_iter, end_iter,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):

    tile_id = start_iter // iters_per_tile
    remain_iters = start_iter % iters_per_tile
    # Assume GROUP_M > 0
    # pid swizzle to get better L2 cache performance
    pid_m, pid_n = calculate_pids(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)

    a_ptr += BLOCK_SIZE_K * stride_ak * remain_iters
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_ptr += BLOCK_SIZE_K * stride_bk * remain_iters
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(start_iter, end_iter):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    if remain_iters == 0 and end_iter % iters_per_tile == 0:
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
        tl.store(c_block_ptr, acc, boundary_check=(0, 1))
    else:
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptr_ = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.atomic_add(c_ptr_, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4,
                      num_warps=32),
    ],
    key=['M', 'N', 'K'],
)


@triton.jit
def first_wave(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Stream-K parameters
        full_tiles, partial_tiles, iters_per_tile,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):

    pid = tl.program_id(axis=0)
    start_iter = pid * full_tiles + tl.minimum(pid, partial_tiles)
    last_iter = (pid + 1) * full_tiles + tl.minimum(pid + 1, partial_tiles)

    while start_iter < last_iter:
        end_iter = start_iter + (iters_per_tile - start_iter % iters_per_tile)
        end_iter = tl.minimum(end_iter, last_iter)
        perform_multiply_accumulate(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 iters_per_tile, start_iter, end_iter, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)

        start_iter = end_iter


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4,
                      num_warps=32),
    ],
    key=['M', 'N', 'K'],
)

@triton.jit
def full_tiles(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Stream-K parameters
        streamk_tiles,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):

    tile_id = tl.program_id(axis=0) + streamk_tiles
    # Assume GROUP_M > 0
    pid_m, pid_n = calculate_pids(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.
def matmul(a: torch.Tensor, b: torch.Tensor):
    num_xe_core = torch.xpu.get_device_capability(0)['gpu_subslice_count']
    streamk_programs = num_xe_core

    # TODO: use autotune config instread of hardcoding
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape

    num_block_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_block_n = triton.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = triton.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_block_m * num_block_n

    # Two-tile SK + DP
    streamk_tiles = total_tiles % streamk_programs
    if total_tiles - streamk_tiles > streamk_programs:  # (total_tiles // total_programs > 1)
        streamk_tiles += streamk_programs

    blocking_tiles = total_tiles - streamk_tiles
    streamk_iters = streamk_tiles * iters_per_tile

    streamk_full_tiles = streamk_iters // streamk_programs
    streamk_partial_tiles = streamk_iters % streamk_programs

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    first_wave[(streamk_programs, )](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        streamk_full_tiles, streamk_partial_tiles, iters_per_tile,  #
        threads_per_warp=16)
    full_tiles[(blocking_tiles, )](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        streamk_tiles, threads_per_warp=16)

    return c



torch.manual_seed(0)
for dtype in [
        torch.float16,
]:
    a = torch.randn((512, 512), device='xpu', dtype=dtype)
    b = torch.randn((512, 512), device='xpu', dtype=dtype)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b).to(torch.float32)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")

    # Note: the torch.matmul and Triton implementations uses different
    # algorithms so we need to adjust tolerance.
    rtol = 1e-3
    if torch.allclose(triton_output, torch_output, atol=1e-3, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        exit("❌ Triton and Torch differ")