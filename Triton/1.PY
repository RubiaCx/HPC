@triton.jit()
def mac_loop(A, B, C,
             M, N, K,
             locks, # 用于在不同 GPU 线程之间同步的 locks 数组
             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, # 各个矩阵维度的步长
             iters_per_tile, # 一个 tile 中的迭代次数
             start_iter, end_iter,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             ACC_TYPE: tl.constexpr, # 用于计算的累加器类型，保证足够的数值精度
             GROUP_M: tl.constexpr):
    # 确定当前块的位置
    tile_id = start_iter // iters_per_tile
    if GROUP_M > 0:
        pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    else:
        pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    # 计算内存地址
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak) + BLOCK_K * stride_ak * (start_iter % iters_per_tile)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn) + BLOCK_K * stride_bk * (start_iter % iters_per_tile)
    # 初始化 BLOCK_M x BLOCK_N 的累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    # 执行矩阵乘法
    for current_iter in range(start_iter, end_iter):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    # 写回结果
    # 最后一次迭代处理当前块，则将累加器中的结果写回到矩阵 C 的相应位置
    if end_iter % iters_per_tile == 0: # last iteration of the tile always happens before its start on another SM
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)  # compute inside the if/else to avoid spilling!
        tl.store(C_, acc)
        if start_iter % iters_per_tile != 0:  # only if tile has been partially processed
            tl.atomic_xchg(locks + tile_id, 1)
    # 块已经被部分处理，使用原子交换（tl.atomic_xchg）来更新锁状态，确保结果的一致性
    else:
        while tl.atomic_cas(locks + tile_id, 1, 1) != 1:
            pass
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)  # compute inside the if/else to avoid spilling!
        tl.atomic_add(C_, acc)