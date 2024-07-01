# HPC学习笔记

## CUDA
- reduce
- prefix Sum
- GEMV
- GEMM
- SpMV CSR
  - CPU
  - GPU scalar
  - GPU vector
- SpGEMM CSR

```
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
```

- 用来将普通的单精度浮点数指针转换为指向 float4 类型的指针，并立即解引用该指针以获取指向的值的工具
- reinterpret_cast<float4 *>(&(pointer))：
  - reinterpret_cast：将 pointer 的地址转换为 float4* 类型。这意味着将开始把原本按单个 float 排列的数据视为 float4 类型的数组
  - &(pointer)：得到pointer 变量自身的地址
- [0]：使用 [0] 来解引用新类型的指针，即获取 float4* 指向的第一个 float4 实例

## Triton

## MLIR

## BANG C
