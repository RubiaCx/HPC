# SpGEMM CSR

## 数据集地址

https://sparse.tamu.edu/

## 定义

在 SpGEMM 中，其计算公式为 

$$ C = A\times B$$

其中，$A$为稀疏矩阵，其形状为$M\times K$，$B$为稠密矩阵，其形状为$K\times N$，$C$为稠密矩阵，其形状为$M\times N$。可以使用稠密矩阵乘法GEMM来实现，区别是$A$中有大量的零元。

只考虑以CSR格式存储。以如下稀疏矩阵为例，可以用三个数组来存储它（均从0开始编址）。

<div align="center">
  <img src="../images/CSR2.png">
</div>

```
PTR = [  0  2  4  7  8 ]
IDX = [  0  1  1  3  2  3  4  5 ]   
VAL = [ 10 20 30 40 50 60 70 80 ]
```

PTR 数组指出了哪些元素属于某一行。在上例中，PTR[0]=0，PTR[1]=2，其意思是在 IDX 和 VAL 数组中的第 [0,2) 个元素属于第 0 行。同样，PTR[1]=2，PTR[2]=4，其意思是在 IDX 和 VAL 数组中的第 [2,4) 个元素属于第 1 行。

IDX 数组指出了在该行上的具体位置，在上例中，我们通过读取 PTR 数组已经知道了 IDX 数组中的第 [0,2) 个元素属于第 0 行，通过读取 IDX[0]=0 IDX[1]=1，我们可以知道，第 0 行所拥有的两个元素在第 0 列和第 1 列。

VAL 数组指出了其具体数值，通过读取 VAL[0]=10 VAL[1]=20，可以知道，第 0 行的两个元素的数值是 10 和 20。

## 实现

```bash
nvcc spgemm_example.c -lcusparse
./a.out SparseMatrix/xxx.mtx
```

<div align="center">
  <img src="../images/CSR.png">
</div>

- 每个元素用`行偏移`和`（列号，数值）`来表示
    - 行偏移表示某一行的第一个元素在values里面的起始偏移位置
    - 第一行元素1是0偏移，第二行元素2是2偏移，第三行元素5是4偏移，第4行元素6是7偏移
    - 在行偏移的最后补上矩阵总的元素个数
        


## 参考
- 