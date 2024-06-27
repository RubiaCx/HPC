# 前缀和

## 问题描述

- 输入：输入一个数组`input[n]`
- 计算新数组`output[n]`，使得对于任意元素`output[i]`都满足：`output[i] = input[0] + input[1] + …… + X`
    - Inclusive: `X = input[i]`
    - Exclusive: `X = input[i-1]`

## Sequential Scan

### CPU

```c++
// LOOP version
void sequential_scan(float* input, float* output, int len){
    // temperary variable to store
    float temp = input[0];
    for (int i = 1; i < len; i++){
        output[i-1] = temp;
        temp += input[i];
    }
}

// recursive version
float recursive_scan(float* input, float* output, int len) {
    float temp;
    if (len == 1){
        temp = input[0];
        output[0] = temp;
        return temp;
    } else {
        temp = input[len-1] + recursive_scan(input, output, len-1);
        output[len-1] = temp;
        return temp;
    }
}
```