#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <memory.h>
#include <math.h>
float* read_param(const std::string& path) 
{
    std::ifstream file(path);
    std::vector<float> params;
    float param;
    while (file >> param) 
    {
        params.push_back(param);
    }

    float* ans = (float *)malloc(sizeof(float) * params.size());
    for (int i = 0; i < params.size(); i++)
    {
        ans[i] = params[i];
    }
    return ans;
}
int main(int argc, char *argv[])
{
    std::string dir = argv[1]; // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
    // cout << dir;

    // 读取模型参数
    auto conv1_weight = read_param(dir + "/conv1.weight.txt");
    float* test = read_param(dir + "/conv1.weight.txt");
    std::cout<<conv1_weight[1]<<" "<<test[1]<<std::endl;
}