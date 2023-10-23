// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70
#include <fstream>
#include <ostream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <memory.h>
#include <math.h>

using DT = float;

//////////////////////////////////////////////////
// 其他工具函数
//////////////////////////////////////////////////
int toBigendian(int magic_number)
{
    return ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
           ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
}

template <typename T>
T max(T a, T b) 
{
    return a > b ? a : b; 
}

//////////////////////////////////////////////////
// 加载数据集和加载程序一模型参数（助教提供）
//////////////////////////////////////////////////
// 读取MNIST数据集
std::vector<std::vector<DT>> read_mnist_images(const std::string &path, int batch_size)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = toBigendian(magic_number);
    num_images = toBigendian(num_images);
    num_rows = toBigendian(num_rows);
    num_cols = toBigendian(num_cols);
    int batch_num = num_images / batch_size;
    int image_size = num_rows * num_cols * batch_size;

    std::vector<std::vector<DT>> images(batch_num, std::vector<DT>(image_size));

    for (int i = 0; i < num_images; ++i)
        for (int j = 0; j < image_size; ++j)
        {
            unsigned char pixel = 0;
            file.read((char *)&pixel, sizeof(pixel));
            images[i][j] = static_cast<DT>(pixel);

        }
    file.close();

    return images;
}

// 读取MNIST label数据集
std::vector<std::vector<int>> read_mnist_labels(const std::string &path, int batch_size)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0, num_items = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&num_items, sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = toBigendian(magic_number);
    num_items = toBigendian(num_items);
    int batch_num = num_items / batch_size;
    std::vector<std::vector<int>> labels(batch_num, std::vector<int>(batch_size));
    for (int i = 0; i < batch_num; ++i)
        for (int j = 0; j < batch_size; ++j)
        {
            unsigned char label = 0;
            file.read((char*)&label, 1);//sizeof(label));
            labels[i][j] = static_cast<int>(label);
        }
    file.close();

    return labels;
}
// 读取模型参数
std::vector<DT> read_param(const std::string &path)
{
    std::ifstream file(path);
    std::vector<DT> params;
    DT param;
    while (file >> param)
        params.push_back(param);
    file.close();
    return params;
}
DT *read_param_DT(const std::string &path) // verified
{
    std::ifstream file(path);
    std::vector<DT> params;
    DT param;
    while (file >> param)
        params.push_back(param);
    file.close();
    DT* ans = (DT*)malloc(sizeof(DT) * params.size());
    for (int i = 0; i < params.size(); i++)
        ans[i] = params[i];
    return ans;
}
//////////////////////////////////////////////////
// LeNet 5
//////////////////////////////////////////////////
class LeNet5
{
public:
    LeNet5(int batch = 1);
    ~LeNet5();
    virtual void load_parameters(std::string path); // for debug
    virtual void print_parameters();                // for debug
    virtual void predict(const DT* const image, int batch) = 0; // 每次传入一个图片！！
    virtual void classify(int *predict, int batch) = 0;

protected:
    void softmax(DT *input, int *output, int B, int size);
    // Internal parameter
    int batch = 1;
    int parameter_initialized = false;
    //  Model Parameters
    DT *conv1_weight; // [1][6][5][5]
    DT *conv1_bias;   // [6]

    DT *conv2_weight; // [6][16][5][5]
    DT *conv2_bias;   // [16]

    DT *fc1_weight; // [256][120] x [400][120]
    DT *fc1_bias;   // [120]

    DT *fc2_weight; // [120][84]
    DT *fc2_bias;   // [84]

    DT *fc3_weight; // [84][10]
    DT *fc3_bias;   // [10]
    // Feature Map
    DT *input;          
    DT *C1_feature_map; 
    DT *S2_feature_map; 
    DT *C3_feature_map; 
    DT *S4_feature_map; 
    DT *C5_layer;       
    DT *F6_layer;      
    DT *output;     
    // Layer and Feature map parameters
    //// INPUT
    int input_size = 28;
    int input_channel = 1; // 3;
    //// Convolutions
    int conv1_in_channel = 1; // 3;
    int conv1_out_channel = 6;
    int conv1_kernel_size = 5;
    int conv1_kernel_stride = 1;
    //// C1 feature map 6@28x28
    int C1_channel = conv1_out_channel;
    int C1_size = input_size - (conv1_kernel_size - 1); // 28
    //// Subsampling S2 feature map 6@14x14
    int S2_channel = C1_channel;
    int S2_size = C1_size / 2; // 14
    //// Convolutions
    int conv2_in_channel = conv1_out_channel;
    int conv2_out_channel = 16;
    int conv2_kernel_size = 5;
    int conv2_kernel_stride = 1;
    //// C3 feature map 16@10x10
    int C3_channel = conv2_out_channel;
    int C3_size = S2_size - (conv2_kernel_size - 1); // 10
    //// Subsampling  S4 feature map 16@5x5
    int S4_channel = C3_channel;
    int S4_size = C3_size / 2; // 5
    //// Fully Connection
    int fc1_in_channel = S4_channel * S4_size * S4_size;
    int fc1_out_channel = 120;
    //// C5 layer
    int C5_size = fc1_out_channel;
    //// Fully Connection
    int fc2_in_channel = fc1_out_channel;
    int fc2_out_channel = 84;
    //// F6 layer
    int F6_size = fc2_out_channel;
    //// Fully Connection
    int fc3_in_channel = fc2_out_channel;
    int fc3_out_channel = 10;
    //// output
    int output_size = fc3_out_channel;
};

LeNet5::LeNet5(int batch)
{
    // Internal variable
    this->batch = batch;
    this->conv1_weight = new DT[conv1_in_channel * conv1_out_channel * conv1_kernel_size * conv1_kernel_size];
    this->conv1_bias = new DT[conv1_out_channel];
    this->conv2_weight = new DT[conv2_in_channel * conv2_out_channel * conv2_kernel_size * conv2_kernel_size];
    this->conv2_bias = new DT[conv2_out_channel];
    this->fc1_weight = new DT[fc1_in_channel * fc1_out_channel];
    this->fc1_bias = new DT[fc1_out_channel];
    this->fc2_weight = new DT[fc2_in_channel * fc2_out_channel];
    this->fc2_bias = new DT[fc2_out_channel];
    this->fc3_weight = new DT[fc3_in_channel * fc3_out_channel];
    this->fc3_bias = new DT[fc3_out_channel];
    // Activation
    this->input = new DT[batch * input_channel * input_size * input_size];
    this->C1_feature_map = new DT[batch * C1_channel * C1_size * C1_size];
    this->S2_feature_map = new DT[batch * S2_channel * S2_size * S2_size];
    this->C3_feature_map = new DT[batch * C3_channel * C3_size * C3_size];
    this->S4_feature_map = new DT[batch * S4_channel * S4_size * S4_size];
    this->C5_layer = new DT[batch * C5_size];
    this->F6_layer = new DT[batch * F6_size];
    this->output = new DT[batch * output_size];
}

LeNet5::~LeNet5()
{
    // Free model parameters memories
    delete[] this->conv1_weight;
    delete[] this->conv1_bias;
    delete[] this->conv2_weight;
    delete[] this->conv2_bias;
    delete[] this->fc1_weight;
    delete[] this->fc1_bias;
    delete[] this->fc2_weight;
    delete[] this->fc2_bias;
    delete[] this->fc3_weight;
    delete[] this->fc3_bias;
    // Free activation memories
    delete[] this->input;
    delete[] this->C1_feature_map;
    delete[] this->S2_feature_map;
    delete[] this->C3_feature_map;
    delete[] this->S4_feature_map;
    delete[] this->C5_layer;
    delete[] this->F6_layer;
    delete[] this->output;
}

void LeNet5::load_parameters(std::string path)
{
    this->conv1_weight = read_param_DT(path + "/conv1.weight.txt");
    this->conv1_bias = read_param_DT(path + "/conv1.bias.txt");
    this->conv2_weight = read_param_DT(path + "/conv2.weight.txt");
    this->conv2_bias = read_param_DT(path + "/conv2.bias.txt");
    this->fc1_weight = read_param_DT(path + "/fc1.weight.txt");
    this->fc1_bias = read_param_DT(path + "/fc1.bias.txt");
    this->fc2_weight = read_param_DT(path + "/fc2.weight.txt");
    this->fc2_bias = read_param_DT(path + "/fc2.bias.txt");
    this->fc3_weight = read_param_DT(path + "/fc3.weight.txt");
    this->fc3_bias = read_param_DT(path + "/fc3.bias.txt");
}
void LeNet5::print_parameters()
{
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    // conv1.weight
    for (int c = 0; c < conv1_in_channel * conv1_out_channel; c++)
    {
        std::cout << "conv1.weight.c" << c + 1 << std::endl;
        for (int i = 0; i < conv1_kernel_size; i++)
        {
            for (int j = 0; j < conv1_kernel_size; j++)
            {
                std::cout << conv1_weight[c * (conv1_kernel_size * conv1_kernel_size) +
                                          i * conv1_kernel_size + j]
                          << " ";
            }
            std::cout << std::endl;
        }
    }
    // conv2.weight
    for (int c = 0; c < conv2_in_channel * conv2_out_channel; c++)
    {
        std::cout << "conv2.weight.c" << c + 1 << std::endl;
        for (int i = 0; i < conv2_kernel_size; i++)
        {
            for (int j = 0; j < conv2_kernel_size; j++)
            {
                std::cout << conv2_weight[c * (conv2_kernel_size * conv2_kernel_size) +
                                          i * conv2_kernel_size + j]
                          << " ";
            }
            std::cout << std::endl;
        }
    }
    // conv1.bias
    std::cout << "conv1.bias" << std::endl;
    for (int oc = 0; oc < conv1_out_channel; oc++)
    {
        std::cout << conv1_bias[oc] << " ";
    }
    std::cout << std::endl;
    // conv2.bias
    std::cout << "conv2.bias" << std::endl;
    for (int oc = 0; oc < conv2_out_channel; oc++)
    {
        std::cout << conv2_bias[oc] << " ";
    }
    std::cout << std::endl;
    // fc1.weight
    for (int oc = 0; oc < fc1_out_channel; oc++)
    {
        std::cout << "fc1.weight.out_channel" << oc + 1 << std::endl;
        for (int ic = 0; ic < fc1_in_channel; ic++)
        {
            std::cout << fc1_weight[oc * fc1_in_channel + ic] << " ";
        }
        std::cout << std::endl;
    }
    // fc2.weight
    for (int oc = 0; oc < fc2_out_channel; oc++)
    {
        std::cout << "fc2.weight.out_channel" << oc + 1 << std::endl;
        for (int ic = 0; ic < fc2_in_channel; ic++)
        {
            std::cout << fc2_weight[oc * fc2_in_channel + ic] << " ";
        }
        std::cout << std::endl;
    }
    // fc3.weight
    for (int oc = 0; oc < fc3_out_channel; oc++)
    {
        std::cout << "fc3.weight.out_channel" << oc + 1 << std::endl;
        for (int ic = 0; ic < fc3_in_channel; ic++)
        {
            std::cout << fc3_weight[oc * fc3_in_channel + ic] << " ";
        }
        std::cout << std::endl;
    }
    // fc1.bias
    std::cout << "fc1.bias" << std::endl;
    for (int oc = 0; oc < fc1_out_channel; oc++)
    {
        std::cout << fc1_bias[oc] << " ";
    }
    std::cout << std::endl;
    // fc2.bias
    std::cout << "fc2.bias" << std::endl;
    for (int oc = 0; oc < fc2_out_channel; oc++)
    {
        std::cout << fc2_bias[oc] << " ";
    }
    std::cout << std::endl;
    // fc3.bias
    std::cout << "fc3.bias" << std::endl;
    for (int oc = 0; oc < fc3_out_channel; oc++)
    {
        std::cout << fc3_bias[oc] << " ";
    }
    std::cout << std::endl;
}
void LeNet5::softmax(DT *input, int *output, int B, int size)
{
    for (int b = 0; b < B; b++)
    {
        // Initialize
        int max_idx = 0;
        DT max_val = std::exp(std::numeric_limits<DT>::lowest());
        // calcualte Z = sum_all(exp(x_i))
        DT Z = 0;
        for (int i = 0; i < size; i++)
            Z += std::exp(input[b * size + i]);
        // Softmax
        for (int i = 0; i < size; i++)
        {
            input[b * size + i] = std::exp(input[b * size + i]) / Z;
            if (input[i] - max_val > std::numeric_limits<DT>::epsilon())
            {
                max_val = input[b * size + i];
                max_idx = i;
            }
        }
        output[b] = max_idx;
    }
}

//////////////////////////////////////////////////
// LeNet CPU
//////////////////////////////////////////////////

class LeNet5_cpu : public LeNet5
{
public:
    LeNet5_cpu(int batch) : LeNet5(batch){};
    ~LeNet5_cpu(){};
    void predict(const DT* const image, int batch) override;
    void print_parameters() override { LeNet5::print_parameters(); };
    void classify(int *predict, int batch) override;

private:
    void relu(DT *feature_map, int size);
    void normalize(const DT* const image, DT *input);
    void conv(DT *input, DT *output, DT *weight, DT *bias, int B, int H, int W, int IC, int OC, int K, int S);
    void pool(DT *input, DT *output, int B, int C, int H, int W);
    void fc(DT *input, DT *output, DT *weight, DT *bias, int B, int IC, int OC);
};
// ReLu层
void LeNet5_cpu::relu(DT* feature_map, int size)
{
    DT zero = 0.0;
    for (int i = 0; i < size; i++)
        feature_map[i] = std::max(feature_map[i], zero);
}
/**
 * 标准化 使用zero-mean normalization x'=(x-u)/δ
 */
void LeNet5_cpu::normalize(const DT* const image, DT *input)
{
    // Initialize variables
    DT max_int = 255.0L; // already done
    DT mean = 0.5L;
    DT var = 0.5L;
    // Normalize
    for (int i = 0; i < batch * input_channel * input_size * input_size; i++)
    {
        input[i] = image[i] / max_int;      // transforms.ToTensor();
        input[i] = (input[i] - mean) / var; // transforms.Normalize();
    }
}
/**
 * Fully Connected
 */
void LeNet5_cpu::fc(DT *input, DT *output, DT *weight, DT *bias, int B, int IC, int OC)
{
    for (int b = 0; b < B; b++) // batch size
        for (int oc = 0; oc < OC; oc++) // output channel
        {
            output[b * OC + oc] = bias[oc];
            for (int ic = 0; ic < IC; ic++)
            {
                output[b * OC + oc] += weight[oc * IC + ic] * input[b * IC + ic];
            }
        }
}
/**
 * Convolution
 * input: B x I x H x W
 * output: B x O x H_OUT x W_OUT
 * kernel: O x I x K x K
 */
void LeNet5_cpu::conv(DT *input, DT *output, DT *weight, DT *bias, int B, int H, int W, int IC, int OC, int K, int S)
{
    // Initialize variable
    int H_OUT = (H - K)/S + 1; 
    int W_OUT = (W - K)/S + 1;
    for (int b = 0; b < B; b++) // mini-batch
        for (int oc = 0; oc < OC; oc++) // Output Channel
            for (int h = 0; h < H_OUT; h++) // Height
                for (int w = 0; w < W_OUT; w++) // Width
                { 
                    int output_index = b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
                    output[output_index] = bias[oc];
                    for (int ic = 0; ic < IC; ic++)
                    {
                        int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
                        int kernel_base = oc * (IC * K * K) + ic * (K * K);
                        for (int kh = 0; kh < K; kh++)
                            for (int kw = 0; kw < K; kw++)
                            {
                                DT val = input[input_base + kh * (W) + kw] * weight[kernel_base + kh * (K) + kw];
                                output[output_index] += val;
                            }
                    }
                }
}
/**
 * Maxpooling
 */
void LeNet5_cpu::pool(DT *input, DT *output, int B, int C, int H, int W)
{
    // Initilaize variable
    int scale = 2;
    int H_OUT = H / scale;
    int W_OUT = W / scale;
    // Max Pooling
    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h += 2) // 核大小写死为2x2
                for (int w = 0; w < W; w += 2)
                {
                    // Init values
                    int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
                    int max_sh = 0;
                    int max_sw = 0;
                    DT max_val = std::numeric_limits<DT>::lowest();
                    // Find maximum
                    for (int sh = 0; sh < scale; sh++)
                        for (int sw = 0; sw < scale; sw++)
                        {
                            DT val = input[input_base + sh * (W) + sw];
                            if (val - max_val > std::numeric_limits<DT>::epsilon())
                            {
                                max_val = val;
                                max_sh = sh;
                                max_sw = sw;
                            }
                        }
                    // Set output with max value
                    int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) + (h / 2) * W_OUT + (w / 2);
                    output[output_index] = max_val;
                }
}

void LeNet5_cpu::predict(const DT* const image, int batch)
{
    // ToTensor and Normalize
    normalize(image, input);
    // Conv2d
    conv(input, C1_feature_map, conv1_weight, conv1_bias, batch, input_size, input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size, conv1_kernel_stride);
    relu(C1_feature_map, batch * C1_channel * C1_size * C1_size);
    // MaxPool2d
    pool(C1_feature_map, S2_feature_map, batch, C1_channel, C1_size, C1_size);
    // Conv2d
    conv(S2_feature_map, C3_feature_map, conv2_weight, conv2_bias, batch, S2_size, S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size, conv2_kernel_stride);
    relu(C3_feature_map, batch * C3_channel * C3_size * C3_size);
    // MaxPool2d
    pool(C3_feature_map, S4_feature_map, batch, C3_channel, C3_size, C3_size);
    // Linear
    fc(S4_feature_map, C5_layer, fc1_weight, fc1_bias, batch, fc1_in_channel, fc1_out_channel);
    relu(C5_layer, batch * C5_size);
    // Linear
    fc(C5_layer, F6_layer, fc2_weight, fc2_bias, batch, fc2_in_channel, fc2_out_channel);
    relu(F6_layer, batch * F6_size);
    // Linear
    fc(F6_layer, output, fc3_weight, fc3_bias, batch, fc3_in_channel, fc3_out_channel);
}

void LeNet5_cpu::classify(int *predict, int batch)
{
    // print_parameters();
    // softmax(output, predict, batch, output_size);
    for (int i = 0; i < batch; i++)
    {
        double predication = output[i*batch];
        int idx = 0;
        for(int j = 0; j < output_size; j++)
        {
            if(predication < output[i*batch + j])
            {
                predication = output[i*batch + j];
                idx = j;
            }
        }
        predict[i] = idx;
    }
    // std::cout<<"OUTPUT"<<std::endl;
    // for(int i =0;i<10;i++)
    // {
    //     std::cout<<output[i]<<" ";
    // }
    // std::cout<<std::endl;
    // std::cout<<predict[0]<<std::endl;
}

int main(int argc, char *argv[])
{
    std::string dir = argv[1]; // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
    // cout << dir;
    int batch = 2;

    auto image_set = read_mnist_images(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte", batch);
    auto label_set = read_mnist_labels(dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte", batch); 
    // 打印每一个标签，仅用于调试！verified
    // auto labels = read_mnist_labels(dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    // for (int i=0; i<10; i++)
    // {
    //     std::cout << fm.label_set[i][0] << "-"<<labels[i]<<" ";
    // }
    // std::cout << std::endl;
    // auto images = read_mnist_images(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    // for (int i=0; i<2; i++)
    // for (int j=0; j<fm.image_col; j++)
    // {
    //     std::cout << fm.image_set[i][j] << "-"<<images[i][j]<<" ";
    // }
    // std::cout << std::endl;

    int row = image_row;
    int** predict_cpu = new int*[label_set.size()];
    for (int i = 0; i <  label_set.size(); i++) 
        predict_cpu[i] = new int[label_set[0].size()]; 

    LeNet5_cpu* net_cpu = new LeNet5_cpu(batch);
    net_cpu->load_parameters(dir);

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < row; r++)
    {
        // std::cout<<fm.image_set[r]<<std::endl;
        net_cpu->predict(image_set[r], batch);
        net_cpu->classify(predict_cpu[r], batch); 
    }
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    int sum = 0;
    int correct = 0;
    for (int i = 0; i < label_set.size(); i++) 
        for(int j = 0; j < label_set[0].size(); j++)
        {
            if(predict_cpu[i][j] == label_set[i][j])
            {
                correct++;
            }
            sum++;
        }

    std::chrono::duration<DT> diff = end - start;
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << (double)correct/sum << std::endl;

    // std::ofstream outfilep("prediction.txt");
    // for (int i = 0; i < fm.label_row; i++) 
    // {
    //     for (int j = 0; j < fm.label_col; j++) 
    //     {
    //         outfilep << predict_cpu[i][j] << " ";
    //     }
    //     outfilep << "\n";
    // }
    // outfilep.close();

    // std::ofstream outfilel("labels.txt");
    // for (int i = 0; i < fm.label_row; i++) 
    // {
    //     for (int j = 0; j < fm.label_col; j++) 
    //     {
    //         outfilel << fm.label_set[i][j] << " ";
    //     }
    //     outfilel << "\n";
    // }
    // outfilel.close();

    return 0;
}