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

// Conv params in constant memory
__constant__ float d_conv1_weight[1*6*5*5];   
__constant__ float d_conv2_weight[6*16*5*5];  
__constant__ float d_conv1_bias[6];    
__constant__ float d_conv2_bias[16];   


/**
 * Wrapper to catch CUDA errors.
 * For debugging only.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
std::vector<std::vector<float>> read_mnist_images(const std::string &path)
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

    int image_size = num_rows * num_cols;
    std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));

    for (int i = 0; i < num_images; ++i)
        for (int j = 0; j < image_size; ++j)
        {
            unsigned char pixel = 0;
            file.read((char *)&pixel, sizeof(pixel));
            images[i][j] = static_cast<float>(pixel) / 255.0f;

        }
    file.close();

    return images;
}

// 读取MNIST label数据集
std::vector<int> read_mnist_labels(const std::string &path)
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

    std::vector<int> labels(num_items);
    for (int i = 0; i < num_items; ++i)
    {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }
    file.close();

    return labels;
}
// 读取模型参数
std::vector<float> read_param(const std::string &path)
{
    std::ifstream file(path);
    std::vector<float> params;
    float param;
    while (file >> param)
        params.push_back(param);
    file.close();
    return params;
}
float *read_param_float(const std::string &path) // verified
{
    std::ifstream file(path);
    std::vector<float> params;
    float param;
    while (file >> param)
        params.push_back(param);
    file.close();
    float* ans = (float*)malloc(sizeof(float) * params.size());
    for (int i = 0; i < params.size(); i++)
        ans[i] = params[i];
    return ans;
}
//////////////////////////////////////////////////
// FashionMNIST
//////////////////////////////////////////////////
class FashionMNIST
{
public:
    float **image_set;
    std::string image_path;
    std::string label_path;
    int batch_size;
    int **label_set;
    int image_row, image_col;
    int label_row, label_col;
    FashionMNIST(std::string image_path, std::string label_path, int batch_size = 1);
    ~FashionMNIST();
    void load_data();
    void read_mnist_images(std::string path, int batch_size);
    void read_mnist_labels(std::string path, int batch_size);
};
FashionMNIST::FashionMNIST(std::string image_path, std::string label_path, int batch_size) 
{
    this->image_path = image_path;
    this->label_path = label_path;
    this->batch_size = batch_size;
}
FashionMNIST::~FashionMNIST()
{
    for (int i = 0; i < image_row; i++)
        delete[] image_set[i];
    delete[] image_set;
    for (int i = 0; i < label_row; i++)
        delete[] label_set[i];
    delete[] label_set;
}
void FashionMNIST::load_data() 
{
    read_mnist_images(image_path, batch_size);
    read_mnist_labels(label_path, batch_size); 
}
void FashionMNIST::read_mnist_images(std::string path, int batch_size)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cout << "Cannot open file!" << std::endl;
        return ;
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&num_images, sizeof(num_images));
    file.read((char *)&num_rows, sizeof(num_rows));
    file.read((char *)&num_cols, sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = toBigendian(magic_number);
    num_images = toBigendian(num_images);
    num_rows = toBigendian(num_rows);
    num_cols = toBigendian(num_cols);
    int batch_num = num_images / batch_size;
    int image_size = num_rows * num_cols * batch_size;
    float** image_array = new float*[batch_num];
    for (int i = 0; i < num_images; i++) 
    {
        image_array[i] = new float[image_size];
    }

    for (int i = 0; i < batch_num; ++i)
        for (int j = 0; j < image_size; ++j)
        {
            unsigned char pixel = 0;
            file.read((char *)&pixel, sizeof(pixel));
            image_array[i][j] = static_cast<float>(pixel); 
        }
    file.close();
    this->image_row = batch_num;
    this->image_col = image_size;
    this->image_set = image_array;  // 像素的存储顺序: batch -> image -> row -> column
                                    // 存储索引: image[i][batch * IMG_HEIGHT * IMG_WIfloatH + row * IMG_WIfloatH + col]
}
void FashionMNIST::read_mnist_labels(std::string path, int batch_size)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cout << "Cannot open file!" << std::endl;
        return ;
    }

    int magic_number = 0, num_items = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    file.read((char *)&num_items, sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = toBigendian(magic_number);
    num_items = toBigendian(num_items);
    
    int batch_num = num_items / batch_size;

    int** labels = new int*[batch_num];
    for (int i = 0; i < batch_num; i++) 
        labels[i] = new int[batch_size]; 
    int idx = 0;
    for (int n = 0; n < batch_num; n++) 
        for (int i = 0; i < batch_size; i++) 
        {
            // 读取一个label
            unsigned char label = 0;
            file.read((char*)&label, 1);
            labels[n][i] = static_cast<int>(label);
            idx++;
            if (idx >= num_items) 
                break; 
        }

    file.close();
    this->label_row = batch_num;
    this->label_col = batch_size;
    this->label_set = labels;
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
    virtual void predict(const float* const image, int batch) = 0; // 每次传入一个图片！！
    virtual void classify(int *predict, int batch) = 0;

protected:
    void softmax(float *input, int *output, int B, int size);
    // Internal parameter
    int batch = 1;
    int parameter_initialized = false;
    //  Model Parameters
    float *conv1_weight; // [1][6][5][5]
    float *conv1_bias;   // [6]

    float *conv2_weight; // [6][16][5][5]
    float *conv2_bias;   // [16]

    float *fc1_weight; // [256][120] x [400][120]
    float *fc1_bias;   // [120]

    float *fc2_weight; // [120][84]
    float *fc2_bias;   // [84]

    float *fc3_weight; // [84][10]
    float *fc3_bias;   // [10]
    // Feature Map
    float *input;          
    float *C1_feature_map; 
    float *S2_feature_map; 
    float *C3_feature_map; 
    float *S4_feature_map; 
    float *C5_layer;       
    float *F6_layer;      
    float *output;     
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
    this->conv1_weight = new float[conv1_in_channel * conv1_out_channel * conv1_kernel_size * conv1_kernel_size];
    this->conv1_bias = new float[conv1_out_channel];
    this->conv2_weight = new float[conv2_in_channel * conv2_out_channel * conv2_kernel_size * conv2_kernel_size];
    this->conv2_bias = new float[conv2_out_channel];
    this->fc1_weight = new float[fc1_in_channel * fc1_out_channel];
    this->fc1_bias = new float[fc1_out_channel];
    this->fc2_weight = new float[fc2_in_channel * fc2_out_channel];
    this->fc2_bias = new float[fc2_out_channel];
    this->fc3_weight = new float[fc3_in_channel * fc3_out_channel];
    this->fc3_bias = new float[fc3_out_channel];
    // Activation
    this->input = new float[batch * input_channel * input_size * input_size];
    this->C1_feature_map = new float[batch * C1_channel * C1_size * C1_size];
    this->S2_feature_map = new float[batch * S2_channel * S2_size * S2_size];
    this->C3_feature_map = new float[batch * C3_channel * C3_size * C3_size];
    this->S4_feature_map = new float[batch * S4_channel * S4_size * S4_size];
    this->C5_layer = new float[batch * C5_size];
    this->F6_layer = new float[batch * F6_size];
    this->output = new float[batch * output_size];
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
    this->conv1_weight = read_param_float(path + "/conv1.weight.txt");
    this->conv1_bias = read_param_float(path + "/conv1.bias.txt");
    this->conv2_weight = read_param_float(path + "/conv2.weight.txt");
    this->conv2_bias = read_param_float(path + "/conv2.bias.txt");
    this->fc1_weight = read_param_float(path + "/fc1.weight.txt");
    this->fc1_bias = read_param_float(path + "/fc1.bias.txt");
    this->fc2_weight = read_param_float(path + "/fc2.weight.txt");
    this->fc2_bias = read_param_float(path + "/fc2.bias.txt");
    this->fc3_weight = read_param_float(path + "/fc3.weight.txt");
    this->fc3_bias = read_param_float(path + "/fc3.bias.txt");
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
void LeNet5::softmax(float *input, int *output, int B, int size)
{
    for (int b = 0; b < B; b++)
    {
        // Initialize
        int max_idx = 0;
        float max_val = std::exp(std::numeric_limits<float>::lowest());
        // calcualte Z = sum_all(exp(x_i))
        float Z = 0;
        for (int i = 0; i < size; i++)
            Z += std::exp(input[b * size + i]);
        // Softmax
        for (int i = 0; i < size; i++)
        {
            input[b * size + i] = std::exp(input[b * size + i]) / Z;
            if (input[i] - max_val > std::numeric_limits<float>::epsilon())
            {
                max_val = input[b * size + i];
                max_idx = i;
            }
        }
        output[b] = max_idx;
    }
}

//////////////////////////////////////////////////
// LeNet CUDA
//////////////////////////////////////////////////
class LeNet5_cuda : public LeNet5
{
public:
    LeNet5_cuda(int batch = 1);
    ~LeNet5_cuda();
    void load_parameters(std::string value_path) override { LeNet5::load_parameters(value_path); };
    void print_parameters() override { LeNet5::print_parameters(); };
    void prepare_device_memory(float* image); 
    void predict(int batch);
    void predict(const float* const image, int batch) override {predict(batch);};
    void classify(int *predict, int batch) override;
private:
    //////////////////////////////////////////////////
    // CPU Fallbacks
    //////////////////////////////////////////////////
    void cpu_relu(float *feature_map, int size);
    void cpu_normalize(const float* const image, float *input);
    void cpu_conv(float *input, float *output, float *weight, float *bias, int B, int H, int W, int IC, int OC, int K, int S);
    void cpu_pool(float *input, float *output, int B, int C, int H, int W);
    void cpu_fc(float *input, float *output, float *weight, float *bias, int B, int IC, int OC);
    // void cpu_softmax(float *input, int *output, int B, int size);
private:
    float* d_fc1_weight;    
    float* d_fc2_weight;     
    float* d_fc3_weight;    
    float* d_fc1_bias;       
    float* d_fc2_bias;     
    float* d_fc3_bias;     
    //////////////////////////////////////////////////
    // Device Feature Maps
    //////////////////////////////////////////////////
    float* d_image;        
    float* d_input;         
    float* d_C1_feature_map; 
    float* d_S2_feature_map; 
    float* d_C3_feature_map; 
    float* d_S4_feature_map; 
    float* d_C5_layer;      
    float* d_F6_layer;      
    float* d_output;        
    int*   d_predict_cuda;  

    // Float host params 
    float* f_conv1_weight;  
    float* f_conv2_weight;  
    float* f_conv1_bias;   
    float* f_conv2_bias;    
    float* f_fc1_weight;  
    float* f_fc2_weight;  
    float* f_fc3_weight;  
    float* f_fc1_bias;    
    float* f_fc2_bias;       
    float* f_fc3_bias;      
    float* f_output;       
    // For unrolled convolution
    float* d_input_unrolled;
    float* d_conv1_weight_unrolled;
    float* d_conv1_bias_unrolled; // is this necessary?
    float* d_S2_feature_map_unrolled;
    float* d_conv2_weight_unrolled;
    float* d_conv2_bias_unrolled;
};

// ReLu层
void LeNet5_cuda::cpu_relu(float* feature_map, int size)
{
    float zero = 0.0;
    for (int i = 0; i < size; i++)
        feature_map[i] = std::max(feature_map[i], zero);
}
// 标准化 使用zero-mean normalization x'=(x-u)/δ
void LeNet5_cuda::cpu_normalize(const float* const image, float *input)
{
    // Initialize variables
    float max_int = 255.0L; // already done
    float mean = 0.5L;
    float var = 0.5L;
    // Normalize
    for (int i = 0; i < batch * input_channel * input_size * input_size; i++)
    {
        input[i] = image[i] / max_int;      // transforms.ToTensor();
        input[i] = (input[i] - mean) / var; // transforms.Normalize();
    }
}
// Fully Connected
void LeNet5_cuda::cpu_fc(float *input, float *output, float *weight, float *bias, int B, int IC, int OC)
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
// Convolution
void LeNet5_cuda::cpu_conv(float *input, float *output, float *weight, float *bias, int B, int H, int W, int IC, int OC, int K, int S) // S = 1
{
    // Initialize variable
    int H_OUT = (H - K)/S + 1; 
    int W_OUT = (W - K)/S + 1;
    // Convolution
    // input: B x I x H x W
    // output: B x O x H_OUT x W_OUT
    // kernel: O x I x K x K
    for (int b = 0; b < B; b++) // mini-batch
        for (int oc = 0; oc < OC; oc++) // Output Channel
            for (int h = 0; h < H_OUT; h++) // Height
                for (int w = 0; w < W_OUT; w++) // Wifloath
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
                                float val = input[input_base + kh * (W) + kw] * weight[kernel_base + kh * (K) + kw];
                                output[output_index] += val;
                            }
                    }
                }
}
// Maxpooling
void LeNet5_cuda::cpu_pool(float *input, float *output, int B, int C, int H, int W)
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
                    float max_val = std::numeric_limits<float>::lowest();
                    // Find maximum
                    for (int sh = 0; sh < scale; sh++)
                        for (int sw = 0; sw < scale; sw++)
                        {
                            float val = input[input_base + sh * (W) + sw];
                            if (val - max_val > std::numeric_limits<float>::epsilon())
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

LeNet5_cuda::LeNet5_cuda(int batch) : LeNet5(batch) 
{
    this->f_conv1_weight = new float[conv1_in_channel * conv1_out_channel * conv1_kernel_size * conv1_kernel_size];
    this->f_conv1_bias = new float[conv1_out_channel];
    this->f_conv2_weight = new float[conv2_in_channel * conv2_out_channel * conv2_kernel_size * conv2_kernel_size];
    this->f_conv2_bias = new float[conv2_out_channel];
    this->f_fc1_weight = new float[fc1_in_channel * fc1_out_channel];
    this->f_fc1_bias = new float[fc1_out_channel];
    this->f_fc2_weight = new float[fc2_in_channel * fc2_out_channel];
    this->f_fc2_bias = new float[fc2_out_channel];
    this->f_fc3_weight = new float[fc3_in_channel * fc3_out_channel];
    this->f_fc3_bias = new float[fc3_out_channel];
    // Activation
    this->f_output = new float[batch * output_size];
}

LeNet5_cuda::~LeNet5_cuda() {
    cudaFree(d_conv1_weight);   
    cudaFree(d_conv2_weight);   
    cudaFree(d_conv1_bias);     
    cudaFree(d_conv2_bias);     
    cudaFree(d_fc1_weight);     
    cudaFree(d_fc2_weight);     
    cudaFree(d_fc3_weight);     
    cudaFree(d_fc1_bias);       
    cudaFree(d_fc2_bias);       
    cudaFree(d_fc3_bias);       

    cudaFree(d_image);          
    cudaFree(d_input);          
    cudaFree(d_C1_feature_map); 
    cudaFree(d_S2_feature_map); 
    cudaFree(d_C3_feature_map); 
    cudaFree(d_S4_feature_map); 
    cudaFree(d_C5_layer);      
    cudaFree(d_F6_layer);     
    cudaFree(d_output);       
    cudaFree(d_predict_cuda);   

    // Free model parameters memories
    delete[] this->f_conv1_weight;
    delete[] this->f_conv1_bias;
    delete[] this->f_conv2_weight;
    delete[] this->f_conv2_bias;
    delete[] this->f_fc1_weight;
    delete[] this->f_fc1_bias;
    delete[] this->f_fc2_weight;
    delete[] this->f_fc2_bias;
    delete[] this->f_fc3_weight;
    delete[] this->f_fc3_bias;
    // // Free activation memories
    // delete[] this->f_input;
    // delete[] this->f_C1_feature_map;
    // delete[] this->f_S2_feature_map;
    // delete[] this->f_C3_feature_map;
    // delete[] this->f_S4_feature_map;
    // delete[] this->f_C5_layer;
    // delete[] this->f_F6_layer;
    delete[] this->f_output;

    // free unrolled
    cudaFree(d_input_unrolled);
    cudaFree(d_conv1_weight_unrolled);
    // float* d_conv1_bias_unrolled; // is this necessary?
    cudaFree(d_S2_feature_map_unrolled);
    cudaFree(d_conv2_weight_unrolled);
    // float* d_conv2_bias_unrolled;
}

void LeNet5_cuda::prepare_device_memory(float* image) {
    // Store all double arrays as floats
    // Note: this was done here instead of loading everything as floats in LeNet5.cpp
    // in order to provide accuracy comparisons to the double CPU version.
    // No additional computations are performed.
    std::copy(this->conv1_weight, 
              this->conv1_weight+conv1_in_channel*conv1_out_channel*conv1_kernel_size*conv1_kernel_size,
              this->f_conv1_weight);
    // reorder_filters(this->conv1_weight, this->f_conv1_weight, conv1_in_channel, conv1_out_channel, conv1_kernel_size);
    std::copy(this->conv1_bias,
              this->conv1_bias+conv1_out_channel,
              this->f_conv1_bias);
    std::copy(this->conv2_weight,
              this->conv2_weight+conv2_in_channel*conv2_out_channel*conv2_kernel_size*conv2_kernel_size,
              this->f_conv2_weight);
    // reorder_filters(this->conv2_weight, this->f_conv2_weight, conv2_in_channel, conv2_out_channel, conv2_kernel_size);
    std::copy(this->conv2_bias,
              this->conv2_bias+conv2_out_channel,
              this->f_conv2_bias);
    std::copy(this->fc1_weight,
              this->fc1_weight+fc1_in_channel*fc1_out_channel,
              this->f_fc1_weight);
    std::copy(this->fc1_bias,
              this->fc1_bias+fc1_out_channel,
              this->f_fc1_bias);
    std::copy(this->fc2_weight,
              this->fc2_weight+fc2_in_channel*fc2_out_channel,
              this->f_fc2_weight);
    std::copy(this->fc2_bias,
            this->fc2_bias+fc2_out_channel,
            this->f_fc2_bias);
    std::copy(this->fc3_weight,
              this->fc3_weight+fc3_in_channel*fc3_out_channel,
              this->f_fc3_weight);
    std::copy(this->fc3_bias,
              this->fc3_bias+fc3_out_channel,
              this->f_fc3_bias);

    // For unrolled convolution
    // cudaMalloc((void**)&d_input_unrolled, sizeof(float)*
    //     batch * conv1_in_channel*conv1_kernel_size*conv1_kernel_size * C1_size*C1_size);
    // cudaMalloc((void**)&d_conv1_weight_unrolled, sizeof(float)*
    //     conv1_out_channel*conv1_kernel_size*conv1_kernel_size);
    // d_conv1_bias_unrolled; // for now just add scalar
    // cudaMalloc((void**)d_S2_feature_map_unrolled;
    // d_conv2_weight_unrolled;
    // d_conv2_bias_unrolled;
    // cudaMalloc((void**)&d_S2_feature_map_unrolled, sizeof(float)*
    //     batch * conv2_in_channel*conv2_kernel_size*conv2_kernel_size * S2_size*S2_size);
    // cudaMalloc((void**)&d_conv2_weight_unrolled, sizeof(float)*
    //     conv2_out_channel*conv2_kernel_size*conv2_kernel_size);

    // Alloc Model Parameters
    // cudaMalloc((void**)&d_conv1_weight,
    //            sizeof(float) * conv1_in_channel * conv1_out_channel *
    //                conv1_kernel_size * conv1_kernel_size);
    // cudaMalloc((void**)&d_conv1_bias, sizeof(float) * conv1_out_channel);
    // cudaMalloc((void**)&d_conv2_weight,
    //            sizeof(float) * conv2_in_channel * conv2_out_channel *
    //                conv2_kernel_size * conv2_kernel_size);
    // cudaMalloc((void**)&d_conv2_bias, sizeof(float) * conv2_out_channel);
    cudaMalloc((void**)&d_fc1_weight,
                sizeof(float) * fc1_in_channel * fc1_out_channel);
    cudaMalloc((void**)&d_fc1_bias, sizeof(float) * fc1_out_channel);
    cudaMalloc((void**)&d_fc2_weight,
                sizeof(float) * fc2_in_channel * fc2_out_channel);
    cudaMalloc((void**)&d_fc2_bias, sizeof(float) * fc2_out_channel);
    cudaMalloc((void**)&d_fc3_weight,
                sizeof(float) * fc3_in_channel * fc3_out_channel);
    cudaMalloc((void**)&d_fc3_bias, sizeof(float) * fc3_out_channel);

    // Alloc Activations
    cudaMalloc((void**)&d_image,
                sizeof(float) * batch * input_size * input_size * input_channel);
    cudaMalloc((void**)&d_input,
                sizeof(float) * batch * input_channel * input_size * input_size);
    cudaMalloc((void**)&d_C1_feature_map,
                sizeof(float) * batch * C1_channel * C1_size * C1_size);
    cudaMalloc((void**)&d_S2_feature_map,
                sizeof(float) * batch * S2_channel * S2_size * S2_size);
    cudaMalloc((void**)&d_C3_feature_map,
                sizeof(float) * batch * C3_channel * C3_size * C3_size);
    cudaMalloc((void**)&d_S4_feature_map,
                sizeof(float) * batch * S4_channel * S4_size * S4_size);
    cudaMalloc((void**)&d_C5_layer, sizeof(float) * batch * C5_size);
    cudaMalloc((void**)&d_F6_layer, sizeof(float) * batch * F6_size);
    cudaMalloc((void**)&d_output, sizeof(float) * batch * output_size);

    // Copy Parameters
    cudaMemcpyToSymbol(d_conv1_weight, f_conv1_weight,
                       sizeof(float) * conv1_in_channel * conv1_out_channel *
                       conv1_kernel_size * conv1_kernel_size);
    cudaMemcpyToSymbol(d_conv1_bias, f_conv1_bias, sizeof(float) * conv1_out_channel);
    cudaMemcpyToSymbol(d_conv2_weight, f_conv2_weight,
                       sizeof(float) * conv2_in_channel * conv2_out_channel *
                       conv2_kernel_size * conv2_kernel_size);
    cudaMemcpyToSymbol(d_conv2_bias, f_conv2_bias, sizeof(float) * conv2_out_channel);

    cudaMemcpy(d_fc1_weight, f_fc1_weight,
                sizeof(float) * fc1_in_channel * fc1_out_channel,
                cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_bias, f_fc1_bias, sizeof(float) * fc1_out_channel,
                cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_weight, f_fc2_weight,
                sizeof(float) * fc2_in_channel * fc2_out_channel,
                cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_bias, f_fc2_bias, sizeof(float) * fc2_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_weight, f_fc3_weight,
               sizeof(float) * fc3_in_channel * fc3_out_channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_bias, f_fc3_bias, sizeof(float) * fc3_out_channel,
               cudaMemcpyHostToDevice);

    // copy input image
    size_t image_size = batch * input_size * input_size * input_channel;
    cudaMemcpy(d_image, image, image_size * sizeof(float),
                cudaMemcpyHostToDevice);
}

/**
 * 将图像数据转成列的形式
 * Each thread writes a (K*K) partial column.
 * Writing by columns is good b/c coalesces over rows.
 *
 * output: (C * K * K) x (H * W)
 */
__global__ void im2col(float *X, float *X_unrolled, int IC, int H, int W, int K)
{
    int H_OUT = H - (K - 1); // output dimensions
    int W_OUT = W - (K - 1);

    int b = blockIdx.x;
    int t = blockIdx.y * 1024 + threadIdx.x;
    int W_prime = H_OUT * W_OUT;
    int H_prime = IC * K * K;

    if (t < IC * W_prime)
    {                         // each thread will update one channel of one column
        int ic = t / W_prime; // channel = row block
        int s = t % W_prime;  // column

        int h_out = s / W_OUT; // original output row
        int w_out = s % W_OUT; // original output column

        int w_unroll = h_out * W_OUT + w_out; // index of
        int h_base = ic * K * K;              // starting row of unrolled

        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                int h_unroll = h_base + p * K + q;
                int in_idx = b * (IC * H * W) + ic * (H * W) + (h_out + p) * (W) + (w_out + q);
                int out_idx = b * (H_prime * W_prime) + h_unroll * (W_prime) + w_unroll;
                // printf("b=%d t=%d ic=%d h_out=%d w_out=%d h_unroll=%d w_unroll=%d out_idx=%d\n",
                //     b, t, ic, h_out, w_out, h_unroll, w_unroll, out_idx);
                X_unrolled[out_idx] = X[in_idx];
                // X_unrolled[out_idx] = 1.0f;
            }
        }
    }
}

/**
 * This can be called in prepare() since it is just reordering
 * (IC x OC x K x K) to (OC x IC x K x K).
 */
void reorder_filters(float *F, float *F_col, int IC, int OC, int K)
{
    for (int ic = 0; ic < IC; ic++)
    {
        for (int oc = 0; oc < OC; oc++)
        {
            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < K; j++)
                {
                    F_col[oc * (IC * K * K) + ic * (K * K) + i * (K) + j] = float(F[ic * (OC * K * K) + oc * (K * K) + i * (K) + j]);
                }
            }
        }
    }
}

void unrolled_conv(float *X, float *X_unrolled, float *Y, float *weight, float *bias,
                   const int B, const int H, const int W, const int IC, const int OC, int K)
{
    int H_OUT = H - (K - 1); // output dimensions
    int W_OUT = W - (K - 1);

    int total_threads = IC * H_OUT * W_OUT;
    dim3 unrollGridDim(B, ceil(total_threads / 1024), 1);
    dim3 unrollBlockDim(1024, 1, 1);
    im2col<<<unrollGridDim, unrollBlockDim>>>(X, X_unrolled, IC, H, W, K);

    // dim3 matmulGridDim();
    // dim3 matmulBlockDim();
    // naive_matmul_1(f_conv1_weight, X_unrolled, Y);
}

void unrolled_conv_1(float *X, float *X_unrolled, float *Y,
                     const int B, const int H, const int W, const int IC, const int OC, int K)
{

    int H_OUT = H - (K - 1); // output dimensions
    int W_OUT = W - (K - 1);

    int total_threads = IC * H_OUT * W_OUT;
    dim3 unrollGridDim(B, ceil(total_threads / 1024), 1);
    dim3 unrollBlockDim(1024, 1, 1);
    im2col<<<unrollGridDim, unrollBlockDim>>>(X, X_unrolled, IC, H, W, K);

    dim3 matmulGridDim();
    dim3 matmulBlockDim();
    // naive_matmul_1(f_conv1_weight, X_unrolled, Y);
}

void unrolled_conv_2(float *X, float *X_unrolled, float *Y,
                     const int B, const int H, const int W, const int IC, const int OC, int K)
{

    int H_OUT = H - (K - 1); // output dimensions
    int W_OUT = W - (K - 1);

    int total_threads = IC * H_OUT * W_OUT;
    dim3 unrollGridDim(B, ceil(total_threads / 1024), 1);
    dim3 unrollBlockDim(1024, 1, 1);
    im2col<<<unrollGridDim, unrollBlockDim>>>(X, X_unrolled, IC, H, W, K);

    // naive_matmul(X_unrolled, Y);
}

__global__ void conv1(float *input, float *output,
                      int B, const int H, const int W, const int IC, int OC,
                      int K)
{
    int H_OUT = H - (K - 1); // output dimensions
    int W_OUT = W - (K - 1);

    int b = blockIdx.x;  // batch
    int oc = blockIdx.y; // output channel
    int w = threadIdx.x; // col
    int h = threadIdx.y; // row

    __shared__ float X_shared[1 * 28 * 28]; // static allocation
    // We can load the entire input into shared memory?
    for (int channel = 0; channel < 1; channel++)
    {
        X_shared[channel * (H * W) + h * (W) + w] = input[b * (IC * H * W) + channel * (H * W) + h * (W) + w];
    }
    __syncthreads();

    if (w < W_OUT && h < H_OUT)
    { // more threads than output
        // Convolution
        int output_index =
            b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
        // output[output_index] = d_conv1_bias[oc];
        float val = d_conv1_bias[oc];

        for (int ic = 0; ic < IC; ic++)
        { // input channels
            // int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
            int kernel_base = oc * (IC * K * K) + ic * (K * K);

            for (int kh = 0; kh < K; kh++)
            { // kernel height
                for (int kw = 0; kw < K; kw++)
                { // kernel width
                    // float val = input[input_base + kh * (W) + kw] *
                    //               d_conv1_weight[kernel_base + kh * (K) + kw];
                    val += X_shared[ic * (H * W) + h * (W) + w + kh * (W) + kw] *
                           d_conv1_weight[kernel_base + kh * (K) + kw];
                }
            }
        }
        output[output_index] = val;
    }
}

__global__ void conv2(float *input, float *output,
                      int B, const int H, const int W, const int IC, int OC,
                      int K)
{
    int H_OUT = H - (K - 1); // output dimensions
    int W_OUT = W - (K - 1);

    int b = blockIdx.x;  // batch
    int oc = blockIdx.y; // output channel
    int w = threadIdx.x; // col
    int h = threadIdx.y; // row

    __shared__ float X_shared[6 * 14 * 14]; // static allocation
    // We can load the entire input into shared memory?
    for (int channel = 0; channel < 6; channel++)
    {
        X_shared[channel * (H * W) + h * (W) + w] = input[b * (IC * H * W) + channel * (H * W) + h * (W) + w];
    }
    __syncthreads();

    if (w < W_OUT && h < H_OUT)
    { // more threads than output
        // Convolution
        int output_index =
            b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
        // output[output_index] = d_conv2_bias[oc];
        float val = d_conv2_bias[oc];
        for (int ic = 0; ic < IC; ic++)
        { // input channels
            // int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
            int kernel_base = oc * (IC * K * K) + ic * (K * K);

            for (int kh = 0; kh < K; kh++)
            { // kernel height
                for (int kw = 0; kw < K; kw++)
                { // kernel width
                    // float val = input[input_base + kh * (W) + kw] *
                    //               d_conv1_weight[kernel_base + kh * (K) + kw];
                    val += X_shared[ic * (H * W) + h * (W) + w + kh * (W) + kw] *
                           d_conv2_weight[kernel_base + kh * (K) + kw];
                }
            }
        }
        output[output_index] = val;
    }
}

/**
 * (batch_size, 1, 1) x (28, 28, 1)
 */
__global__ void normalize(int batch, int input_channel, int input_size, const float *const d_image, float *d_input)
{
    // automatically placed in registers. Should these be in shared / constant?
    // probably not because they're just single variables
    const float max_int = 255.0f;
    const float mean = 0.5f;
    const float var = 0.5f;

    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    const int col = threadIdx.x;
    const int row = threadIdx.y;

    float val;

    if (col < input_size && row < input_size)
    {
        // standard normalize, center at 0
        // one global memory read, one write
        val = d_image[batch_id * input_channel * input_size * input_size + channel_id * input_size * input_size + row * input_size + col];
        val = ((val / max_int) - mean) / var;
        d_input[batch_id * input_channel * input_size * input_size + channel_id * input_size * input_size + row * input_size + col] = val;
    }
}

/**
 * (batch_size, out_channels, 1) x (width, height, 1)
 */
__global__ void naive_conv(float *input, float *output, float *weight,
                           float *bias, int B, int H, int W, int IC, int OC,
                           int K)
{
    int H_OUT = H - (K - 1); // output dimensions
    int W_OUT = W - (K - 1);

    int b = blockIdx.x;  // batch
    int oc = blockIdx.y; // output channel
    int w = threadIdx.x; // col
    int h = threadIdx.y; // row

    // Convolution
    int output_index = b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
    output[output_index] = bias[oc];
    for (int ic = 0; ic < IC; ic++)
    {
        int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
        int kernel_base = oc * (IC * K * K) + ic * (K * K);
        for (int kh = 0; kh < K; kh++)
        {
            for (int kw = 0; kw < K; kw++)
            {
                float val = input[input_base + kh * (W) + kw] *
                            weight[kernel_base + kh * (K) + kw];
                output[output_index] += val;
            }
        }
    }
}

/**
 * (batch_size, in_channels, 1) x (width, height, 1)
 * Element-wise.
 */
__global__ void naive_relu(float *feature_map, int channels, int width, int height)
{
    int b = blockIdx.x;  // batch
    int oc = blockIdx.y; // output channel
    int w = threadIdx.x; // col
    int h = threadIdx.y; // row

    int index = b * (channels * width * height) + oc * (width * height) + h * width + w;

    feature_map[index] = fmax(feature_map[index], 0.0f);
}

/**
 * (batch_size, in_channels, 1) x (width, height, 1)
 */
__global__ void naive_pool(float *input, float *output, int C, int H, int W)
{
    int scale = 2;
    int H_OUT = H / scale;
    int W_OUT = W / scale;

    int b = blockIdx.x;  // batch
    int c = blockIdx.y;  // output channel
    int w = threadIdx.x; // col
    int h = threadIdx.y; // row

    int input_base = b * (C * H * W) + c * (H * W) + (h * 2) * (W) + (w * 2);
    int max_sh = 0;
    int max_sw = 0;
    float max_val = 0.0f; // since after relu

    // Find maximum
    for (int sh = 0; sh < scale; sh++)
    {
        for (int sw = 0; sw < scale; sw++)
        {
            float val = input[input_base + sh * (W) + sw];
            if (val > max_val)
            {
                max_val = val;
            }
        }
    }

    // Set output with max value
    int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
                       h * (W_OUT) + w;
    output[output_index] = max_val;
}

/**
 * (batch, 1, 1) x (output_nodes, 1, 1)
 */
__global__ void naive_fc(float *input, float *output, float *weight, float *bias,
                         int IC, int OC)
{
    int b = blockIdx.x;
    int oc = threadIdx.x;

    // output[b * OC + oc] = bias[oc];
    float val = bias[oc];
    for (int ic = 0; ic < IC; ic++)
        val += weight[oc * IC + ic] * input[b * IC + ic];

    output[b * OC + oc] = val;
}

/**
 * (batch, num_rowblocks, 1) x (output_nodes, 1, 1)
 */
__global__ void fc_rowblock(float *input, float *output, float *weight, float *bias,
                            int IC, int OC, int num_rowblocks)
{
    int batch = blockIdx.x;
    int block = blockIdx.y;
    int t = threadIdx.x;

    int block_length = OC / num_rowblocks; // assume divisible
    int row_start = block * IC * block_length;

    extern __shared__ float shmem[]; // stores one complete row block of W

    if (t < IC)
    {
        for (int r = 0; r < block_length; r++)
        {
            shmem[r * IC + t] = weight[row_start + r * IC + t];
        }
    }
    __syncthreads();

    if (t < block_length)
    { // calculate one output element of y
        float val = bias[block * block_length + t];

#pragma unroll
        for (int ic = 0; ic < IC; ic++)
        {
            val += shmem[t * IC + ic] * input[batch * IC + ic];
        }
        output[batch * OC + block * block_length + t] = val;
    }
    // __syncthreads();
}
// s1 all In one kernel
// s2 one block inference one image
// s3 直接返回准确率
void LeNet5_cuda::predict(int batch)
{
    // cpu_normalize(image, input);
    dim3 normGridDim(batch, 1, 1);
    dim3 normBlockDim(28, 28, 1);
    normalize<<<normGridDim, normBlockDim>>>(batch, input_channel, input_size, d_image, d_input);
    // // debug
    // cudaMemcpy(input, d_input, sizeof(float)*batch*input_channel*input_size*input_size, cudaMemcpyDeviceToHost);
    // std::cout<<"input:"<<std::endl;
    // for(int i=0;i<batch*input_channel*input_size*input_size;i++)
    // {
    //     std::cout<<input[i]<<" ";
    // }
    // Conv2d
    // cpu_conv(input, C1_feature_map, conv1_weight, conv1_bias, batch, input_size,
    //      input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);
    // unrolled_conv(d_input, d_input_unrolled, d_C1_feature_map, d_conv1_weight, d_conv1_bias,
    //     batch, input_size, input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);

    dim3 conv1GridDim(batch, 6, 1);
    dim3 conv1BlockDim(28, 28, 1);
    conv1<<<conv1GridDim, conv1BlockDim>>>(d_input, d_C1_feature_map, batch, input_size,
                                           input_size, conv1_in_channel, conv1_out_channel, conv1_kernel_size);

    // // debug
    // cudaMemcpy(C1_feature_map, d_C1_feature_map, sizeof(float)*batch*conv1_out_channel*C1_size*C1_size, cudaMemcpyDeviceToHost);
    // std::cout<<"C1_feature_map:"<<std::endl;
    // for(int i=0;i<batch*conv1_out_channel*C1_size*C1_size;i++)
    // {
    //     std::cout<<C1_feature_map[i]<<" ";
    // }
    // cpu_relu(C1_feature_map, batch * C1_channel * C1_size * C1_size);
    dim3 relu1GridDim(batch, 6, 1);
    dim3 relu1BlockDim(28, 28, 1);
    naive_relu<<<relu1GridDim, relu1BlockDim>>>(d_C1_feature_map, C1_channel, C1_size, C1_size);
    // cudaMemcpy(C1_feature_map, d_C1_feature_map, sizeof(float)*batch*conv1_out_channel*C1_size*C1_size, cudaMemcpyDeviceToHost);

    // MaxPool2d
    // cpu_pool(C1_feature_map, S2_feature_map, batch, C1_channel, C1_size, C1_size);
    dim3 pool1GridDim(batch, 6, 1);
    dim3 pool1BlockDim(14, 14, 1);
    naive_pool<<<pool1GridDim, pool1BlockDim>>>(d_C1_feature_map, d_S2_feature_map, C1_channel, C1_size, C1_size);
    // cudaMemcpy(S2_feature_map, d_S2_feature_map, sizeof(float)*batch*conv1_out_channel*S2_size*S2_size, cudaMemcpyDeviceToHost);

    // Conv2d
    // cpu_conv(S2_feature_map, C3_feature_map, conv2_weight, conv2_bias, batch, S2_size,
    //      S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);

    // unrolled_conv(d_S2_feature_map, d_S2_feature_map_unrolled, d_C3_feature_map, d_conv2_weight, d_conv2_bias,
    //     batch, S2_size, S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);

    dim3 conv2GridDim(batch, 16, 1);
    dim3 conv2BlockDim(16, 16, 1); // too few threads?
    conv2<<<conv2GridDim, conv2BlockDim>>>(d_S2_feature_map, d_C3_feature_map,
                                           batch, S2_size, S2_size, conv2_in_channel, conv2_out_channel, conv2_kernel_size);

    // cpu_relu(C3_feature_map, batch * C3_channel * C3_size * C3_size);
    dim3 relu2GridDim(batch, 16, 1);
    dim3 relu2BlockDim(10, 10, 1);
    naive_relu<<<relu2GridDim, relu2BlockDim>>>(d_C3_feature_map, C3_channel, C3_size, C3_size);

    // MaxPool2d
    // cpu_pool(C3_feature_map, S4_feature_map, batch, C3_channel, C3_size, C3_size);
    dim3 pool2GridDim(batch, 16, 1);
    dim3 pool2BlockDim(5, 5, 1); // doesn't fill 32 threads per block -> underutilized
    naive_pool<<<pool2GridDim, pool2BlockDim>>>(d_C3_feature_map, d_S4_feature_map, C3_channel, C3_size, C3_size);
    // cudaMemcpy(S4_feature_map, d_S4_feature_map, sizeof(float)*batch*conv2_out_channel*S4_size*S4_size, cudaMemcpyDeviceToHost);

    // Linear
    // cpu_fc(S4_feature_map, C5_layer, fc1_weight, fc1_bias, batch, fc1_in_channel,
    //    fc1_out_channel);
    int fc1RowblockNum = 6;
    dim3 fc1GridDim(batch, fc1RowblockNum, 1);
    dim3 fc1BlockDim(432, 1, 1);
    fc_rowblock<<<fc1GridDim, fc1BlockDim, sizeof(float) * fc1_in_channel *(fc1_out_channel / fc1RowblockNum)>>>(d_S4_feature_map, d_C5_layer, d_fc1_weight, d_fc1_bias,
                                                                                                                 fc1_in_channel, fc1_out_channel, fc1RowblockNum);

    // cpu_relu(C5_layer, batch * C5_size);
    dim3 relu3GridDim(batch, 1, 1);
    dim3 relu3BlockDim(120, 1, 1);
    naive_relu<<<relu3GridDim, relu3BlockDim>>>(d_C5_layer, 1, 120, 1);

    // Linear
    // cpu_fc(C5_layer, F6_layer, fc2_weight, fc2_bias, batch, fc2_in_channel,
    //    fc2_out_channel);
    int fc2RowblockNum;
    if (batch == 1)
        fc2RowblockNum = 12;
    else
        fc2RowblockNum = 1;
    dim3 fc2GridDim(batch, fc2RowblockNum, 1);
    dim3 fc2BlockDim(128, 1, 1);
    fc_rowblock<<<fc2GridDim, fc2BlockDim, sizeof(float) * fc2_in_channel *(fc2_out_channel / fc2RowblockNum)>>>(d_C5_layer, d_F6_layer, d_fc2_weight, d_fc2_bias,
                                                                                                                 fc2_in_channel, fc2_out_channel, fc2RowblockNum);

    // cpu_relu(F6_layer, batch * F6_size);
    dim3 relu4GridDim(batch, 1, 1);
    dim3 relu4BlockDim(84, 1, 1);
    naive_relu<<<relu4GridDim, relu4BlockDim>>>(d_F6_layer, 1, 84, 1);

    // Linear
    // cpu_fc(F6_layer, output, fc3_weight, fc3_bias, batch, fc3_in_channel,
    //    fc3_out_channel);
    int fc3RowblockNum;
    if (batch == 1)
        fc3RowblockNum = 10;
    else
        fc3RowblockNum = 1;
    dim3 fc3GridDim(batch, fc3RowblockNum, 1);
    dim3 fc3BlockDim(128, 1, 1);
    fc_rowblock<<<fc3GridDim, fc3BlockDim, sizeof(float) * fc3_in_channel *(fc3_out_channel / fc3RowblockNum)>>>(d_F6_layer, d_output, d_fc3_weight, d_fc3_bias,
                                                                                                                 fc3_in_channel, fc3_out_channel, fc3RowblockNum);

    // dest, source, number of bytes, transfer type
    // cudaMemcpy(d_output, output, sizeof(float)*batch*output_size, cudaMemcpyHostToDevice);

    /* NOTE: unless you want to make a major change to this class structure,
     *  you need to write your output to the device memory d_output
     *  so that classify() can handle the rest.
     */
}

void LeNet5_cuda::classify(int* predict, int batch) 
{
    // GPU -> CPU
    // dest, source, number of bytes, transfer type
    cudaMemcpy(f_output, d_output, sizeof(float) * output_size * batch, cudaMemcpyDeviceToHost);

    // // float back to double
    // std::copy(f_output,
    //           f_output+batch*output_size,
    //           output);
    // // debug
    // for(int j = 0; j < output_size; j++)
    // {
    //     std::cout<<f_output[j]<<" ";
    // }
    // std::cout<<std::endl;
    for (int i = 0; i < batch; i++)
    {
        float predication = f_output[i*batch];
        int idx = 0;
        for(int j = 0; j < output_size; j++)
        {
            if(predication < f_output[i*batch + j])
            {
                predication = f_output[i*batch + j];
                idx = j;
            }
        }
        predict[i] = idx;
        // std::cout<<"idx: "<<idx<<std::endl;

    }
}

int main(int argc, char *argv[])
{
    std::string dir = argv[1]; // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
    // cout << dir;
    int batch = 1;
    FashionMNIST fm(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte", dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte", batch);
    fm.load_data();

    int row = fm.image_row;
    int** predict_cuda = new int*[fm.label_row];
    for (int i = 0; i < fm.label_row; i++) 
        predict_cuda[i] = new int[fm.label_col]; 

    LeNet5_cuda* net_cuda = new LeNet5_cuda(batch);
    net_cuda->load_parameters(dir);
    
    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < row; r++)
    {
        // cudaEvent_t start, stop;
        // float cudaElapsedTime;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start, 0);
        net_cuda->prepare_device_memory(fm.image_set[r]); 
        net_cuda->predict(fm.image_set[r], batch);
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&cudaElapsedTime, start, stop);
        // std::cout << "[INFO] CUDA elapsed time is " << cudaElapsedTime << " msec" << std::endl;
        net_cuda->classify(predict_cuda[r], batch); 
    }
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    int sum = 0;
    int correct = 0;
    for (int i = 0; i < fm.label_row; i++) 
        for(int j = 0; j < fm.label_col; j++)
        {
            if(predict_cuda[i][j] == fm.label_set[i][j])
            {
                correct++;
            }
            sum++;
        }

    std::ofstream outfilep("prediction_cuda.txt");
    for (int i = 0; i < fm.label_row; i++) 
    {
        for (int j = 0; j < fm.label_col; j++) 
        {
            outfilep << predict_cuda[i][j] << " ";
        }
        outfilep << "\n";
    }
    outfilep.close();

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
    
    std::chrono::duration<float> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
        std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << (double)correct/sum << std::endl;


    return 0;
}