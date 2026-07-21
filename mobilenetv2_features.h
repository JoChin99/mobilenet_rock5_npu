#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#ifdef NDEBUG
constexpr bool debugOutput = false;
#else
constexpr bool debugOutput = true;
#endif


/**
 * MobileNetV2 features with pre-trained weights running on Rock5B+ NPU

 * This class wraps the RKNN C API
 */
class MobileNetV2Features : public torch::nn::Module
{
public:
    static constexpr int INPUT_SIZE      = 224;
    static constexpr int N_OUTPUT_FEATURES = 1280;
    

    /** Load the RKNN model */
    MobileNetV2Features(const std::string &rknn_model_file = "mobilenet_features.rknn")
    {
        // Read the .rknn file
        std::ifstream f(rknn_model_file, std::ios::binary | std::ios::ate);
        if (!f)
            throw std::runtime_error("-E- Failed to open RKNN model file: " + rknn_model_file);
        std::vector<char> model_data(f.tellg());
        f.seekg(0);
        f.read(model_data.data(), model_data.size());

        // Initialize NPU context
        int ret = rknn_init(&ctx_, model_data.data(), model_data.size(), 0, nullptr);
        if (ret < 0) 
            throw std::runtime_error("-E- Failed to intialize RKNN model: ");
        std::cout << "RKNN model loaded: " << rknn_model_file << "\n";

        /*
        Just for debugging purpose to print how many input and output tensors from loaded rknn model file has.

        rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
        if (debugOutput)
        {
            std::cerr << "[MobileNetV2Features] NPU model loaded: "
                << rknn_model_file << " (in=" << io_num_.n_input << ", out=" << io_num_.n_output << ")\n";
        }
        */
    }
    ~MobileNetV2Features()
    {
        if (ctx_) 
            // Release to unload the rknn model before initialize to allocate resources on the NPU/system memory
            rknn_destroy(ctx_);
    }

    // Holds an NPU context
    MobileNetV2Features(const MobileNetV2Features&) = delete;
    MobileNetV2Features& operator=(const MobileNetV2Features&) = delete;
    
    /**
    * @brief Performs the forward pass.
    *
    * @param x The batch of input images.
    */
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.contiguous().cpu();
        const int N = x.size(0);
        auto out = torch::zeros({N, N_OUTPUT_FEATURES}, torch::kFloat);

        for (int i = 0; i < N; i++)
        {
            /*
            int channel = 3;
            int width = 0;
            int height = 0;
            if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
            {
                printf("model is NCHW input fmt\n");
                channel = input_attrs[0].dims[1];
                height = input_attrs[0].dims[2];
                width = input_attrs[0].dims[3];
            }
            else
            {
                printf("model is NHWC input fmt\n");
                height = input_attrs[0].dims[1];
                width = input_attrs[0].dims[2];
                channel = input_attrs[0].dims[3];
            }
            */

            // Slice [3,224,224] to [224,224,3] (NHWC) for RKNN
            auto hwc = x[i].permute({1, 2, 0}).contiguous();
            rknn_input inputs[1];
            rknn_output outputs[1];
            std::memset(&inputs, 0, sizeof(inputs));
            inputs[0].index = 0;
            inputs[0].type = RKNN_TENSOR_UINT8;
            inputs[0].size = INPUT_SIZE * INPUT_SIZE * 3;
            inputs[0].fmt = RKNN_TENSOR_NHWC;
            inputs[0].buf = hwc.data_ptr<uint8_t>();

            if (rknn_inputs_set(ctx_, 1, inputs) < 0)
                throw std::runtime_error("-E- Failed to set RKNN inputs.");
            if (rknn_run(ctx_, nullptr) < 0)
                throw std::runtime_error("-E- Failed to run RKNN model.");

            std::memset(outputs, 0, sizeof(outputs));
            outputs[0].want_float = 1;

            if (rknn_outputs_get(ctx_, 1, outputs, nullptr) < 0)
                throw std::runtime_error("-E- Failed to get RKNN outputs.");
            //std::cout << "buf=" << outputs[0].buf << '\n';
            //std::cout << "size=" << outputs[0].size << '\n';
            //std::cout << "expected=" << N_OUTPUT_FEATURES * 7 * 7 * sizeof(float) << '\n';
            std::memcpy(out[i].data_ptr<float>(), outputs[0].buf, N_OUTPUT_FEATURES * sizeof(float));
            rknn_outputs_release(ctx_, 1, outputs);
        }
        return out;
    }


    /**
     * @brief Preprocessing of an OpenCV image for inference
     */
    static torch::Tensor preprocess(cv::Mat img, bool resizeOnly = false)
    {
        constexpr int numChannels         = 3;  // colour
        if (img.depth() != CV_8U)
            throw std::invalid_argument("Image is not 8bit.");
        if (img.channels() != numChannels)
            throw std::invalid_argument("Image is not BGR / colour.");

        cv::resize(img, img, cv::Size(INPUT_SIZE, INPUT_SIZE));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Return uint8 CHW tensor
        torch::Tensor tensor = torch::from_blob(img.data, {INPUT_SIZE, INPUT_SIZE, 3}, torch::kByte);
        tensor = tensor.permute({2, 0, 1}).contiguous();   // HWC -> CHW
        return tensor;
    }

private:
    rknn_context ctx_ = 0;  //
    //rknn_input_output_num io_num_;
    // Store the binary contents of the RKNN model file
    std::vector<unsigned char> model_data_;
};
