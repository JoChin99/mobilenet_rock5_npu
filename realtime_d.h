#pragma once

#include "mobilenetv2_features.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <functional>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

/**
 * Realtime classifier with different backend, NPU and CPU
 * Camera loop calls doAsyncStep() at the framerate
 * but a new inference thread is only started if the
 * previous one has finished
 * 
 * Backend is selected at construction time: NPU or CPU
 */
class RTClassifier
{
public:
    using ClassificationCallback = std::function<void(int class_label, float class_prob, long infer_time)>;
    RTClassifier(const std::string &model_path, const std::string &classifier_path, int nClasses, bool npu_backend) : npu_backend_(npu_backend)
    
    {
        if(npu_backend_)
        {
            features = std::make_unique<MobileNetV2Features>(model_path);
        }
        else
        {
            // CPU via ONNX Runtime
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            sess = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
        }

        classifier = torch::nn::Sequential(
            torch::nn::Dropout(0.2),
            torch::nn::Linear(MobileNetV2Features::N_OUTPUT_FEATURES, nClasses));
        load_classifier_weights(classifier_path);
        classifier->eval();
    }
    
    ~RTClassifier()
    {
        if(thr.joinable())
            thr.join();
    }

    // Submit a frame for async inference and return false and drops the frame
    // if the previous inference is still running
    bool doAsyncStep(cv::Mat img)
    {
        if (isRunning)
        {
            return false;
        }
        if(thr.joinable())
        {
            thr.join();
        }
        isRunning = true;
        thr = std::thread(&RTClassifier::worker, this, img);
        return true;
    }

    void doSyncStep(cv::Mat img, int &class_label, float &class_prob)
    {
        torch::NoGradGuard no_grad;
        torch::Tensor input = MobileNetV2Features::preprocess(img).unsqueeze(0);
        torch::Tensor ft;

        if(npu_backend_)
        {
            ft = features->forward(input);
        }
        else
        {
            // Convert to float and normalize
            torch::Tensor x = input.to(torch::kFloat).div_(255.0);
            torch::Tensor mean = torch::tensor({0.485f,0.456f,0.406f}).view({1,3,1,1});
            torch::Tensor std = torch::tensor({0.229f,0.224f,0.225f}).view({1,3,1,1});
            x = x.sub_(mean).div_(std). contiguous();

            // Run ONNX
            std::array<int64_t,4> shape{1,3,224,224};
            Ort::Value tensor = Ort::Value::CreateTensor<float>(mem,x.data_ptr<float>(),x.numel(),shape.data(),shape.size());

            const char *in[] = {"input"};
            const char *out[] = {"features"};
            auto result = sess->Run(Ort::RunOptions{nullptr},in,&tensor,1,out,1);

            float *data = result[0].GetTensorMutableData<float>();
            ft = torch::from_blob(data,{1,1280},torch::kFloat);
        }

        torch::Tensor logits = classifier->forward(ft);
        class_label = logits.argmax(1).item<int>();
        class_prob = torch::softmax(logits, 1)[0][class_label].item<float>();
    }

    //Wait for the current inference thread to complete
    void waitForCompletion()
    {
        if(thr.joinable())
            thr.join();
    }
    ClassificationCallback classificationCallback;

private:
    void worker(cv::Mat img)
    {
        //isRunning = true;
        int class_label = -1;
        float class_prob = 0;
        const auto start = std::chrono::steady_clock::now();
        doSyncStep(img, class_label, class_prob);
        const auto end = std::chrono::steady_clock::now();
        const long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        if (classificationCallback)
            classificationCallback(class_label,class_prob,duration);
        isRunning = false;
    }

    // The classifier is saved saved by transfer.cpp as the pickled key/tensor dict
    // to avoid LibTorch/PyTorch naming convention mismatches
    void load_classifier_weights(const std::string &pt_path)
    {
        std::ifstream file(pt_path, std::ios::binary);
        if (!file)
            throw std::runtime_error("-E- Failed to open classifier model file: " + pt_path);
        std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        auto in_value = torch::pickle_load(bytes);
        auto dict = in_value.toGenericDict();
        auto params = classifier->named_parameters();
        torch::NoGradGuard no_grad;

        for(const auto &kv : dict)
        {
            const std::string key = kv.key().toStringRef();
            auto *p = params.find(key); // 1.weight, 1.bias
            if (!p)
                throw std::runtime_error("-E- Unexpected key in classifier file: " + key);
            p->copy_(kv.value().toTensor());
        }
    }
    bool npu_backend_;
    std::unique_ptr<MobileNetV2Features> features;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "demo"};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::unique_ptr<Ort::Session> sess;

    //MobileNetV2Features features;
    torch::nn::Sequential classifier{nullptr};  // Trained classifier (via LibTorch)
    std::thread thr;
    std::atomic<bool> isRunning = false;
};