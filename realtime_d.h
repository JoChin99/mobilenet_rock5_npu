#pragma once

#include "mobilenetv2_features.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <chrono>
#include <functional>
#include <fstream>
#include <string>
#include <vector>

/**
 * Realtime classifier
 * Camera loop calls doAsyncStep() at the framerate
 * but a new inference thread is only started if the
 * previous one has finished
 * 
 */
class RTClassifier
{
public:
    using ClassificationCallback = std::function<void(int class_label, float class_prob, long infer_time)>;
    RTClassifier(const std::string &rknn_model_path, const std::string &classifier_path, int nClasses) : features(rknn_model_path)
    
    {
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

    // Perform classificaiton in a background thread
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
        thr = std::thread(&RTClassifier::worker, this, img);
        return true;
    }

    void doSyncStep(cv::Mat img, int &class_label, float &class_prob)
    {
        torch::NoGradGuard no_grad;
        torch::Tensor input = MobileNetV2Features::preprocess(img).unsqueeze(0);
        torch::Tensor ft = features.forward(input);
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
        isRunning = true;
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

    // Loads the pickled key/tensor dict by transfer.cpp
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
    MobileNetV2Features features;
    torch::nn::Sequential classifier{nullptr};  // Trained classifier (via LibTorch)
    std::thread thr;
    std::atomic<bool> isRunning = false;
};