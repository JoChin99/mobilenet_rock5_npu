#include "mobilenetv2_features.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

// Path of the RKNN model
const fs::path rknn_model_path  = "~/mobilenet_rock5/models/mobilenetv2_features.rknn";

// Path to the classifier file
const char classifier_model_path[] = "~/mobilenet_rock5/models/classifier.pt";

// Subdirs of the classes
const std::vector<std::string> classes = {"circle", "heart", "star"};

// Initialize camera
const int camera_dev = 0;


int main()
{
    // Load models
    MobileNetV2Features features(rknn_model_path.string());

    // Load the saved [weight, bias] tensors
    std::vector<torch::Tensor> params;
    // params[0]=weight, params[1]=bias
    torch::load(params, classifier_model_path);   

    // Reconstruct the same Sequential
    torch::nn::Sequential classifier(
        torch::nn::Dropout(0.2),
        torch::nn::Linear(MobileNetV2Features::N_OUTPUT_FEATURES, (int)classes.size())
    );
    auto param = classifier->named_parameters();
    param["1.weight"].copy_(params[0]);
    param["1.bias"].copy_(params[1]);
    classifier->eval();
    //auto classifier = load_classifier(classifier_model_path);

    // Open camera
    cv::VideoCapture cap(camera_dev);
    if (!cap.isOpened()) {
        std::cerr << "-E- Failed to open camera " << camera_dev << "\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    std::cout << "Press 'q' to quit\n";
    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;
        // Preprocess and run NPU
        auto input= MobileNetV2Features::preprocess(frame).unsqueeze(0);    // [1,3,224,224]
        // Classify
        torch::NoGradGuard no_grad;
        auto start = std::chrono::high_resolution_clock::now();
        auto feat = features.forward(input);    // [1, 1280]
        auto logits = classifier->forward(feat);
        auto end =std::chrono::high_resolution_clock::now();
        auto pred_idx = logits.argmax(1).item<int>();
        auto score = torch::softmax(logits, 1)[0][pred_idx].item<float>();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Inference: " << duration.count() << " ms\n";

        // Overlay result on frame
        std::string label = classes[pred_idx] + " " + std::to_string((int)(score * 100)) + "%";
        cv::putText(frame, label, {10, 40},cv::FONT_HERSHEY_SIMPLEX, 1.2, {0, 255, 0}, 2);
        cv::imshow("Shape Classifier", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}