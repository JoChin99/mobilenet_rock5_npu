#include "realtime_d.h"
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <pwd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <csignal>
#include <atomic>
#include <chrono>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// Path of the RKNN model
const fs::path rknn_model_path  = "mobilenet_rock5/models/mobilenetv2_features.rknn";

// Path to the classifier file
const char classifier_model_path[] = "mobilenet_rock5/models/classifier.pt";

// Subdirs of the classes
const std::vector<std::string> classes = {"circle", "heart", "star"};

// Initialize camera
//const int cam_dev = 0;
const char cam_dev[] = "/dev/video11";

// Check if RKAIQ Camera Engine is not using, true = not running
const bool rkaiq_cam_eng = false;


std::atomic<bool> running(true);

void sigHandler(int) 
{
    running = false;
}

// Optional if using RKAIQ camera engine. This is for AWB when the RKAIQ is not able to use
void V4L2Camera(cv::Mat &img)
{
    if(img.channels() == 3)
    {
        std::vector<cv::Mat> channels;
        cv::split(img,channels);

        const double meanB = cv::mean(channels[0])[0];
        const double meanG = cv::mean(channels[1])[0];
        const double meanR = cv::mean(channels[2])[0];
        const double gray = (meanB + meanG + meanR) / 3.0;

        channels[0].convertTo(channels[0], -1, gray / std::max(meanB, 1.0));
        channels[1].convertTo(channels[1], -1, gray / std::max(meanG, 1.0));
        channels[2].convertTo(channels[2], -1, gray / std::max(meanR, 1.0));

        cv::merge(channels, img);

        img.convertTo(img, -1, 1.25, 8.0);
    }
}

void setV4L2OpenCVParam(const char *dev, int dev_id, int value)
{
    int f = open(dev, O_RDWR);

    if (f < 0)
    {
        std::cerr << "-E- Failed to open V4l2 device" << std::endl;
        return;
    }

    // To tell driver to make changes on camera settings
    struct v4l2_control ctrl{};
    ctrl.id = dev_id;
    ctrl.value = value;
    if (ioctl(f, VIDIOC_S_CTRL, &ctrl) < 0)
    {
        std::cerr << "-E- VIDIOC_S_CTRL failed" << std::endl;
    }

    close(f);
}

int main()
{
    std::signal(SIGINT, sigHandler);
    const fs::path home_dir(getpwuid(getuid())->pw_dir);
    RTClassifier rtClassifier((home_dir / rknn_model_path).string(), (home_dir / classifier_model_path).string(), (int)classes.size());

    std::atomic<long> nFrames{0};
    std::atomic<long> nInferFrames{0};
    std::atomic<long> nSkipFrames{0};
    std::atomic<long> total_infer{0};

    int last_pred_class = -1;
    int new_class = -1;
    int classPred_count = 0;

    const float min_prob = 0.75f;
    const int frame_thres = 3;

    // Inference result callback
    rtClassifier.classificationCallback = [&](int pred_class, float pred_score, long duration)
    {
        //const long current_infer = nInferFrames.fetch_add(1) + 1;
        //const long current_total = nFrames.load();
        nInferFrames.fetch_add(1);
        total_infer += duration;

        /* Avoid spamming the terminal with output messages */ 
        // Ignore uncertain predictions
        if (pred_score < min_prob)
            pred_class = -1;

        // Count consecutive same predictions
        if (pred_class == new_class)
            classPred_count++;
        else
        {
            new_class = pred_class;
            classPred_count = 1;
        }

        // Wait until prediction is stable
        if (classPred_count < frame_thres)
            return;
        
        // Stay quiet if the output is the same class as before
        /*if(pred_class == last_pred_class)
        {
            return;
        }
        */
        //last_pred_class = pred_class;
        static auto prev_out = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();

        if (pred_class == last_pred_class)
        {
            if (now - prev_out < std::chrono::seconds(5))
                return;
        }
        else
        {
            last_pred_class = pred_class;   // Print on shape change
        }
        prev_out = now;

        //std::cout << "[" << current_infer << "/" << current_total << "]" << "Class Name: " << classes[pred_class] << ", Prediction score: " << (int)(pred_score*100) << "%, Latency: " << duration << "ms  \r" << std::flush;
        if (pred_class == -1)
        {
            std::cout << "-I- Class Name: Not detected" << ", Prediction score: " << (int)(pred_score*100) << "%, Inference Time: " << duration << "ms\n" << std::flush;
        }
        else
        {
            std::cout << "-I- Class Name: " << classes[pred_class] << ", Prediction score: " << (int)(pred_score*100) << "%, Inference Time: " << duration << "ms\n" << std::flush;
        }
    };


    // Increase exposure
    setV4L2OpenCVParam(
        "/dev/v4l-subdev2",
        V4L2_CID_EXPOSURE,
        2500
    );

    // Increase gain
    setV4L2OpenCVParam(
        "/dev/v4l-subdev2",
        V4L2_CID_GAIN,
        6000
    );

    cv::VideoCapture cam(cam_dev, cv::CAP_V4L2);
    if (!cam.isOpened()) 
    {
        std::cerr << "-E- Failed to open camera: " << cam_dev << "\n";
        return 1;
    }
    
    // Camera settings
    cam.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('N','V','1','2')); //Pixel format
    //cam.set(cv::CAP_PROP_GAIN, 6000);
    cam.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    //cam.set(cv::CAP_PROP_FPS, 30);
    cam.set(cv::CAP_PROP_CONVERT_RGB, 1);
    std::cout << "Camera: " << cam.getBackendName() << std::endl;
    std::cout << "-I- Camera frame size: " << cam.get(cv::CAP_PROP_FRAME_WIDTH) << ", " << cam.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    //For debugging purpose
    /*
    const double fps = cam.get(cv::CAP_PROP_FPS);
    const int fourcc = cam.get(cv::CAP_PROP_FOURCC);
    std::cout << "-I- Pipeline: " << (rkaiq_cam_eng ? "Grey-world white balance" : "RKAIQ (ISP)") << "\n";

    cv::Mat test_frame; 
    cam >> test_frame; 
    if (!test_frame.empty()) { 
        cv::imwrite((home_dir / "mobilenet_rock5/cam_test.jpg").string(), test_frame);
        std::cout << "Frame type: " << test_frame.channels() << "channels, Type= " << test_frame.type() << "\n"; // Check if it is BGR (3 channels)
        std::cout << "Test frame is saved to /home/ning/mobilenet_rock5/cam_test.jpg\n"; 
    } 
    */

    std::cout << "-I- Running real-time classification for demo...\n";
    cv::Mat frame;
    cv::Mat last_frame;
    const auto benchStart = std::chrono::steady_clock::now();
    //std::cout << cv::getBuildInformation() << std::endl;

    while (running)
    {
        cam >> frame;
        if (frame.empty())
        {
            std::cerr << "\n-E- Empty frame found.\n";
            break;
        }
        nFrames++;
        
        // Only apply when RKAIQ is not available
        if(rkaiq_cam_eng)
            V4L2Camera(frame);

        //Center crop to resize to 224x224
        const int s = std::min(frame.cols, frame.rows);
        const cv::Rect roi((frame.cols-s) / 2, (frame.rows-s) / 2, s, s);
        last_frame = frame(roi);

        if(!rtClassifier.doAsyncStep(frame(roi)))
            nSkipFrames++;
    }
    rtClassifier.waitForCompletion();

    //For debugging purpose
    if (!last_frame.empty())
    {
        cv::resize(last_frame, last_frame, cv::Size(224, 224));
        cv::imwrite((home_dir / "mobilenet_rock5/debug_frame_final.jpg").string(), last_frame);
    }

    const auto benchEnd = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration_cast<std::chrono::milliseconds>(benchEnd-benchStart).count()/1000.0;
    const long total = nFrames.load();
    const long infer = nInferFrames.load();
    const long skip_frame = nSkipFrames.load();
    const long infer_time = total_infer.load();

    std::cout << "--- Benchmark Results ---\n";
    std::cout << "Duration: " << seconds << "s\n";
    std::cout << "Camera frames: " << total << " (" << total / seconds << "fps)\n";
    std::cout << "Inferences started: " << infer << " (" << infer / seconds << "Hz)\n";
    std::cout << "Frames skipped: " << skip_frame << "\n";
    
    if(infer > 0)
        std::cout << "Avg inference time: " << (infer_time / infer) << "ms\n";

    return 0;
}