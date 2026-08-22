/**
 * Real-Time Classification Demo on CPU and NPU with a Small Monitor Display
 * (c) 2026 Bernd Porr and Jui Ning Chin
**/

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
#include <iomanip>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// Path of the RKNN model (NPU)
const fs::path rknn_model_path  = "mobilenet_rock5/models/mobilenetv2_features.rknn";

// Path of the ONNX model (CPU)
const fs::path onnx_model_path = "mobilenet_rock5/models/mobilenetv2_features.onnx";

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

int main(int argc, char *argv[])
{
    std::signal(SIGINT, sigHandler);
    bool npu_backend = true;

    for(int i = 1; i<argc; i++)
    {
        std::string arg(argv[i]);
        if(arg == "cpu")
            npu_backend = false;
    }

    const fs::path home_dir(getpwuid(getuid())->pw_dir);
    const fs::path model_path = home_dir / (npu_backend ? rknn_model_path : onnx_model_path);
    std::cout << "-I- Backend: " << (npu_backend ? "NPU" : "CPU") << "\n";

    RTClassifier rtClassifier(model_path.string(), (home_dir / classifier_model_path).string(), (int)classes.size(),npu_backend);

    // Per frame counters accessed from both the main and worker thread
    std::atomic<long> nFrames{0};
    std::atomic<long> nInferFrames{0};
    std::atomic<long> nSkipFrames{0};
    std::atomic<long> total_infer{0};
    std::atomic<int> last_label{0};
    std::atomic<int> last_prob_int{0};
    std::atomic<long> last_infer_ms{0};

    int last_pred_class = -1;
    int new_class = -1;
    int classPred_count = 0;

    const float min_prob = 0.75f;
    const int frame_thres = 3;

    // Inference result callback
    rtClassifier.classificationCallback = [&](int pred_class, float pred_score, long duration)
    {
        nInferFrames.fetch_add(1);
        total_infer += duration;
        last_label.store(pred_score > min_prob ? pred_class : -1);
        last_prob_int.store(static_cast<int>(pred_score * 100.0f));
        last_infer_ms.store(duration);

        // Avoid spamming the terminal with output messages
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

        if (pred_class == -1)
        {
            std::cout << "-I- Class Name: Not detected" << ", Prediction score: " << (int)(pred_score*100) << "%, Inference Time: " << duration << "ms\n" << std::flush;
        }
        else
        {
            std::cout << "-I- Class Name: " << classes[pred_class] << ", Prediction score: " << (int)(pred_score*100) << "%, Inference Time: " << duration << "ms\n" << std::flush;
        }
    };

    // Increase gain
    setV4L2OpenCVParam(
        "/dev/v4l-subdev2",
        V4L2_CID_GAIN,
        3500
    );

    cv::VideoCapture cam(cam_dev, cv::CAP_V4L2);
    if (!cam.isOpened()) 
    {
        std::cerr << "-E- Failed to open camera: " << cam_dev << "\n";
        return 1;
    }
    
    // Camera settings
    cam.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('N','V','1','2')); //Pixel format
    cam.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cam.set(cv::CAP_PROP_FPS, 30);
    cam.set(cv::CAP_PROP_CONVERT_RGB, 1);
    std::cout << "Camera: " << cam.getBackendName() << std::endl;
    std::cout << "-I- Camera frame size: " << cam.get(cv::CAP_PROP_FRAME_WIDTH) << ", " << cam.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    // Open framebuffer device for display
    int fb_dev = open("/dev/fb0", O_RDWR);
    bool has_display = (fb_dev>=0);

    // Run one silient inference so the NPU context is fully initialised before the timed loop starts
    cv::Mat initFrame(480,640, CV_8UC3, cv::Scalar(0,0,0));
    int cl;
    float cp;
    rtClassifier.doSyncStep(initFrame,cl,cp);
    
    std::cout << "-I- Running real-time classification for demo...\n";
    cv::Mat frame;
    cv::Mat last_frame;
    const auto benchStart = std::chrono::steady_clock::now();

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

        last_frame = frame.clone();

        if(has_display){
            cv::Mat display = frame.clone();

            //Add classification overlay
            const int pred = last_label.load();
            const int prob = last_prob_int.load();
            const long infer_ms = last_infer_ms.load();
            //const long rTotal = nFrames.load();
            //const long rDrop = nSkipFrames.load();

            std::string label_text = (pred>=0 && pred<(int)classes.size()) ? classes[pred] + " (" + std::to_string(prob) + "%)" : "Not detected";

            //Black bcakground box
            cv::rectangle(display,cv::Point(10,10),cv::Point(400,90),cv::Scalar(0,0,0),-1);

            // Text overlay
            cv::putText(display, label_text, cv::Point(20,50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 3);
            cv::putText(display, "Inference: " + std::to_string(infer_ms) + "ms", cv::Point(20,80), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200,200,200), 1);
            //cv::putText(display, "Frames dropped: (" + std::to_string(rDrop) + "/" + std::to_string(rTotal) + ")", cv::Point(20,display.rows-20),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(255,255,255),1);

            cv::Mat fb_img;
            cv::resize(display,fb_img,cv::Size(1280,720));
            cv::cvtColor(fb_img, fb_img, cv::COLOR_BGR2BGRA);

            // Write line-by-line respecting framebuffer stride
            const int fb_line_length = 5120;  // From fbset output
            const int img_line_bytes = 1280 * 4;  // 1280 pixels * 4 bytes (BGRA)
            
            lseek(fb_dev, 0, SEEK_SET);  // Reset to start of framebuffer

            for(int y = 0; y < 720; y++) {
                write(fb_dev, fb_img.ptr(y), img_line_bytes);
                // Skip padding bytes to next line
                if(fb_line_length > img_line_bytes) {
                    lseek(fb_dev, fb_line_length - img_line_bytes, SEEK_CUR);
                }
            }
        }


        //Center crop to resize to 224x224
        //const int s = std::min(frame.cols, frame.rows);
        //const cv::Rect roi((frame.cols-s) / 2, (frame.rows-s) / 2, s, s);
        //last_frame = frame.clone();
        

        const long rTotal = nFrames.load();
        const long rInfer = nInferFrames.load();
        const long rDrop_frame = nSkipFrames.load();
        const double rDrop_pct = rTotal > 0 ? 100.0 * rDrop_frame / rTotal : 0.0;
        const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-benchStart).count() / 1000.0;
        const double rFps = elapsed > 0 ? rTotal / elapsed : 0.0;
        static auto lastPrint = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();

        if(now-lastPrint > std::chrono::seconds(1))
        {
            std::cout << "[RT] Frames=" << rTotal << ", Inferred=" << rInfer << ", Dropped=" << rDrop_frame << " (" << std::fixed << std::setprecision(1) << rDrop_pct << "%)" << ", FPS=" << rFps << "\n" << std::flush;
            lastPrint = now;
        }

        if(!rtClassifier.doAsyncStep(frame))
            nSkipFrames++;
    }
    if(has_display)
        close(fb_dev);
    rtClassifier.waitForCompletion();
    std::cout << "\n";

    // Save the last captured frame for debugging purpose
    if (!last_frame.empty())
    {
        cv::imwrite((home_dir / "mobilenet_rock5/debug_frame_final.jpg").string(), last_frame);
    }

    const auto benchEnd = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration_cast<std::chrono::milliseconds>(benchEnd-benchStart).count()/1000.0;
    const long total = nFrames.load();
    const long infer = nInferFrames.load();
    const long drop_frame = nSkipFrames.load();
    const long infer_time = total_infer.load();

    std::cout << "--- Benchmark Results (" << (npu_backend ? "NPU" : "CPU") << ")---\n";
    std::cout << "Duration: " << seconds << "s\n";
    std::cout << "Camera frames: " << total << " (" << total / seconds << "fps)\n";
    std::cout << "Inferences: " << infer << " (" << infer / seconds << "Hz)\n";
    std::cout << "Frames dropped: " << drop_frame << "\n";
    
    if(infer > 0)
        std::cout << "Avg inference time: " << (infer_time / infer) << "ms\n";

    return 0;
}