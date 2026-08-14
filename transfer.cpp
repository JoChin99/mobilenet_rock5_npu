#include "mobilenetv2_features.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <pwd.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstring>

namespace fs = std::filesystem;

// Path of the Kaggle dataset
const fs::path datasetpath = "mobilenet_rock5/data/2d-geometric-shapes-17-shapes/2D_Geometric_Shapes_Dataset/";

// Path of the RKNN model
const fs::path rknn_model_path  = "mobilenet_rock5/models/mobilenetv2_features.rknn";

// Path to the classifier file
const char classifier_model_path[] = "mobilenet_rock5/models/classifier.pt";

// Path to the loss log file
const char loss_file[] = "mobilenet_rock5/models/loss.dat";

// Subdirs of the classes
const std::vector<fs::path> classes = {"circle", "heart", "star"};

// The batch size for training
const int batch_size = 16;

// The number of epochs
const int epochs = 30;

// The number of learning rate
const double lr = 1e-3;

// Dataset implementation
struct ImageFolderDataset : torch::data::Dataset<ImageFolderDataset>
{
    struct Sample
    {
        fs::path image_path;
        int label;
    };

    std::vector<Sample> samples;

    ImageFolderDataset(const fs::path &root, const std::vector<fs::path> &classes)
    {
        for (size_t label = 0; label < classes.size(); label++)
        {
            const fs::path class_path = root / classes[label];
            for (const auto &p : fs::directory_iterator(class_path))
            {
                if (p.is_regular_file())
                {
                    samples.push_back({p.path(), (int)label});
                }
            }
        }
        std::cout << "Loaded " << samples.size() << " samples from " << datasetpath.string() << "\n";
    }
    
    torch::data::Example<> get(size_t idx) override
    {
        const auto &sample = samples[idx];
        const cv::Mat img = cv::imread(sample.image_path.string());
        if (img.empty())
        {
            throw std::runtime_error("Failed to load image: " + sample.image_path.string());
        }
        const torch::Tensor data = MobileNetV2Features::preprocess(img);
        const torch::Tensor label = torch::tensor(sample.label, torch::kLong);
        return {data, label};
    }

    torch::optional<size_t> size() const override
    {
        return samples.size();
    }
};

void progress(int epoch, int epochs, double loss, float f)
{
    std::cout << "Epoch [" << epoch << "/" << epochs << "], Loss: "
              << loss << "\t\t" << f << "Hz" << "\r" << std::flush;
}

struct MobileNetV2Classifier : torch::nn::Module
{
    const char *classifierModuleName = "classifier";
    MobileNetV2Classifier(int nFeatures, int nClasses)
    {
        sequ = torch::nn::Sequential(
            torch::nn::Dropout(0.2),
            torch::nn::Linear(nFeatures, nClasses));
        register_module(classifierModuleName, sequ);

        for (auto &module : sequ->modules(/*include_self=*/false))
        {
            if (auto M = dynamic_cast<torch::nn::LinearImpl *>(module.get()))
            {
                torch::nn::init::normal_(M->weight, 0.0, 0.01);
                torch::nn::init::zeros_(M->bias);
            }
        }
    }

    /** 
     * @brief Save classifier weights as a pickled key/tensor dict with converter
     * To solve the difference naming conventions between pytorch and libtorch.
     * This is to ensure the key/parameters pairs are loaded into named parameters properly.
    */
    void save_classifier_weights(const std::string& pt_path)
    {
        c10::Dict<std::string, torch::Tensor> weights;

        std::cout << "-I- Saving the parameters from the model: " << "\n";
        for (const auto &p : sequ->named_parameters())
        {
            weights.insert(p.key(), p.value().detach().cpu().contiguous());
            std::cout << p.key() << " , " << p.value().sizes() << "\n";
        }

        const std::vector<char> bytes = torch::pickle_save(weights);
        std::ofstream output(pt_path, std::ios::binary);
        if (!output)
            throw std::runtime_error("-E- Failed to open file for writing: " + pt_path);
        output.write(bytes.data(), bytes.size());
        output.close();
    }
    torch::nn::Sequential sequ{nullptr};
};

// Main training program
int main()
{
    torch::manual_seed(42);
    torch::Device device(torch::kCPU);

    const fs::path homedir(getpwuid(getuid())->pw_dir);
    ImageFolderDataset ds(homedir / datasetpath, classes);

    // Creates a DataLoader instance for a stateless dataset.
    auto loader = torch::data::make_data_loader(
        ds.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size));

    // Model setup
    MobileNetV2Features features(homedir / rknn_model_path.string());
    MobileNetV2Classifier classifier(features.N_OUTPUT_FEATURES, classes.size());

    // Optimizer only for classifier
    torch::optim::SGD optimizer(classifier.sequ->parameters(), torch::optim::SGDOptions(lr));
    torch::optim::StepLR scheduler(optimizer, /*step_size=*/15, /*gamma=*/0.1);
    torch::nn::CrossEntropyLoss criterion;

    // Logging of the loss
    std::fstream floss;
    const fs::path loss_path = homedir / loss_file;
    floss.open(loss_path, std::fstream::out);

    float f = 0;
    // Training loop
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        float cumloss = 0;
        classifier.train();
        int n = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (auto &batch : *loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            optimizer.zero_grad();
            // mobilenetv2 feature detector (without learning and pre-trained weights)
            auto fout = features.forward(data);
            // features are in a 7x7x1280 grid and need to be collapsed to just 1280 features
            auto output = classifier.sequ->forward(fout);
            auto loss = criterion(output, target);
            loss.backward();
            optimizer.step();
            auto current = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
            f = n * 1000 / (float)duration.count();
            progress(epoch, epochs, loss.item<double>(), f);
            cumloss += loss.item<double>();
            n++;
        }
        const double avgLoss = cumloss / (double)n;
        progress(epoch, epochs, avgLoss, f);
        floss << epoch << "\t" << avgLoss << std::endl;
        scheduler.step();
        std::cout << std::endl
                  << std::flush;
    }
    const fs::path classifier_path = homedir / classifier_model_path;
    classifier.save_classifier_weights(classifier_path.string());
    std::cout << "Done.\n";

    return 0;
}