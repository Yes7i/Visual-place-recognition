=#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class FeatureExtractor {
public:
    FeatureExtractor(const std::string& imagePath, const std::string& configPath);

    // Extract descriptors from a single image. Returns Nx128 CV_32F descriptors (one descriptor per row).
    cv::Mat extractFromImage(const std::string& imagePath);

    // Traverse folder and extract descriptors for each PNG image. Returns vector where each element is the descriptor Mat for one image.
    std::vector<cv::Mat> extractFromDataset(const std::string& folderPath);

    // Save/load features to a single binary file
    bool saveFeatures(const std::vector<cv::Mat>& features, const std::string& filename);
    std::vector<cv::Mat> loadFeatures(const std::string& filename);

    int dictionarySize() const { return dictionary_size_; }
    int topK() const { return top_k_ > 0 ? top_k_ : 10; }

private:
    int dictionary_size_ = 1000;
    int flann_checks_ = 32;
    int top_k_ = 10;
    std::string feature_type_ = "SIFT";

    cv::Ptr<cv::SIFT> sift_;
};
