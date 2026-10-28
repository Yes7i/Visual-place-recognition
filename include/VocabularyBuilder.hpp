#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class VocabularyBuilder {
public:
    VocabularyBuilder(int totalFeatures, int dictionarySize = 1000);

    // Build vocabulary (cluster centers) from allFeatures (concatenation of descriptors from many images)
    // allFeatures: vector of Mats, each row is a descriptor (CV_32F)
    bool buildVocabulary(const std::vector<cv::Mat>& allFeatures);

    // Save/load vocabulary centers as YAML
    bool saveVocabulary(const std::string& filename) const;
    bool loadVocabulary(const std::string& filename);

    // Assign a single descriptor to nearest visual word index
    int assignWord(const cv::Mat& descriptor) const;

private:
    int total_features_ = 0;
    int dictionary_size_ = 1000;
    cv::Mat centers_; // dictionary_size_ x dim (CV_32F)

    // FLANN index for nearest-neighbor search
    cv::Ptr<cv::flann::Index> flann_index_;
};
