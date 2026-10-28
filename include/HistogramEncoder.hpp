#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class HistogramEncoder {
public:
    explicit HistogramEncoder(const cv::Mat& centroids);

    // Encode descriptors into a histogram (length == dictionary size). Result is L2-normalized.
    std::vector<float> encodeImage(const cv::Mat& descriptors) const;

    // Compute IDF weights from all histograms (each histogram is float vector length == dictionary size)
    std::vector<float> computeIDF(const std::vector<std::vector<float>>& allHistograms) const;

    // Apply TF-IDF to a histogram in-place and L2-normalize
    void applyTFIDF(std::vector<float>& histogram, const std::vector<float>& idfWeights) const;

    // L2-normalize a vector in-place
    static void normalizeVector(std::vector<float>& vec);

    int dictionarySize() const { return dictionary_size_; }

private:
    cv::Mat centers_;
    int dictionary_size_ = 0;

    // FLANN for nearest-neighbor assignment
    cv::Ptr<cv::flann::Index> flann_index_;
    // soft-assignment parameters: top-k neighbors to consider
    int assign_k_ = 3;
public:
    void setAssignK(int k) { if (k > 0) assign_k_ = k; }
};
