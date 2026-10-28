#include "HistogramEncoder.hpp"

#include <opencv2/flann.hpp>
#include <cmath>
#include <iostream>

HistogramEncoder::HistogramEncoder(const cv::Mat& centroids)
{
    if (centroids.empty()) return;
    centroids.convertTo(centers_, CV_32F);
    dictionary_size_ = centers_.rows;
    // 单树 KD-Tree + checks=-1 精确搜索：
    // cvflann 的多树 KD-Tree 用全局 RNG 随机建树，checks 有限时是近似搜索，
    // 结果会随进程内 RNG 状态变化 → 同一张图两次编码不一致（自检索得分 <1）。
    // 单树 + 精确搜索与建树结构无关，保证编码完全确定且最接近“最近单词”语义。
    cv::flann::KDTreeIndexParams indexParams(1);
    flann_index_ = cv::makePtr<cv::flann::Index>(centers_, indexParams, cvflann::FLANN_DIST_L2);
}

std::vector<float> HistogramEncoder::encodeImage(const cv::Mat& descriptors) const
{
    std::vector<float> hist(dictionary_size_, 0.0f);
    if (centers_.empty() || descriptors.empty()) return hist;

    cv::Mat desc32;
    descriptors.convertTo(desc32, CV_32F);

    // For each descriptor, find top-k nearest centroids using FLANN and apply soft assignment
    int k = std::min(assign_k_, dictionary_size_);
    std::vector<int> indices(k);
    std::vector<float> dists(k);
    for (int i = 0; i < desc32.rows; ++i) {
        cv::Mat q = desc32.row(i);
        std::fill(indices.begin(), indices.end(), -1);
        std::fill(dists.begin(), dists.end(), 0.0f);
        flann_index_->knnSearch(q, indices, dists, k, cv::flann::SearchParams(-1)); // -1 = FLANN_CHECKS_UNLIMITED 精确搜索

        // compute inverse-distance weights
        double sumw = 0.0;
        std::vector<double> weights(k, 0.0);
        const double eps = 1e-6;
        for (int j = 0; j < k; ++j) {
            double d = static_cast<double>(dists[j]); // squared L2 distance from FLANN
            double w = 1.0 / (d + eps);
            weights[j] = w;
            sumw += w;
        }
        if (sumw <= 0.0) continue;
        for (int j = 0; j < k; ++j) {
            int idx = indices[j];
            if (idx >= 0 && idx < dictionary_size_) {
                hist[idx] += static_cast<float>(weights[j] / sumw);
            }
        }
    }

    normalizeVector(hist);
    return hist;
}

std::vector<float> HistogramEncoder::computeIDF(const std::vector<std::vector<float>>& allHistograms) const
{
    std::vector<float> idf(dictionary_size_, 0.0f);
    if (dictionary_size_ == 0) return idf;

    int N = static_cast<int>(allHistograms.size());
    if (N == 0) return idf;

    std::vector<int> df(dictionary_size_, 0);
    for (const auto& h : allHistograms) {
        for (int j = 0; j < dictionary_size_ && j < static_cast<int>(h.size()); ++j) {
            if (h[j] > 0.0f) df[j] += 1;
        }
    }

    for (int j = 0; j < dictionary_size_; ++j) {
        idf[j] = std::log(static_cast<double>(N) / (1.0 + static_cast<double>(df[j])));
    }
    return idf;
}

void HistogramEncoder::applyTFIDF(std::vector<float>& histogram, const std::vector<float>& idfWeights) const
{
    if (static_cast<int>(histogram.size()) != dictionary_size_ || static_cast<int>(idfWeights.size()) != dictionary_size_) return;
    for (int i = 0; i < dictionary_size_; ++i) histogram[i] *= idfWeights[i];
    normalizeVector(histogram);
}

void HistogramEncoder::normalizeVector(std::vector<float>& vec)
{
    double s = 0.0;
    for (float v : vec) s += static_cast<double>(v) * static_cast<double>(v);
    double n = std::sqrt(s);
    if (n <= 1e-12) return;
    for (auto& v : vec) v = static_cast<float>(v / n);
}
