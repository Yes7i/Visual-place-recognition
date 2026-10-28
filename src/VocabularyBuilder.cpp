#include "VocabularyBuilder.hpp"

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <iostream>

VocabularyBuilder::VocabularyBuilder(int totalFeatures, int dictionarySize)
    : total_features_(totalFeatures), dictionary_size_(dictionarySize)
{
}

bool VocabularyBuilder::buildVocabulary(const std::vector<cv::Mat>& allFeatures)
{
    // Concatenate all descriptors into one Mat (CV_32F)
    int dim = 0;
    size_t totalRows = 0;
    for (const auto& m : allFeatures) {
        if (m.empty()) continue;
        dim = m.cols; break;
    }
    for (const auto& m : allFeatures) if (!m.empty()) totalRows += m.rows;

    if (totalRows == 0 || dim == 0) return false;

    cv::Mat data(static_cast<int>(totalRows), dim, CV_32F);
    int r = 0;
    for (const auto& m : allFeatures) {
        if (m.empty()) continue;
        cv::Mat tmp;
        m.convertTo(tmp, CV_32F);
        tmp.copyTo(data.rowRange(r, r + tmp.rows));
        r += tmp.rows;
    }

    // kmeans parameters
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.01);
    int attempts = 3;
    int flags = cv::KMEANS_PP_CENTERS;

    // KMeans++ 初始化内部使用全局 RNG (cv::theRNG())。
    // 固定种子保证同样的训练数据总能得到同样的词典（词典“固定不变”的要求）。
    cv::theRNG() = cv::RNG(20240815u);

    cv::Mat labels;
    centers_ = cv::Mat();
    cv::kmeans(data, dictionary_size_, labels, criteria, attempts, flags, centers_);

    // Build FLANN index for centers
    if (centers_.empty()) return false;
    // Ensure CV_32F
    centers_.convertTo(centers_, CV_32F);
    // 单树 KD-Tree + 精确搜索，与 HistogramEncoder 保持一致（确定性）
    cv::flann::KDTreeIndexParams indexParams(1);
    flann_index_ = cv::makePtr<cv::flann::Index>(centers_, indexParams, cvflann::FLANN_DIST_L2);

    return true;
}

bool VocabularyBuilder::saveVocabulary(const std::string& filename) const
{
    if (centers_.empty()) return false;
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "dictionary_size" << dictionary_size_;
    fs << "centers" << centers_;
    fs.release();
    return true;
}

bool VocabularyBuilder::loadVocabulary(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["dictionary_size"] >> dictionary_size_;
    fs["centers"] >> centers_;
    fs.release();

    if (centers_.empty()) return false;
    centers_.convertTo(centers_, CV_32F);
    cv::flann::KDTreeIndexParams indexParams(1);
    flann_index_ = cv::makePtr<cv::flann::Index>(centers_, indexParams, cvflann::FLANN_DIST_L2);
    return true;
}

int VocabularyBuilder::assignWord(const cv::Mat& descriptor) const
{
    if (centers_.empty() || descriptor.empty()) return -1;
    cv::Mat query;
    descriptor.convertTo(query, CV_32F);
    if (query.cols != centers_.cols) query = query.reshape(1, 1);

    std::vector<int> indices(1);
    std::vector<float> dists(1);
    flann_index_->knnSearch(query, indices, dists, 1, cv::flann::SearchParams(-1));
    return indices.empty() ? -1 : indices[0];
}
