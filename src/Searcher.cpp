#include "Searcher.hpp"
#include "Indexer.hpp"
#include "FeatureExtractor.hpp"
#include "HistogramEncoder.hpp"

#include <algorithm>
#include <numeric>

Searcher::Searcher(const Indexer& index)
    : index_(index)
{
}

std::vector<std::pair<std::string, float>> Searcher::search(const std::vector<float>& queryVector, int topK) const
{
    std::vector<std::pair<std::string, float>> results;
    const auto& db = index_.getVectors();
    const auto& paths = index_.getImagePaths();
    int n = static_cast<int>(db.size());
    if (n == 0 || queryVector.empty()) return results;

    int dim = static_cast<int>(queryVector.size());

    // compute dot products (cosine since vectors are normalized)
    std::vector<std::pair<float,int>> scores;
    scores.reserve(n);
    for (int i = 0; i < n; ++i) {
        const auto& v = db[i];
        if (static_cast<int>(v.size()) != dim) { scores.emplace_back(0.0f, i); continue; }
        float s = 0.0f;
        for (int d = 0; d < dim; ++d) s += v[d] * queryVector[d];
        scores.emplace_back(s, i);
    }

    // use nth_element to partition topK largest (guard topK == n, where nth == end() would be UB)
    auto cmp = [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first > b.first; };
    if (topK >= n) {
        std::sort(scores.begin(), scores.end(), cmp);
    } else {
        std::nth_element(scores.begin(), scores.begin() + topK, scores.end(), cmp);
        std::sort(scores.begin(), scores.begin() + topK, cmp);
    }

    results.reserve(topK);
    for (int i = 0; i < topK; ++i) {
        int idx = scores[i].second;
        float sc = scores[i].first;
        results.emplace_back(paths[idx], sc);
    }

    return results;
}

std::vector<std::pair<std::string, float>> Searcher::searchFromImage(const std::string& imagePath,
                                                                      FeatureExtractor& extractor,
                                                                      HistogramEncoder& encoder,
                                                                      int topK) const
{
    cv::Mat desc = extractor.extractFromImage(imagePath);
    std::vector<float> hist = encoder.encodeImage(desc);
    return search(hist, topK);
}
