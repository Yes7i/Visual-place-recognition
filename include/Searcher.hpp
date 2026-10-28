#pragma once

#include <vector>
#include <string>
#include <utility>

class Indexer;
class FeatureExtractor;
class HistogramEncoder;

class Searcher {
public:
    explicit Searcher(const Indexer& index);

    // Search by query vector (assumes vectors are L2-normalized). Returns topK pairs (imagePath, score)
    std::vector<std::pair<std::string, float>> search(const std::vector<float>& queryVector, int topK) const;

    // Full pipeline: extract features from image, encode histogram, then search. Default topK=10
    std::vector<std::pair<std::string, float>> searchFromImage(const std::string& imagePath,
                                                               FeatureExtractor& extractor,
                                                               HistogramEncoder& encoder,
                                                               int topK = 10) const;

private:
    const Indexer& index_;
};
