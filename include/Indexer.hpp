#pragma once

#include <vector>
#include <string>

class Indexer {
public:
    Indexer() = default;

    // Store TF-IDF vectors and corresponding image paths
    void buildIndex(const std::vector<std::vector<float>>& tfidfVectors, const std::vector<std::string>& imagePaths);

    // Binary save/load
    bool saveIndex(const std::string& filename) const;
    bool loadIndex(const std::string& filename);

    const std::vector<std::vector<float>>& getVectors() const { return vectors_; }
    const std::vector<std::string>& getImagePaths() const { return image_paths_; }

private:
    std::vector<std::vector<float>> vectors_;
    std::vector<std::string> image_paths_;
};
