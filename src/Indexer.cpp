#include "Indexer.hpp"

#include <fstream>
#include <iostream>

void Indexer::buildIndex(const std::vector<std::vector<float>>& tfidfVectors, const std::vector<std::string>& imagePaths)
{
    vectors_ = tfidfVectors;
    image_paths_ = imagePaths;
}

bool Indexer::saveIndex(const std::string& filename) const
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) return false;

    int n = static_cast<int>(vectors_.size());
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(int));
    if (n == 0) {
        ofs.close();
        return true;
    }

    int dim = static_cast<int>(vectors_[0].size());
    ofs.write(reinterpret_cast<const char*>(&dim), sizeof(int));

    for (const auto& v : vectors_) {
        ofs.write(reinterpret_cast<const char*>(v.data()), sizeof(float) * v.size());
    }

    int m = static_cast<int>(image_paths_.size());
    ofs.write(reinterpret_cast<const char*>(&m), sizeof(int));
    for (const auto& p : image_paths_) {
        int len = static_cast<int>(p.size());
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(int));
        ofs.write(p.data(), len);
    }

    ofs.close();
    return true;
}

bool Indexer::loadIndex(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) return false;

    int n = 0;
    ifs.read(reinterpret_cast<char*>(&n), sizeof(int));
    vectors_.clear();
    image_paths_.clear();
    if (n == 0) {
        ifs.close();
        return true;
    }

    int dim = 0;
    ifs.read(reinterpret_cast<char*>(&dim), sizeof(int));
    vectors_.resize(n, std::vector<float>(dim));

    for (int i = 0; i < n; ++i) {
        ifs.read(reinterpret_cast<char*>(vectors_[i].data()), sizeof(float) * dim);
    }

    int m = 0;
    ifs.read(reinterpret_cast<char*>(&m), sizeof(int));
    image_paths_.resize(m);
    for (int i = 0; i < m; ++i) {
        int len = 0;
        ifs.read(reinterpret_cast<char*>(&len), sizeof(int));
        std::string s(len, '\0');
        ifs.read(&s[0], len);
        image_paths_[i] = std::move(s);
    }

    ifs.close();
    return true;
}
