#include "FeatureExtractor.hpp"
#include "utils.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

namespace fs = std::filesystem;

FeatureExtractor::FeatureExtractor(const std::string& imagePath, const std::string& configPath)
{
    std::cerr << "[DEBUG] Entering FeatureExtractor constructor, configPath=" << configPath << std::endl;
    // Load configuration from YAML-like file using a simple parser to avoid FileStorage exceptions
    if (!configPath.empty()) {
        if (fs::exists(configPath)) {
            std::ifstream ifs(configPath);
            if (ifs.is_open()) {
                std::string line;
                while (std::getline(ifs, line)) {
                    // Trim leading spaces
                    size_t start = line.find_first_not_of(" \t\r\n");
                    if (start == std::string::npos) continue;
                    line = line.substr(start);
                    if (line.empty() || line[0] == '#') continue;
                    auto pos = line.find(':');
                    if (pos == std::string::npos) continue;
                    std::string key = line.substr(0, pos);
                    std::string value = line.substr(pos + 1);
                    // Trim spaces from value
                    size_t vstart = value.find_first_not_of(" \t\r\n");
                    size_t vend = value.find_last_not_of(" \t\r\n");
                    if (vstart == std::string::npos) value = "";
                    else value = value.substr(vstart, vend - vstart + 1);

                    if (key == "dictionary_size") dictionary_size_ = std::stoi(value);
                    else if (key == "flann_checks") flann_checks_ = std::stoi(value);
                    else if (key == "top_k") top_k_ = std::stoi(value);
                    else if (key == "feature_type") feature_type_ = value;
                }
                ifs.close();
                std::cerr << "[DEBUG] Finished parsing config file" << std::endl;
            } else {
                std::cerr << "Warning: cannot open config file: " << configPath << std::endl;
            }
        } else {
            std::cerr << "Warning: config file not found: " << configPath << std::endl;
        }
    }

    // Initialize SIFT (in modern OpenCV, SIFT is available in main contrib)
    sift_ = cv::SIFT::create();
    std::cerr << "[DEBUG] SIFT created successfully" << std::endl;
}

cv::Mat FeatureExtractor::extractFromImage(const std::string& imagePath)
{
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to open image: " << imagePath << std::endl;
        return cv::Mat();
    }

    std::vector<cv::KeyPoint> kps;
    cv::Mat descriptors;
    sift_->detectAndCompute(img, cv::noArray(), kps, descriptors);

    if (descriptors.empty()) return descriptors;

    // Ensure descriptors are CV_32F and have 128 columns
    descriptors.convertTo(descriptors, CV_32F);
    if (descriptors.cols != 128) {
        descriptors = descriptors.reshape(1, descriptors.rows);
    }

    // Apply RootSIFT: L1-normalize each descriptor then take square root element-wise
    for (int r = 0; r < descriptors.rows; ++r) {
        cv::Mat row = descriptors.row(r);
        double sum = 0.0;
        for (int c = 0; c < row.cols; ++c) sum += std::abs(row.at<float>(0, c));
        if (sum <= 1e-12) continue;
        for (int c = 0; c < row.cols; ++c) {
            float v = row.at<float>(0, c) / static_cast<float>(sum);
            // clamp small negatives to zero just in case
            if (v < 0.0f) v = 0.0f;
            row.at<float>(0, c) = std::sqrt(v);
        }
    }

    return descriptors;
}

std::vector<cv::Mat> FeatureExtractor::extractFromDataset(const std::string& folderPath)
{
    std::vector<cv::Mat> results;
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) return results;

    for (auto const& entry : fs::directory_iterator(folderPath)) {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png") {
            cv::Mat d = extractFromImage(path.string());
            results.push_back(d);
        }
    }

    return results;
}

bool FeatureExtractor::saveFeatures(const std::vector<cv::Mat>& features, const std::string& filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) return false;

    int count = static_cast<int>(features.size());
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(int));
    for (const auto& m : features) {
        if (!writeBinaryMat(ofs, m)) { ofs.close(); return false; }
    }
    ofs.close();
    return true;
}

std::vector<cv::Mat> FeatureExtractor::loadFeatures(const std::string& filename)
{
    std::vector<cv::Mat> out;
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) return out;

    int count = 0;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(int));
    for (int i = 0; i < count; ++i) {
        cv::Mat m;
        if (!readBinaryMat(ifs, m)) break;
        out.push_back(m);
    }

    ifs.close();
    return out;
}
