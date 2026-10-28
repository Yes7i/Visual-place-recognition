#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>

// Write/read a single cv::Mat to/from a binary stream. These functions are low-level helpers used by FeatureExtractor.
bool writeBinaryMat(std::ofstream& ofs, const cv::Mat& mat);
bool readBinaryMat(std::ifstream& ifs, cv::Mat& mat);
