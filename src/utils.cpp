#include "utils.hpp"
#include <iostream>

bool writeBinaryMat(std::ofstream& ofs, const cv::Mat& mat)
{
    if (!ofs.good()) return false;
    int rows = mat.rows;
    int cols = mat.cols;
    int type = mat.type();
    size_t dataSize = mat.total() * mat.elemSize();

    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&type), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&dataSize), sizeof(size_t));

    if (dataSize > 0)
        ofs.write(reinterpret_cast<const char*>(mat.data), dataSize);

    return ofs.good();
}

bool readBinaryMat(std::ifstream& ifs, cv::Mat& mat)
{
    if (!ifs.good()) return false;
    int rows = 0, cols = 0, type = 0;
    size_t dataSize = 0;

    ifs.read(reinterpret_cast<char*>(&rows), sizeof(int));
    if (ifs.eof()) return false;
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&type), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&dataSize), sizeof(size_t));

    if (rows <= 0 || cols <= 0) {
        mat.release();
        return true;
    }

    mat.create(rows, cols, type);
    if (dataSize > 0)
        ifs.read(reinterpret_cast<char*>(mat.data), dataSize);

    return ifs.good();
}
