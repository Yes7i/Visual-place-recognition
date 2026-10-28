cmake_minimum_required(VERSION 3.10)
project(visual_place_recognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d flann imgcodecs highgui)

include_directories(${CMAKE_SOURCE_DIR}/include)

# 手动列出源文件
set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/FeatureExtractor.cpp
    ${CMAKE_SOURCE_DIR}/src/VocabularyBuilder.cpp
    ${CMAKE_SOURCE_DIR}/src/HistogramEncoder.cpp
    ${CMAKE_SOURCE_DIR}/src/Indexer.cpp
    ${CMAKE_SOURCE_DIR}/src/Searcher.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

add_executable(vpr ${SRC_FILES})

target_include_directories(vpr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(vpr PRIVATE ${OpenCV_LIBS})