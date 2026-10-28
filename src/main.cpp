#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cstdlib>

#include "FeatureExtractor.hpp"
#include "VocabularyBuilder.hpp"
#include "HistogramEncoder.hpp"
#include "Indexer.hpp"
#include "Searcher.hpp"

namespace fs = std::filesystem;

static std::vector<std::string> collectImagePaths(const std::string& folder)
{
    std::vector<std::string> paths;
    if (!fs::exists(folder)) return paths;
    for (auto const& entry : fs::recursive_directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") paths.push_back(entry.path().string());
    }
    return paths;
}

static void ensureOutputDir()
{
    fs::create_directories("output");
}

// 生成网页中图片的相对路径(相对 output/ 目录),file:// 与 http:// 打开都能正常显示
static std::string htmlImageSrc(const std::string& p)
{
    try { return fs::relative(p, "output").generic_string(); }
    catch (...) { return "../" + p; }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./vpr --build | --query <image_path>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    ensureOutputDir();

    if (mode == "--build") {
        std::cout << "Build mode: extracting features and building index..." << std::endl;
        auto imagePaths = collectImagePaths("data/Freiburg");
        if (imagePaths.empty()) {
            std::cerr << "No images found in data/Freiburg" << std::endl;
            return 1;
        }

        FeatureExtractor extractor("", "config/config.yaml");
        std::cout << "[INFO] dictionary_size (K) = " << extractor.dictionarySize() << std::endl;
        std::vector<cv::Mat> perImageDescriptors;
        perImageDescriptors.reserve(imagePaths.size());

        // Extract features
        for (const auto& p : imagePaths) {
            cv::Mat d = extractor.extractFromImage(p);
            perImageDescriptors.push_back(d);
        }

        // Save raw descriptors
        extractor.saveFeatures(perImageDescriptors, "output/features.bin");

        // Count total features
        int totalFeatures = 0;
        for (const auto& m : perImageDescriptors) totalFeatures += (m.empty() ? 0 : m.rows);

        VocabularyBuilder builder(totalFeatures, extractor.dictionarySize());
        if (!builder.buildVocabulary(perImageDescriptors)) {
            std::cerr << "Failed to build vocabulary" << std::endl;
            return 1;
        }
        builder.saveVocabulary("output/vocabulary.yml");

        // Load centers to build encoder
        cv::Mat centers;
        {
            cv::FileStorage fsCenters("output/vocabulary.yml", cv::FileStorage::READ);
            fsCenters["centers"] >> centers;
            fsCenters.release();
        }
        HistogramEncoder encoder(centers);
        // allow overriding assignment k via environment variable ASSIGN_K
        if (const char* env_k = std::getenv("ASSIGN_K")) {
            try {
                int k = std::stoi(env_k);
                encoder.setAssignK(k);
                std::cerr << "[INFO] ASSIGN_K=" << k << std::endl;
            } catch (...) { }
        }

        // Encode all images and compute IDF
        std::vector<std::vector<float>> allHistograms;
        allHistograms.reserve(perImageDescriptors.size());
        for (const auto& m : perImageDescriptors) {
            auto h = encoder.encodeImage(m);
            allHistograms.push_back(h);
        }

        auto idfWeights = encoder.computeIDF(allHistograms);
        // Save idf
        {
            cv::Mat idfMat(1, static_cast<int>(idfWeights.size()), CV_32F);
            for (int i = 0; i < static_cast<int>(idfWeights.size()); ++i) idfMat.at<float>(0,i) = idfWeights[i];
            cv::FileStorage fsIdf("output/idf.yml", cv::FileStorage::WRITE);
            fsIdf << "idf" << idfMat;
            fsIdf.release();
        }

        // Apply TF-IDF and build tfidf vectors
        std::vector<std::vector<float>> tfidfVectors;
        tfidfVectors.reserve(allHistograms.size());
        for (auto h : allHistograms) {
            encoder.applyTFIDF(h, idfWeights);
            tfidfVectors.push_back(std::move(h));
        }

        Indexer indexer;
        indexer.buildIndex(tfidfVectors, imagePaths);
        indexer.saveIndex("output/tfidf_vectors.bin");

        std::cout << "Build complete. Outputs in output/" << std::endl;
        return 0;
    }
    else if (mode == "--query") {
        if (argc < 3) { std::cerr << "Missing image path for --query" << std::endl; return 1; }
        std::string queryPath = argv[2];

        // Load index
        Indexer indexer;
        if (!indexer.loadIndex("output/tfidf_vectors.bin")) { std::cerr << "Failed to load index" << std::endl; return 1; }

        // Load vocabulary centers
        cv::Mat centers;
        {
            cv::FileStorage fsCenters("output/vocabulary.yml", cv::FileStorage::READ);
            if (!fsCenters.isOpened()) { std::cerr << "Failed to open vocabulary file" << std::endl; return 1; }
            fsCenters["centers"] >> centers;
            fsCenters.release();
        }
        HistogramEncoder encoder(centers);
        std::cout << "[INFO] loaded vocabulary K = " << centers.rows << std::endl;
        if (const char* env_k = std::getenv("ASSIGN_K")) {
            try {
                int k = std::stoi(env_k);
                encoder.setAssignK(k);
                std::cerr << "[INFO] ASSIGN_K=" << k << std::endl;
            } catch (...) { }
        }

        // Load idf
        std::vector<float> idfWeights;
        {
            cv::FileStorage fsIdf("output/idf.yml", cv::FileStorage::READ);
            cv::Mat idfMat;
            fsIdf["idf"] >> idfMat;
            fsIdf.release();
            if (!idfMat.empty()) {
                idfWeights.resize(idfMat.cols);
                for (int i = 0; i < idfMat.cols; ++i) idfWeights[i] = idfMat.at<float>(0,i);
            }
        }

        FeatureExtractor extractor("", "config/config.yaml");
        cv::Mat desc = extractor.extractFromImage(queryPath);
        auto hist = encoder.encodeImage(desc);
        if (!idfWeights.empty()) encoder.applyTFIDF(hist, idfWeights);

        Searcher searcher(indexer);
        auto results = searcher.search(hist, extractor.topK());

        // Generate HTML
        std::ofstream ofs("output/results.html");
        ofs << "<!doctype html><html><head><meta charset=\"utf-8\"><title>VPR Results</title>";
        ofs << "<style>body{font-family:Arial,Helvetica,sans-serif;background:#f6f8fa;color:#222;margin:16px} .kbar{display:flex;align-items:center;gap:12px;background:#fff;border:1px solid #e6e9ee;border-radius:8px;padding:10px 14px;margin-bottom:18px;flex-wrap:wrap} .kbar b{color:#1e88e5} .kbar input{width:90px;padding:5px 8px;border:1px solid #ccc;border-radius:5px;font-size:14px} .kbar button{background:#1e88e5;color:#fff;border:none;border-radius:5px;padding:6px 14px;font-size:14px;cursor:pointer} .kbar button:hover{background:#1565c0} .kbar .hint{color:#888;font-size:12px} .header{display:flex;align-items:center;gap:20px} .query-img{max-width:360px;border-radius:6px;border:1px solid #ddd;padding:6px;background:#fff} .results{display:flex;flex-wrap:wrap;gap:14px;margin-top:18px} .card{width:220px;background:#fff;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.06);padding:10px;border:1px solid #e6e9ee;text-align:center} .rank{display:inline-block;background:#1e88e5;color:#fff;padding:6px 10px;border-radius:18px;font-weight:700;margin-bottom:8px} .filename{display:block;font-size:12px;color:#333;margin-top:8px;word-break:break-all} .score{font-size:13px;color:#2e7d32;font-weight:700;margin-top:6px} .qbar{display:flex;align-items:center;gap:10px;background:#fff;border:1px solid #e6e9ee;border-radius:8px;padding:10px 14px;margin-bottom:18px;flex-wrap:wrap} .qbar #q-path{width:340px;padding:5px 8px;border:1px solid #ccc;border-radius:5px;font-size:14px} .qbar button{background:#2e7d32;color:#fff;border:none;border-radius:5px;padding:6px 14px;font-size:14px;cursor:pointer} .qbar button:hover{background:#1b5e20} .qbar .hint{color:#888;font-size:12px} .filebtn{display:inline-block;background:#eef2f7;color:#1e88e5;border:1px solid #bcd3ea;border-radius:5px;padding:6px 12px;font-size:14px;cursor:pointer} .filebtn:hover{background:#e3ecf5}</style>";
        ofs << "</head><body>\n";
        // 顶部控制栏:显示当前聚类中心个数 + 填写新个数重新训练
        ofs << "<div class=\"kbar\">";
        ofs << "<span>当前聚类中心个数:<b id=\"cur-k\">" << centers.rows << "</b></span>";
        ofs << "<form action=\"/set_k\" method=\"post\" onsubmit=\"return checkProto()\">";
        ofs << "<span>修改为:</span>";
        ofs << "<input type=\"number\" name=\"dictionary_size\" min=\"1\" step=\"1\" value=\"" << centers.rows << "\" required>";
        ofs << "<button type=\"submit\">重新训练</button>";
        ofs << "</form>";
        ofs << "<span class=\"hint\" id=\"proto-hint\">未训练过的个数会自动训练;已训练过的个数直接复用缓存,无需等待</span>";
        ofs << "<span class=\"hint\" id=\"trained-hint\"></span>";
        ofs << "</div>\n";
        // 查询相似图片面板:图库下拉选择 / 本机图片上传 / 输入路径或文件名
        ofs << "<div class=\"qbar\">";
        ofs << "<span>查询相似图片:</span>";
        ofs << "<input id=\"q-path\" list=\"gallery-list\" placeholder=\"从图库选择,或输入图片路径/文件名\">";
        ofs << "<datalist id=\"gallery-list\"></datalist>";
        ofs << "<label class=\"filebtn\">选择本机图片<input type=\"file\" id=\"q-file\" accept=\"image/*\" hidden></label>";
        ofs << "<button type=\"button\" onclick=\"doQuery()\">查询相似图片</button>";
        ofs << "<span class=\"hint\" id=\"q-count\"></span>";
        ofs << "<span class=\"hint\" id=\"q-hint\"></span>";
        ofs << "</div>\n";
        ofs << "<div class=\"header\">";
        ofs << "<div><h1 style=\"margin:0;\">Query</h1>";
        ofs << "<div style=\"margin-top:8px;\"><img class=\"query-img\" src=\"" << htmlImageSrc(queryPath) << "\"></div></div>";
        ofs << "<div style=\"margin-left:12px;\"><h2 style=\"margin-top:0;\">Top Results</h2><div style=\"color:#555;\">Showing " << results.size() << " matches</div></div>";
        ofs << "</div>\n";

        ofs << "<div class=\"results\">\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto &r = results[i];
            int rankNum = static_cast<int>(i) + 1;
            std::string rankLabel;
            if (rankNum == 1) rankLabel = "1st";
            else if (rankNum == 2) rankLabel = "2nd";
            else if (rankNum == 3) rankLabel = "3rd";
            else rankLabel = std::to_string(rankNum) + "th";

            ofs << "<div class=\"card\">";
            ofs << "<div class=\"rank\">" << rankLabel << "</div>";
            ofs << "<div><img src=\"" << htmlImageSrc(r.first) << "\" style=\"width:200px;height:auto;border-radius:4px;margin-top:6px\"></div>";
            ofs << "<div class=\"filename\">" << r.first << "</div>";
            ofs << std::fixed << std::setprecision(4);
            ofs << "<div class=\"score\">score: " << r.second << "</div>";
            ofs << "</div>\n";
        }

        ofs << "</div>\n";
        ofs << "<script>"
               "function checkProto(){if(location.protocol==='file:'){alert('请先在项目目录运行 ./open_web.sh 打开本页,才能使用修改/查询功能');return false;}return true;}"
               "function showHint(t){document.getElementById('q-hint').textContent=t;}"
               "function doQuery(){"
                 "if(location.protocol==='file:'){alert('请用 ./open_web.sh 打开本页后再查询');return;}"
                 "showHint('');"
                 "var file=document.getElementById('q-file').files[0];"
                 "var path=document.getElementById('q-path').value.trim();"
                 "if(!file&&!path){showHint('请从图库选择图片、选择本机图片或输入图片名');return;}"
                 "if(file){"
                   "if(file.size>20971520){showHint('图片超过 20MB,无法上传');return;}"
                   "var xhr=new XMLHttpRequest();"
                   "xhr.open('POST','/upload?name='+encodeURIComponent(file.name));"
                   "xhr.onload=function(){if(xhr.status===200){location.href='/';}else{showHint('查询失败: '+xhr.responseText);}};"
                   "xhr.onerror=function(){showHint('上传失败: 网络错误');};"
                   "xhr.send(file);"
                 "}else{"
                   "var fd=new URLSearchParams();fd.append('image_path',path);"
                   "fetch('/query',{method:'POST',body:fd}).then(function(r){"
                     "if(r.ok){location.href='/';}else{return r.text().then(function(t){showHint('查询失败: '+t);});}"
                   "}).catch(function(e){showHint('查询失败: '+e);});"
                 "}"
               "}"
               "function initPage(){"
                 "if(location.protocol==='file:'){document.getElementById('proto-hint').textContent='提示:当前为文件方式打开,修改/查询功能不可用;请在项目目录运行 ./open_web.sh';return;}"
                 "fetch('/images').then(function(r){return r.json()}).then(function(d){"
                   "var dl=document.getElementById('gallery-list');"
                   "for(var i=0;i<d.images.length;i++){var o=document.createElement('option');o.value=d.images[i];dl.appendChild(o);}"
                   "var c=document.getElementById('q-count');c.textContent='图库共 '+d.images.length+' 张';"
                 "}).catch(function(){});"
                 "fetch('/trained').then(function(r){return r.json()}).then(function(d){"
                   "if(d.ks&&d.ks.length){document.getElementById('trained-hint').textContent='已训练过(直接复用,不重新训练):K='+d.ks.join('、');}"
                 "}).catch(function(){});"
               "}"
               "initPage();"
               "</script>\n";
        ofs << "</body></html>\n";
        ofs.close();

        std::cout << "Results written to output/results.html" << std::endl;
        return 0;
    }
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
}
