# Visual Place Recognition (BoVW)

基于词袋模型 (Bag-of-Visual-Words) 的视觉位置识别：
SIFT 特征 → K-Means 视觉词典 → 词频直方图 → TF-IDF 加权 → 余弦距离检索。

## 依赖

- C++17
- OpenCV >= 4.0（组件：core, imgproc, features2d, flann, imgcodecs, highgui）
- CMake >= 3.10

## 数据集

代码仓库不包含图片，以避免把约 514 MB 的数据写入 Git 历史。请在 GitHub 仓库的 **Releases** 页面下载 `vpr-freiburg-reference-images.zip`。该包内含 692 张 PNG，解压到项目根目录后会生成 `data/Freiburg/`，这是程序读取图库的默认路径：

```bash
unzip vpr-freiburg-reference-images.zip
```

图片来源：[TUM RGB-D Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)。TUM 官方页面注明，除非另有说明，数据按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。该数据集论文引用信息：J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, “A Benchmark for the Evaluation of RGB-D SLAM Systems,” IROS 2012。发布数据包时请保留来源、许可和引用信息；如所用序列另有说明，以该序列的具体条款为准。

## 构建与首次运行

```bash
cmake -S . -B build
cmake --build build --config Release

# 首次使用需要训练并生成本地索引
./build/vpr --build

# 查询示例图片并生成结果网页
./build/vpr --query data/Freiburg/imageCompressedCam0_0005650.png

# 启动网页界面
./open_web.sh
```

## 网页界面

```bash
./open_web.sh
```

网页服务启动后会自动打开浏览器；按 Ctrl+C 停止服务。也可以直接打开 `output/results.html` 查看静态结果，但网页中的训练和查询功能需要通过 `./open_web.sh` 使用。

网页（output/results.html）顶部控制栏：

- **当前聚类中心个数**：直接显示词典实际中心数
- **修改为 [输入框] [重新训练]**：填入想要的个数 → 自动修改 config.yaml → **若该个数已训练过则直接复用缓存（秒级，不重新训练）**，否则才真正训练（--build）→ 用原查询图重新检索 → 完成后自动刷新页面
- **已训练过（直接复用）**：顶部灰色提示列出所有缓存过的 K，选这些数不用等训练
- **查询相似图片**：三种方式选目标图 → 自动执行检索并刷新下方结果区
  - 输入框：从图库下拉列表选择（图库共 N 张），或直接输入图片路径 / 文件名
  - 「选择本机图片」：上传本地任意图片（≤20MB，暂存到 uploads/ 目录）再检索
  - 「查询相似图片」按钮：触发查询；找不到图片时会给出提示

> 提示：修改功能需通过 `./open_web.sh`（HTTP 方式）打开网页才可用；直接双击 results.html（文件方式）只能查看结果，页面会提示改用命令打开。

## 训练缓存（按 K 值复用）

- 每个训练完成的 K 会保存一份模型到 `output/models/k<K>/`（vocabulary.yml + idf.yml + tfidf_vectors.bin）。
- 之后再选同一个 K：**只恢复缓存 + 重新检索，不再跑 K-Means**（秒级完成）。
- 只有没训练过的 K 才需要真正训练；训练成功后自动入库。
- 模型缓存和检索输出仅保存在本地，不提交到 GitHub；首次训练耗时取决于数据量和 K 值。
- 数据目录 `data/` 或特征算法变化时，旧缓存不再匹配，请手动删除 `output/models/` 后重新训练。

## 更改聚类中心个数（K）

只需修改 **一处**：`config/config.yaml` 的 `dictionary_size`，然后重新训练：

```bash
# 1) 编辑 config/config.yaml
#    dictionary_size: 2000   ← 改成你想要的值，例如 3000

# 2) 重新训练（K-Means 重新聚类，K 越大越慢，K=2000 约 15 分钟）
./build/vpr --build

# 3) （可选）重新检索，让 results.html 使用新词典
./build/vpr --query data/Freiburg/imageCompressedCam0_0005650.png
```

- 程序启动时打印 `[INFO] dictionary_size (K) = ...`，可确认当前生效的 K。
- K 全流程自动传递：config.yaml → K-Means → `vocabulary.yml`（K×128 聚类中心）→ IDF → TF-IDF 向量维度，无需改任何代码。
- ⚠️ 必须重新执行 `--build` 才会真正重建词典；词典与底库必须来自同一次训练，混用会导致检索全部得 0 分。

## 流程（与设计文档对应）

1. 训练库跑 K-Means → 固定词典 `output/vocabulary.yml`（K×128，K 由 config.yaml 决定）
2. 用固定词典遍历待检索图像库 → 每张图的词频直方图（TF）
3. 统计每个单词出现在多少张图里 → IDF（`output/idf.yml`）；词典或图像库任一变化都要重算 TF 与 IDF
4. TF × IDF → 每张图的 TF-IDF 向量（检索底库）`output/tfidf_vectors.bin`
5. 查询图 → SIFT → 词典映射 → TF → ×IDF → 与底库算余弦距离 → top-K → `output/results.html`

## 输出产物

| 文件 | 内容 |
| --- | --- |
| output/features.bin | 全部图像的 128 维 SIFT 特征（二进制） |
| output/vocabulary.yml | 视觉词典（K 个聚类中心） |
| output/idf.yml | IDF 权重 |
| output/tfidf_vectors.bin | TF-IDF 检索底库（向量 + 图像路径） |
| output/results.html | 检索结果可视化网页 |

## 实现要点（与理论对应）

- **RootSIFT**：描述子 L1 归一化后逐元素开方，欧氏距离近似 Hellinger 距离
- **K-Means**：K-Means++ 初始化（`KMEANS_PP_CENTERS`）+ 3 次运行取 Inertia 最小 + 固定随机种子 → 词典可复现
- **编码**：FLANN 单树 KD-Tree + `checks=-1` 精确搜索；软分配 top-3（环境变量 `ASSIGN_K=1` 即经典硬分配）；直方图 L2 归一化
- **确定性**：精确搜索与建树随机性无关 → 同一张图任何时刻编码结果一致（自检索余弦得分恰为 1.0）
- **TF-IDF**：IDF = log(N/(1+nᵢ))（+1 平滑），只在待检索图像库上计算
- **检索**：余弦距离 = L2 归一化向量的点积；`nth_element` 部分排序取 top-K，免全量排序

## 目录结构

- `config/config.yaml`：字典大小、检索返回条数 top_k、FLANN 参数等
- `include/` `src/`：各模块接口与实现（FeatureExtractor / VocabularyBuilder / HistogramEncoder / Indexer / Searcher / utils）
