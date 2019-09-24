# Language Model Playground
[\[中文文件\]](#中文文件) [\[English\]](#English-Document)

# 中文文件
使用 PyTorch 實作 Language Model。

## 環境
1. Python 版本: 3.6+
2. CUDA 版本: 9.0+

## 安裝
1. 從 github 複製專案。
```
git clone https://github.com/ProFatXuanAll/language-model-playground.git
```
2. 移動到資料夾中。
```
cd language-model-playground
```
3. 安裝相依套件。
```
pip install -r requirements.txt
```
4. 新增資料夾 `data` 。
```
mkdir data
```

## 範例
1. 從 kaggle 上下載[資料](https://www.kaggle.com/ceshine/yet-another-chinese-news-dataset/downloads/yet-another-chinese-news-dataset.zip/8)，並解壓縮 `zip` 檔後把資料放到 `data/news_collection.csv`。
```
unzip yet-another-chinese-news-dataset.zip && \
chmod 666 news_collection.csv && \
mv news_collection.csv data/news_collection.csv
```
2. 訓練範例模型。
```
python example_train.py
```
3. 生成範例句子。
```
python example_generate.py
```
4. 試著修改檔案 `example_train.py` 中的超參數或是更換模型，接著重新訓練。替換檔案 `example_generate.py` 中第 26 行的句子，模型會自己把剩下的句子生成。
```
python example_train.py && \
python example_generate.py
```

# English Document
Language Model implemented with PyTorch.

## Environment
1. Python version: 3.6+
2. CUDA version: 9.0+

## Install
1. Clone the project.
```
git clone https://github.com/ProFatXuanAll/language-model-playground.git
```
2. Move to project directory
```
cd language-model-playground
```
3. Install dependencies.
```
pip install -r requirements.txt
```
4. Create `data` folder.
```
mkdir data
```

## Example
1. Download [data](https://www.kaggle.com/ceshine/yet-another-chinese-news-dataset/downloads/yet-another-chinese-news-dataset.zip/8) from kaggle, extract from `zip` and put it at `data/news_collection.csv`.
```
unzip yet-another-chinese-news-dataset.zip && \
chmod 666 news_collection.csv && \
mv news_collection.csv data/news_collection.csv
```
2. Train example model.
```
python example_train.py
```
3. Generate example sentences.
```
python example_generate.py
```
4. Try different hyperparameters or change to different model by modifing `example_train.py`, then train model again. Replace `example_generate.py` line 26 with your own character sequence, then model will generate the rest sentence for you.
```
python example_train.py && \
python example_generate.py
```
