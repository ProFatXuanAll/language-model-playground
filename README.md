# Language Model Playground
Language Model implemented with PyTorch.

# INSTALL

1. Clone the project.
```
git clone https://github.com/ProFatXuanAll/char-RNN.git
```

2. Install dependencies.
```
pip install -r requirements.txt
```

3. Create `data` folder.
```
mkdir data
```

# TRY EXAMPLE
1. Download [data](https://www.kaggle.com/ceshine/yet-another-chinese-news-dataset/downloads/yet-another-chinese-news-dataset.zip/8) from kaggle, extract from `zip` and put it at `data/news_collection.csv`.

2. Train model.
```
python example_train.py
```

3. Replace `example_generate.py` line 26 with your character sequence, then model will generate the rest sentence for you.
```
python example.generate.py
```