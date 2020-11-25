# Language Model Playground

- [中文文件](#中文文件)
- [English](#English-Document)

## 中文文件

使用 PyTorch 實作語言模型（Language Model）。

### 環境

1. Python 版本: 3.6+

2. CUDA 版本: 10.0+

### 安裝

1. 從 github 複製專案。

```sh
git clone https://github.com/ProFatXuanAll/language-model-playground.git
```

2. 移動到資料夾中。

```sh
cd language-model-playground
```

3. 安裝相依套件。

```sh
pipenv install
```

4. 啟動虛擬環境。

```sh
pipenv shell
```

### 文件

1. 安裝文件編譯相依套件

```sh
pipenv install --dev
```

2. 編譯文件

```sh
pipenv run doc
```

3. 在瀏覽器中開啟 `doc/build/index.html`

```sh
xdg-open doc/build/index.html
```

### 測試

1. 安裝文件編譯相依套件

```sh
pipenv install --dev
```

2. 執行測試

```sh
isort .
autopep8 -r -i -a -a -a lmp
autopep8 -r -i -a -a -a test
coverage -m pytest
```

### 訓練

1. 訓練範例中文語言模型。

```sh
# 使用 `news_collection_title` 資料集訓練中文語言模型並保存為 experiment 1
python run_train.py --experiment 1 --batch_size 32 --checkpoint -1 --checkpoint_step 500 --d_emb 100 --d_hid 300 --dataset news_collection_title --dropout 0.1 --epoch 10 --is_uncased --learning_rate 1e-4 --max_norm 1.0 --max_seq_len 60 --min_count 1 --model_class lstm --num_linear_layers 1 --num_rnn_layers 1 --optimizer_class adam --seed 42 --tokenizer_class char_dict
```

2. 訓練範例英文語言模型。

```sh
# 使用 `wiki_train_tokens` 資料集訓練英文語言模型並保存為 experiment 2
python run_train.py --experiment 2 --batch_size 32 --checkpoint -1 --checkpoint_step 500 --d_emb 100 --d_hid 300 --dataset wiki_train_tokens --dropout 0.1 --epoch 10 --is_uncased --learning_rate 1e-4 --max_norm 1.0 --max_seq_len 60 --min_count 1 --model_class lstm --num_linear_layers 1 --num_rnn_layers 1 --optimizer_class adam --seed 42 --tokenizer_class whitespace_dict
```

3. 使用 `tensorboard` 觀察語言模型誤差表現。

```sh
# 在 Windows 上路徑請用 `.\data\log`
tensorboard --logdir ./data/log
```

### 評估

1. 評估中文語言模型在 `news_collection_title` 資料集上的 perplexity 表現。

```sh
# 使用第 500 步存檔點進行評估
python run_perplexity_evaluation.py --experiment 1 --checkpoint 500 --dataset news_collection_title
```

2. 評估英文語言模型在 `wiki_test_tokens` 資料集上的 perplexity 表現。

```sh
# 使用第 500 步存檔點進行評估
python run_perplexity_evaluation.py --experiment 2 --checkpoint 500 --dataset wiki_test_tokens
```

3. 評估英文語言模型在 `word_test_v1` 文字類比測試。

```sh
# 使用第 500 步的存檔點進行評估
python run_analogy_evaluation.py --experiment 2 --checkpoint 500 --dataset word_test_v1
```

### 驗證

1. 指定中文語言模型存檔點生成範例句子。

```sh
# 使用第 500 步的存檔點進行句子生成
python run_generate.py --experiment 1 --checkpoint 500 --begin_of_sequence "今天" --beam_width 4 --max_seq_len 60
```

2. 指定英文語言模型存檔點生成範例句子。

```sh
# 使用第 500 步的存檔點進行句子生成
python run_generate.py --experiment 2 --checkpoint 500 --begin_of_sequence "today is" --beam_width 4 --max_seq_len 60
```

3. 指定中文語言模型存檔點生成類比文字。

```sh
# 使用第 500 步的存檔點進行類比文字生成
python run_analogy_inference.py --experiment 1 --checkpoint 500 --word_a "早" --word_b "晚" --word_c "日"
```

4. 指定英文語言模型存檔點生成類比文字。

```sh
# 使用第 500 步的存檔點進行類比文字生成
python run_analogy_inference.py --experiment 2 --checkpoint 500 --word_a "Taiwan" --word_b "Taipei" --word_c "Japan"
```

5. 試著使用不同的超參數或更換模型並使用 `run_train.py` 重新訓練。接著使用 `run_generate.py` 給予相同 `begin_of_sequence` 進行生成並比較生成結果之不同。

### 開發

1. 請參考 [Google python style guide](https://google.github.io/styleguide/pyguide.html) 撰寫程式碼並使程式碼符合其風格。

2. 請參考 [`typing`](https://docs.python.org/3/library/typing.html) 為每個 `function` 與 `method` 加上型態註記。

3. 請為每個 `class`, `function` 與 `method` 補上 `docstring`。

4. 請執行 `pylint your_code.py` 自動驗證你的程式是否符合 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 的規範。

5. 請執行 `autopep8 -i -a -a your_code.py` 自動修改你的程式使其符合 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 的規範。

6. 請執行 `mypy your_code.py` 自動驗證程式碼的型別正確性。

7. 請執行 `python -m unittest` 確認程式碼是否通過單元測試。

8. 請撰寫單元測試程式碼讓程式碼容易維護。

## English Document

Language Model implemented with PyTorch.

### Environment

1. Python version: 3.6+

2. CUDA version: 10.0+

### Install

1. Clone the project.

```sh
git clone https://github.com/ProFatXuanAll/language-model-playground.git
```

2. Move to project directory

```sh
cd language-model-playground
```

3. Install and Launch Virtual Environment.

```sh
# Use python built-in virtual environment.
# The following script need `python3-dev` `python3-venv`.
# For example on Ubuntu use `apt-get install python3-dev python3-venv`.

# If you are using `conda` do `conda create --name venv python=3.6` instead.
python3 -m venv venv # Install virtual environment.

# If you are using `conda` do `activate venv` instead.
source venv/bin/active # Launch virtual environment.
```

4. Install Dependencies.

```sh
pip install -r requirements.txt
```

### Download dataset

1. Create folder `data`.

```sh
mkdir data
```

2. Download the Chinese dataset: download [news_colleciton.csv](https://www.kaggle.com/ceshine/yet-another-chinese-news-dataset) from kaggle, unzip the `zip` file and put it in `data/news_collection.csv`.

```sh
unzip yet-another-chinese-news-dataset.zip && chmod 666 news_collection.csv && mv news_collection.csv data/news_collection.csv
```

3. Download the English dataset: Download from The WikiText Long Term Dependency Language Modeling Dataset [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/ ), unzip the `zip` file and put the data in `data/wiki.train.tokens`, `data/wiki.valid.tokens`, `data/wiki.test.tokens`.

4. Download the word analogy dataset: download `word-test.v1.txt` from the following [website](http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt) and put it in `data/word-test.v1.txt`.

```sh
wget -c http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt && chmod 666 word-test.v1.txt && mv word-test.v1.txt data/word-test.v1.txt
```

### Train model

1. Train example Chinese language model.

```sh
# Use `news_collection_title` dataset to train Chinese language model and save as experiment 1.
python run_train.py --experiment 1 --batch_size 32 --checkpoint -1 --checkpoint_step 500 --d_emb 100 --d_hid 300 --dataset news_collection_title --dropout 0.1 --epoch 10 --is_uncased --learning_rate 1e-4 --max_norm 1.0 --max_seq_len 60 --min_count 1 --model_class lstm --num_linear_layers 1 --num_rnn_layers 1 --optimizer_class adam --seed 42 --tokenizer_class char_dict
```

2. Train example English language model.

```sh
# Use `wiki_train_tokens` dataset to train English language model and save as experiment 2.
python run_train.py --experiment 2 --batch_size 32 --checkpoint -1 --checkpoint_step 500 --d_emb 100 --d_hid 300 --dataset wiki_train_tokens --dropout 0.1 --epoch 10 --is_uncased --learning_rate 1e-4 --max_norm 1.0 --max_seq_len 60 --min_count 1 --model_class lstm --num_linear_layers 1 --num_rnn_layers 1 --optimizer_class adam --seed 42 --tokenizer_class whitespace_dict
```

3. Use `tensorboard` to observe language model training loss.

```sh
# On Windows use path `.\data\log`
tensorboard --logdir ./data/log
```

### Evaluate

1. Evaluate Chinese language model performance on `news_collection_title` dataset by calculating perplexity.

```sh
# Evaluate on checkpoint 500.
python run_perplexity_evaluation.py --experiment 1 --checkpoint 500 --dataset news_collection_title
```

2. Evaluate English language model performance on `wiki_test_tokens` dataset by calculating perplexity.

```sh
# Evaluate on checkpoint 500.
python run_perplexity_evaluation.py --experiment 2 --checkpoint 500 --dataset wiki_test_tokens
```

3. Evaluate English language model performance on `word_test_v1` dataset by word analogy.

```sh
# Evaluate on checkpoint 500.
python run_analogy_evaluation.py --experiment 2 --checkpoint 500 --dataset word_test_v1
```

### Validation

1. Generate sequences using Chinese language model checkpoints.

```sh
# Using checkpoint 500 to generate sequence.
python run_generate.py --experiment 1 --checkpoint 500 --begin_of_sequence "今天" --beam_width 4 --max_seq_len 60
```

2. Generate sequences using English language model checkpoints.

```sh
# Using checkpoint 500 to generate sequence.
python run_generate.py --experiment 2 --checkpoint 500 --begin_of_sequence "today is" --beam_width 4 --max_seq_len 60
```

3. Generate analog word using Chinese language model checkpoints.

```sh
# Using checkpoint 500 to generate analog word.
python run_analogy_inference.py --experiment 1 --checkpoint 500 --word_a "早" --word_b "晚" --word_c "日"
```

4. Generate analog word using English language model checkpoints.

```sh
# Using checkpoint 500 to generate analog word.
python run_analogy_inference.py --experiment 2 --checkpoint 500 --word_a "Taiwan" --word_b "Taipei" --word_c "Japan"
```

5. Try different hyperparameters or change model, then use `run_train.py` to perform training as above example. Run `run_generate.py` to compare generated results given exactly same `begin_of_sequence`.

### Development

1. Make sure your code conform [Google python style guide](https://google.github.io/styleguide/pyguide.html).

2. Do type annotation for every `function` and `method` (You might need to see [`typing`](https://docs.python.org/3/library/typing.html)).

3. Write `docstring` for every `class`, `function` and `method`.

4. Run `pylint your_code.py` to automatically check your code whether conform to [PEP 8](https://www.python.org/dev/peps/pep-0008/).

5. Run `autopep8 -i -a -a your_code.py` to automatically fix your code and conform to [PEP 8](https://www.python.org/dev/peps/pep-0008/).

6. Run `mypy your_code.py` to check type annotaions.

7. Run `python -m unittest` to perform unit tests.

8. Write unit tests for your code and make them maintainable.
