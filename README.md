# DuplicateBugFinder
Duplicate Bug Reports Detection\
Project of IS706-Software Mining and Analysis\
Baseline: [Towards Accurate Duplicate Bug Retrieval Using Deep Learning Techniques](http://ieeexplore.ieee.org/document/8094414/)

## Members
TRUONG Quoc Tuan\
PHAM Hong Quang\
LE Trung Hoang

## How to run
Get data sets from here: http://alazar.people.ysu.edu/msr14data/

### Data preprocessing
```
python data_prepare.py
```
```
optional arguments:
  -h, --help            show this help message and exit
  -d, --data            DATA
                        Path to data folder
  -r, --ratio           SPLIT_RATIO
                        Split ratio of training data (default: 0.8)
  -wv, --word_vocab     WORD_VOCAB_SIZE
                        Word vocabulary size (default: 20,000)
  -cv, --char_vocab     CHAR_VOCAB_SIZE
                        Character vocabulary size (default: 100)
```

### Training and Evaluating
```
python main.py
```
```
optional arguments:
  -h, --help            show this help message and exit
  -d, --data            DATA
                        Path to data folder
  -k, --top_k           TOP_K
                        Number of top candidates for Recall@k evaluation (default: 25)
  -e, --epochs          EPOCHS
                        Number of training epochs (default: 30)
  -b, --baseline        BASELINE
                        Run with baseline model (default: False)
  -nw, --n_words        NUM_WORDS
                        Number of words in vocabulary (default: 20,000)
  -nc, --n_chars        NUM_CHARS
                        Number of characters in vocabulary (default: 100)
  -wd, --word_dim       WORD_DIM
                        Dimension of word embeddings (default: 300)
  -cd, --char_dim       CHAR_DIM
                        Dimension of character embeddings (default: 50)
  -nf, --n_filters      NUM_CNN_FILTERS
                        Number of filters for CNN (default: 64)
  -np, --n_prop         NUM_PROPERTIES
                        Number of properties of the bug report (depending on data sets)
  -bs, --batch_size     BATCH_SIZE
                        Batch size of training (default: 64)
  -nn, --n_neg          NUM_NEGATIVE_SAMPLES
                        Number of negative samples (default: 1)
  -lr, --learning_rate  Learning rate (default: 1e-3)
```