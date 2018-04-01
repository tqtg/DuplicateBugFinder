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
python3 data_prepare.py 
```
```
optional arguments:
  -h, --help            show this help message and exit
  -d, --data            DATA
                        Path to data folder
  -r, --ratio           SPLIT_RATIO
                        Split ratio of training data (default: 0.8)
  -wv, --word_vocab     WORD_VOCAB_SIZE
                        Word vocabulary size (default: 50,000)
  -cv, --char_vocab     CHAR_VOCAB_SIZE
                        Character vocabulary size (default: 100)
```
