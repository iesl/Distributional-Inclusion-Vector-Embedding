# Distributional Inclusion Vector Embedding (DIVE)
## Requirements
Python 2.7

Tensorflow (To train)

IPython notebook (To visualize)

You might need to install some other python packages if you find code needs them but you haven't installed it. 

## Pre-trained embedding and visualization
The pretrained embeddings are stored in `./model/word-emb.json` (without POS) and in `./model/word-emb_POS.json` (with POS). 

You can visualize it by running ipython notebook code `embedding_visualization.ipynb`

We also visualize the DIVE embedding and corresponding contexts of top 1% words in material science papers: https://bl.ocks.org/chsu5358/raw/f08d4755b0f04e113c139a72a977df5c/.

## To train DIVE:

### Input data
The code assumes the tokenization has already been done and each token in input corpus should be splitted by space. To reproduce our results, please use [WaCkypedia_EN](http://wacky.sslmit.unibo.it/doku.php?id=start). The input corpus could either be raw text (e.g., `anarchism be a political philosophy...`) or append the POS after each token (e.g., `anarchism|NN be|VB a|DT political|JJ philosophy|NN...`). If your input data are compressed by gz and with extension `.gz`, you can directly run the code without decompressing it.

### Running the training code
Let `CORPUS_LOC` in `train.sh` point to your corpus file. Run `./train.sh`. If your input corpus contains POS, remember to add `-p "$delimiter"` when the `train.sh` script runs `python dataset_preproc.py`.

When the script runs `DIVE_train.py`, you might see the loss becomes inf after several epochs. There might not be anything wrong. Backpropagtion could still work properly when the loss function become inf.

Notice that the main code is written by tensorflow, so you can try different objective function easily. However, the current implementation is not optimized for speed and only can take the input where all sentences have the same length. If you would like to share a implementation which is more flexible, faster, or more scalable, please send an email to the first author of the paper.



## Evaluation of unsupervised hypernym detection:

### Evaluation Datasets preparation
* Downlaod BLESS, EVALution, LenciBenotto, and Weeds from [here](https://github.com/vered1986/UnsupervisedHypernymy). Concatenate `*.val` and `*.test` together to become `*.all`, and put them into `./eval_datasets`.
* Downlaod levy2014, kotlerman2010, turney2014, and baroni2012 from [here](https://github.com/stephenroller/emnlp2016/). Run `awk -F $'\t' 'BEGIN {OFS = FS} {if($3 == "True") {print $1,$2,$3,"hyper"} else {print $1,$2,$3,"random"}}' $INPUT_DIR/$INPUT_FILE_NAME > ./eval_datasets/$INPUT_FILE_NAME` where `$INPUT_DIR/$INPUT_FILE_NAME` are the locations of these dataset files. Remove the header line (i.e. word1   word2   label   random) on each file.
* Downlaod HypeNet from [here](https://github.com/vered1986/HypeNET). Uncompress `datasets.rar`, and run `mkdir -p ./eval_datasets/HypeNET/dataset_rnd; tr ' ' ',' < $INPUT_DIR/dataset_rnd/test.tsv | awk -F $'\t' 'BEGIN {OFS = FS} {if($3 == "True") {print $1,$2,$3,"hyper"} else {print $1,$2,$3,"random"}}' > ./eval_datasets/HypeNET/dataset_rnd/test.tsv` where `$INPUT_DIR` is the locations of the dataset.
* Downlaod HyperLex from [here](http://people.ds.cam.ac.uk/iv250/hyperlex) and put `hyperlex-all.txt` it into `./eval_datasets`.
* For the wordnet dataset from [order embedding](https://github.com/ivendrov/order-embeddings-wordnet), the preprocessing step is more complicated, so we directly put the dataset `wordnet_test.txt` into `./eval_datasets`.


### Running the evaluation code
Train skip gram model using gensim for evaluation. You can see an example in `train_word2vec.py`.
In `eval_emb.sh`, set `$POS_WORD2VEC` (with POS) and `$WORD2VEC` (without POS) to be the path of the skip gram models and modify the path in `$POS_INPUT_FILE` and `$INPUT_FILE` to be the path of your output embeddings.

### Results meaning
The output files will look like following.
```
...
summation, 0.73103199394, 0.749850641585, 0.729241877256, 0.851263537906, 0.855368882396
summation_word2vec, 0.866852640254, 0.777838259814, 0.789891696751
...
```
It means dS has AP@all = 73.1%, F1 = 75.0%, and accuracy = 72.9%. When predicting which one is more general in the hypernymy pairs, the accuracy is 85.1% (on all pairs), and 85.5% (not including the case when the scores tie).

## Citation
If you use the code, please cite our [paper](https://arxiv.org/abs/1710.00880).
```
Haw-Shiuan Chang, ZiYun Wang, Luke Vilnis, and Andrew McCallum, 
Distributional Inclusion Vector Embedding for Unsupervised Hypernymy Detection, 
Human Language Technology Conference of the North American Chapter of the 
Association of Computational Linguistics (HLT/NAACL), 2018
```
