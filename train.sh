#!/bin/bash

mkdir -p "./data"
echo "CLEANING DATA..."

PY_PATH="python"

CORPUS_LOC=''
PROCESSED_CORPUS='./data/wackypedia_512k_matrix_wostop'
#PROCESSED_CORPUS='data/wackypedia_POS_matrix_wostop'
STOP_WORD_LIST='./stop_word_list'
MAX_NUM_SENT=512000
SENT_LEN=100

$PY_PATH ./dataset_preproc.py -s $STOP_WORD_LIST -r $SENT_LEN -l $MAX_NUM_SENT -i $CORPUS_LOC -o $PROCESSED_CORPUS
#$PY_PATH ./dataset_preproc.py -p "|" -s $STOP_WORD_LIST -r $SENT_LEN -l $MAX_NUM_SENT -i $CORPUS_LOC -o $PROCESSED_CORPUS #use this line instead when the input corpus contains POS

NUM_SENT=`wc -l $PROCESSED_CORPUS | cut -d ' ' -f 1` #If your dataset after cleaning is compressed, decompress it before counting the number of lines 

echo "APPLY PMI FILTERING..."

#Computing co-occurrence would take quite some time, so we store the intermediate file
INTERMEDIATE_FILE='./data/wiki_512k_wostop_coocur_pmi_w_10_th10'
#INTERMEDIATE_FILE='data/wiki_512k_POS_wostop_coocur_pmi_w_10_th10' 
TRAINING_FILE='./data/wiki_512k_wostop_pmi_w_10_neg30_th10'
#TRAINING_FILE='./data/wiki_512k_POS_wostop_pmi_w_10_neg30_th10'
NUM_NEG=30
WIN_SIZE=10
PMI_FILTER_SIZE=10

$PY_PATH ./prepare_PMI.py -r $SENT_LEN -p $PMI_FILTER_SIZE -w $WIN_SIZE -n $NUM_NEG -l $NUM_SENT -i $PROCESSED_CORPUS -c $INTERMEDIATE_FILE -o $TRAINING_FILE

echo "TRAINING..."

LOGDIR="model/wiki_f512k_window_pmi_wostop_neg30_pw1_e15_th10"
#LOGDIR="model/wiki_f512k_POS_window_pmi_wostop_neg30_pw1_e15_th10"
EMB_SIZE=100
NUM_EPOCH=15

$PY_PATH ./DIVE_train.py --window_size $WIN_SIZE --log_dir $LOGDIR --emb_size $EMB_SIZE --data_shard_cols $SENT_LEN --data_shard_rows $NUM_SENT --training_file $TRAINING_FILE --use_projection  --num_neg_samples $NUM_NEG --max_epoch $NUM_EPOCH

