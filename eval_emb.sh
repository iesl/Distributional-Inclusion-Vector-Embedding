#!/bin/bash

mkdir -p "./output"

PY_PATH="python"

TEMP_COMMAND_FILE="./temp_commands"

POS_WORD2VEC="./model/gensim_model_lower_POS"
WORD2VEC="./model/gensim_model_lower"

POS_INPUT_FILE="-i ./model/wiki_f512k_POS_window_pmi_wostop_neg30_pw1_e15_th10/word-emb.json"
INPUT_FILE="-i ./model/wiki_f512k_window_pmi_wostop_neg30_pw1_e15_th10/word-emb.json"
RESULT_OUTPUT="./output/result_DIVE"


> $TEMP_COMMAND_FILE


declare -a EVAL_FILE_ARR=("./eval_datasets/BLESS.all"
"./eval_datasets/EVALution.all"
"./eval_datasets/LenciBenotto.all"
"./eval_datasets/Weeds.all")

for EVAL_FILE in "${EVAL_FILE_ARR[@]}"
do
   FILE_NAME=$(basename "$EVAL_FILE")
   echo "$PY_PATH ./test_emb.py -e $EVAL_FILE $POS_INPUT_FILE -w $POS_WORD2VEC -o ${RESULT_OUTPUT}_$FILE_NAME -v -a '|' -u -f 20" >> $TEMP_COMMAND_FILE
done

declare -a EVAL_FILE_ARR=("./eval_datasets/levy2014.tsv"
"./eval_datasets/kotlerman2010.tsv"
"./eval_datasets/turney2014.tsv"
"./eval_datasets/baroni2012.tsv")

for EVAL_FILE in "${EVAL_FILE_ARR[@]}"
do
   FILE_NAME=$(basename "$EVAL_FILE")
   echo "$PY_PATH ./test_emb.py -e $EVAL_FILE $INPUT_FILE -w $WORD2VEC -o ${RESULT_OUTPUT}_$FILE_NAME -v -u -f 20" >> $TEMP_COMMAND_FILE
done

EVAL_FILE="./eval_datasets/HypeNET/dataset_rnd/test.tsv"
FILE_NAME=$(basename "$EVAL_FILE")
echo "$PY_PATH ./test_emb.py -e $EVAL_FILE $INPUT_FILE -w $WORD2VEC -o ${RESULT_OUTPUT}_$FILE_NAME -c as -v -u -f 20" >> $TEMP_COMMAND_FILE

EVAL_FILE="./eval_datasets/wordnet_test.txt"
FILE_NAME=$(basename "$EVAL_FILE")
echo "$PY_PATH ./test_emb.py -e $EVAL_FILE $POS_INPUT_FILE -w $POS_WORD2VEC -o ${RESULT_OUTPUT}_$FILE_NAME -c as -v -a '|' -u -f 20" >> $TEMP_COMMAND_FILE



EVAL_FILE="./eval_datasets/hyperlex-all.txt"
FILE_NAME=$(basename "$EVAL_FILE")
echo "$PY_PATH ./test_by_hyperlex.py -e $EVAL_FILE $POS_INPUT_FILE -w $POS_WORD2VEC -o ${RESULT_OUTPUT}_$FILE_NAME -a '|' -u" >> $TEMP_COMMAND_FILE


cat $TEMP_COMMAND_FILE
bash $TEMP_COMMAND_FILE 
rm $TEMP_COMMAND_FILE



