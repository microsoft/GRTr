#!/bin/bash
MODEL=$1
EVAL_DATA_DIR="./dstc8_test_data"

ZIPFILE="${EVAL_DATA_DIR}/dstc8_multiwoz2.0.zip"
TESTSPEC="${EVAL_DATA_DIR}/test_spec_multiwoz2.0.jsonl"
OUTPUT_DIR="${MODEL}_multiwoz_gen"

ALL_DOMAINS=( "attraction" "hospital" "hotel" "police" "restaurant" "taxi" "train" )
MAX_HISTORY=3
FP16='' # --fp16 01
BATCH_SIZE="16"

for domain in "${ALL_DOMAINS[@]}"; do
  python ./scripts/predict $ZIPFILE $TESTSPEC $OUTPUT_DIR $MODEL $OUTPUT_DIR/.model --train_batch_size $BATCH_SIZE --valid_batch_size $BATCH_SIZE --max_history $MAX_HISTORY $FP16 --domain_names $domain
done
