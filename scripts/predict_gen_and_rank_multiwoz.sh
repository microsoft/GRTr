#!/bin/bash
MODEL=$1
EVAL_DATA_DIR="./dstc8_test_data"

ZIPFILE="${EVAL_DATA_DIR}/dstc8_multiwoz2.0.zip"
TESTSPEC="${EVAL_DATA_DIR}/test_spec_multiwoz2.0.jsonl"
OUTPUT_DIR="${MODEL}_multiwoz_gen_rank_finetuned"

ALL_DOMAINS=( "attraction" "hospital" "hotel" "police" "restaurant" "taxi" "train" )
MAX_HISTORY=3
FP16='' # --fp16 01
BATCH_SIZE="16"

# prediction done with batch size 16 on 4 P100s with `--fp 01`

for domain in "${ALL_DOMAINS[@]}"; do
  python ./scripts/predict_generate_and_rank $ZIPFILE $TESTSPEC $OUTPUT_DIR $MODEL $OUTPUT_DIR/.model --fine_tune --train_batch_size $BATCH_SIZE --valid_batch_size $BATCH_SIZE --max_history $MAX_HISTORY $FP16 --domain_names $domain
done
