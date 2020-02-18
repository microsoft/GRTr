#!/bin/bash
MODEL=$1
EVAL_DATA_DIR="./dstc8_test_data"

ZIPFILE="${EVAL_DATA_DIR}/dstc8_metalwoz_heldout.zip"
TESTSPEC="${EVAL_DATA_DIR}/test_spec_metalwoz_heldout_pure_task.jsonl"
OUTPUT_DIR="${MODEL}_metalwoz_pure_gen"

ALL_DOMAINS=( "BOOKING_FLIGHT" "HOTEL_RESERVE" "TOURISM" "VACATION_IDEAS" )
MAX_HISTORY=3
FP16='' # --fp16 O1
BATCH_SIZE="16"

for domain in "${ALL_DOMAINS[@]}"; do
  python ./scripts/predict $ZIPFILE $TESTSPEC $OUTPUT_DIR $MODEL $OUTPUT_DIR/.model --train_batch_size $BATCH_SIZE --valid_batch_size $BATCH_SIZE --max_history $MAX_HISTORY $FP16 --domain_names $domain
done
