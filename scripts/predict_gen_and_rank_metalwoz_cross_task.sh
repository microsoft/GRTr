#!/bin/bash
MODEL=$1
EVAL_DATA_DIR="./dstc8_test_data"

ZIPFILE="${EVAL_DATA_DIR}/dstc8_metalwoz_heldout.zip"
TESTSPEC="${EVAL_DATA_DIR}/test_spec_metalwoz_heldout_cross_task.jsonl"
OUTPUT_DIR="${MODEL}_metalwoz_cross_gen_rank_finetuned"

ALL_DOMAINS=( "BOOKING_FLIGHT" "HOTEL_RESERVE" "TOURISM" "VACATION_IDEAS" )
MAX_HISTORY=3
FP16='' # --fp16 O1
BATCH_SIZE="16"

# prediction done with batch size 16 on 4 P100s with `--fp O1`

for domain in "${ALL_DOMAINS[@]}"; do
  python ./scripts/predict_generate_and_rank $ZIPFILE $TESTSPEC $OUTPUT_DIR $MODEL $OUTPUT_DIR/.model --fine_tune --train_batch_size $BATCH_SIZE --valid_batch_size $BATCH_SIZE --max_history $MAX_HISTORY $FP16 --domain_names $domain
done
