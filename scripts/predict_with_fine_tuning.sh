ZIP=( "dstc8_metalwoz_heldout.zip" "dstc8_metalwoz_heldout.zip" "dstc8_multiwoz2.0.zip" )
SPEC=( "test_spec_metalwoz_heldout_pure_task.jsonl" "test_spec_metalwoz_heldout_cross_task.jsonl" "test_spec_multiwoz2.0.jsonl" )
OUT_DIR=( "Sep23_gpt2_fine_tuned_metalwoz_pure_task" "Sep23_gpt2_fine_tuned_metalwoz_cross_task" "Sep23_gpt2_fine_tuned_multiwoz" )

for ((i=0;i<${#ZIP[@]};++i)); do
  zip_i="${ZIP[i]}"
  spec_i="${SPEC[i]}"
  out_dir_i="${OUT_DIR[i]}"

  python predict.py \
    data/dstc8-fast-adaptation-evaluation/$zip_i \
    data/dstc8-fast-adaptation-evaluation/$spec_i \
    $out_dir_i \
    --fine_tune \
    --model_checkpoint runs/Sep23 \
    --train_batch_size 32 \
    --valid_batch_size 32 \
    --max_history 2 \
    --train_batches_per_epoch 32 \
    --fp16 O1
done
