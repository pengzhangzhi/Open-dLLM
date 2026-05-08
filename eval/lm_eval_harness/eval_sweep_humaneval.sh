# read the num of gpus from nvidia-smi
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Calculate total number of combinations
TOTAL_STEPS=$(( (2000-100)/200 + 1 ))
TOTAL_TEMPS=$(( (1.0-0.01)/0.1 + 1 | bc ))
TOTAL_JOBS=$(( TOTAL_STEPS * TOTAL_TEMPS ))

# Counter for round-robin GPU assignment
job_counter=0

for steps in {100..2000..200}; do
  for temp in $(seq 0.01 0.1 1.0); do
    # Calculate which GPU to use (round-robin)
    gpu_id=$((job_counter % NUM_GPUS))
    
    # Run command on specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id accelerate launch --main_process_port $((29510 + gpu_id)) eval/lm_harness/eval.py \
      --model custom_coder \
      --model_args pretrained=fredzzp/open-dcoder-0.5B,max_new_tokens=200,steps=$steps,temperature=$temp,top_p=0.95,add_bos_token=true,escape_until=true \
      --tasks humaneval \
      --num_fewshot 0 \
      --batch_size 16 \
      --output_path evals_results/humaneval-ns0-steps${steps}-temp${temp} \
      --log_samples \
      --confirm_run_unsafe_code &
      
    job_counter=$((job_counter + 1))
    
    # If we've launched jobs on all GPUs, wait for them to complete
    if [ $((job_counter % NUM_GPUS)) -eq 0 ]; then
      wait
    fi
  done
done

# Wait for any remaining jobs
wait
