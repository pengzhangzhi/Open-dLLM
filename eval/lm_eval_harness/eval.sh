
# tasks="gsm8k_cot mbpp minerva_math bbh"
tasks="mbpp"
export CUDA_VISIBLE_DEVICES=4,5,6,7  # Configure available GPUs
num_processes=4  # Auto-detect number of GPUs
nshots="0"
lengths="128"
steps="300"
temperatures="0.8"

# model=fredzzp/open-dcoder-3B-repr-align-10-steps-10k
model=fredzzp/open-dcoder-0.5B
# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra STEPS_ARRAY <<< "$steps"
read -ra TEMP_ARRAY <<< "$temperatures"

export HF_ALLOW_CODE_EVAL=1
# accelerate launch --main_process_port 29510 eval.py --model custom_coder \
#     --model_args pretrained=${model},max_new_tokens=512,diffusion_steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true \
#     --tasks humaneval \
#     --num_fewshot 0 \
#     --batch_size 1 \
#     --output_path evals_results/humaneval-ns0 \
#     --log_samples \
#     --confirm_run_unsafe_code 
# ## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}

# Iterate through the arrays
for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_results_test_aug26/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path; GPUs: $num_processes"
    accelerate launch --num_processes $num_processes eval.py --model custom_coder \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},steps=${STEPS_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=0.95,alg=p2 \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 20 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code
done