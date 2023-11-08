RUN_DIR="$PWD"
task=mbpp
model=Llama-2-7b-hf
#model=GPT-2-small

for peft_name in Llama-2-7b-hf_multitask_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster0_fromLlama-2-7b-hf_fp16_all_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster1_fromLlama-2-7b-hf_fp16_all_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster2_fromLlama-2-7b-hf_fp16_all_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster3_fromLlama-2-7b-hf_fp16_all_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster0_fromLlama-2-7b-hf_fp16_inputoutput_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster1_fromLlama-2-7b-hf_fp16_inputoutput_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster2_fromLlama-2-7b-hf_fp16_inputoutput_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster3_fromLlama-2-7b-hf_fp16_inputoutput_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine  Llama-2-7b-hf_multitask_cluster0_fromLlama-2-7b-hf_fp16_instruction_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster1_fromLlama-2-7b-hf_fp16_instruction_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster2_fromLlama-2-7b-hf_fp16_instruction_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine Llama-2-7b-hf_multitask_cluster3_fromLlama-2-7b-hf_fp16_instruction_cluster4_lora_fp16_bs16lr3e-4epoch3decay0.0cosine
do
  accelerate launch  main.py \
    --model ${RUN_DIR}/../../../PLM/${model} \
    --peft_model_path  ${RUN_DIR}/../../dump/${peft_name} \
    --tasks ${task} \
    --max_length_generation 512 \
    --temperature 0.1 \
    --top_p  0.95    \
    --do_sample False \
    --batch_size 1 \
    --precision  fp16 \
    --allow_code_execution \
    --save_generations_path ${RUN_DIR}/dump/${task}/output_${peft_name}.jsonl  \
    --metric_output_path ${RUN_DIR}/dump/${task}/metric_${peft_name}.jsonl  
done


  #  --limit <NUMBER_PROBLEMS> \
  #   --allow_code_execution \