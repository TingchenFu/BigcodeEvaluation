RUN_DIR="$PWD"
task=HumanEval
model=Llama-2-7b-hf

accelerate launch  main.py \
  --model ${model} \
  --tasks ${task} \
  --max_length_generation 512 \
  --temperature 1.0 \
  --do_sample False \
  --batch_size 16 \
  --precision  fp16 \
  --generation_only 
  --save_generations  \
  --save_generations_path ${RUN_DIR}/dump/${task}/${model}.jsonl  \




  #  --limit <NUMBER_PROBLEMS> \
  #   --allow_code_execution \