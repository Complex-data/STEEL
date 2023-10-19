#!/bin/bash
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

#source "${main_dir}/configs/model_webglm.sh"

DATA_PATH="data/LIAR_QA_v1.jsonl"

#--webglm_ckpt_path $GENERATOR_CHECKPOINT_PATH \
run_cmd="python ${main_dir}/evaluate.py \
       --webglm_ckpt_path $GENERATOR_CHECKPOINT_PATH \
       --task LIAR_QA_v3_eq \
       --evaluate_task_data_path $DATA_PATH
       --searcher bing
       --retriever_ckpt_path ${main_dir}/download/retriever-pretrained-checkpoint"
#
eval ${run_cmd}