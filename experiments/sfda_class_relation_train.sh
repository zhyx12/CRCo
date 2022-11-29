#!/usr/bin/env bash
job_id=$1
config_file=$2

project_home='class-relation-learning'

###
shell_folder=$(cd "$(dirname "$0")"; pwd)
echo $shell_folder
echo $HOME
cd $HOME'/PycharmProjects/'${project_home} || exit

trainer_class=sfda_class_relation
validator_class=sfda_class_relation
scripts_path=$HOME'/PycharmProjects/'${project_home}'/experiments/get_visible_card_num.py'
port_scripts_path=$HOME'/PycharmProjects/'${project_home}'/experiments/generate_random_port.py'
GPUS=$(python ${scripts_path})
PORT=$(python ${port_scripts_path})

python_file=./train.py
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
  ${python_file} --job_id ${job_id} --config ${config_file} \
  --trainer ${trainer_class} --validator ${validator_class}
