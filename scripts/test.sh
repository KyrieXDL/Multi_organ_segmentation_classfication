#!/bin/bash

data_dir=$1

cd model
rm -rf tmp_data
cd ..
rm -rf result

python my_code/main_segment.py \
--data_dir $data_dir \
--phase 'test' \
--batch_size 1 \
--device 'cuda' \
--accumulate_steps 1 \
--save_name 'base_segment' \
--test_transform 'trans5' \
--pretrain './model/pretrain_model/unet.pth' \
--output_dir './model/tmp_data'

python my_code/postprocess.py

python3 my_code/main_classify_organ.py \
--data_dir $data_dir \
--phase 'test' \
--batch_size 1 \
--device 'cuda' \
--accumulate_steps 1 \
--save_name 'base_classify_organ_aug' \
--task 'classify' \
--test_transform 'trans_organ_mask' \
--loss_func 'asl_loss' \
--class_num 4 \
--task_type 'multi' \
--share_weight \
--resume \
--use_swa \
--pretrain './model/pretrain_model/unet.pth' \
--output_dir './best_model'

cd ./result
zip submit.zip ./* -r
cd ..
mv ./result/submit.zip ./
