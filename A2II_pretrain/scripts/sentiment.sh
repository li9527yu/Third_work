export CUDA_VISIBLE_DEVICES=7
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 

python  train.py \
  --task_type sentiment \
  --dataset twitter2017 \
  --data_dir /data/lzy1211/code/A2II/instructBLIP/data \
  --output_dir /data/lzy1211/code/pretrain/outputs/sentiment/1116_new_pretrain \
  --BATCH_SIZE 16 \
  --EPOCHS 30 \
  --LEARNING_RATE 2e-5 \
  --use_qformer_online \
  --freeze_vision \
  --pretrained_relation_path /data/lzy1211/code/pretrain/outputs/relation_pretrain_1116/relation_pretrained.bin
