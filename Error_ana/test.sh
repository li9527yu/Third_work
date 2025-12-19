export CUDA_VISIBLE_DEVICES=2
#  -m debugpy --connect 127.0.0.1:2233 "twitter2015" 

# 错误分析，针对使用图文情感线索，但没有使用相关性控制
for dataset in  "twitter2015"
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python    ana.py \
    --dataset ${dataset}\
    --data_dir  /data/lzy1211/code/pretrain/data/emotion_gemini \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir //data/lzy1211/code/pretrain/outputs/mutltiTask_textSenti/${dataset}/${seed} \
    --seed ${seed} \
    --LEARNING_RATE 5e-5 \
    --weight 0.5
    done
done

# for dataset in "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  -m debugpy --connect 127.0.0.1:2233  ana.py \
#     --dataset ${dataset}\
#     --data_dir  /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parese_img_clues \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-ThreeRel-Gemini/Img_Cules/${dataset}/${seed} \
#     --seed ${seed} \
#     --LEARNING_RATE 5e-5
#     done
# done
