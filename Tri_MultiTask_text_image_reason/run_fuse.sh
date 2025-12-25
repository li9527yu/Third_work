export CUDA_VISIBLE_DEVICES=1
# debug: -m debugpy --connect 127.0.0.1:2233
# 12-17: 多任务: 文本情感预测，图像情感预测多任务113116

# 以max_f1为验证集指标
# 12-23: 图像情感推理 分支修改输入和标签
for dataset in "twitter2015"   
do
    for seed in "24"
    do
    echo ${dataset}
    echo ${seed}
    python -m debugpy --connect 127.0.0.1:2233 train.py \
    --dataset ${dataset} \
    --data_dir /data/lzy1211/code/pretrain/data/emotion_gemini \
    --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
    --output_dir /data/lzy1211/code/pretrain/outputs/mutltiTask_text_image_resaon/${dataset}/${seed} \
    --BATCH_SIZE 16 \
    --seed ${seed} \
    --EPOCHS 20 \
    --LEARNING_RATE 5e-5 \
    --weight 0.5
    done
done


# 1127:在不考虑相关性的情况下直接使用情感线索，同时加入主观性分类的多任务
# for dataset in "twitter2015"   
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/ \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/pretrain/outputs/mtl_subject/${dataset}/${seed} \
#     --BATCH_SIZE 8 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 2e-5 \
#     --weight 0.5
#     done
# done


# sota结果复现
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/emotion_clue_reproduce/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 25 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done

# 实验结果调参
# for ep in "15" "20"
# do
#     for bs in "8" "16" "32"
#     do
#         for lr in "1e-5" "2e-5" "5e-5"
#         do
#             echo ${ep} ${bs} ${lr}
#             python train.py \
#             --dataset twitter2015 \
#             --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#             --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#             --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/A_tiaocan/twitter2015/${ep}_${bs}_${lr} \
#             --BATCH_SIZE ${bs} \
#             --seed 24 \
#             --EPOCHS ${ep} \
#             --LEARNING_RATE ${lr} \
#             --weight 0.5
#         done
#     done
# done


# 测试Gemini文本情感线索
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parese_text_clues/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/late_fuse_GeminiTextClues/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done


# Prompt中文本图像情感线索混用
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/late_fuse_hybridClues/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done


# 在融合模态中额外显式加入相关性标签
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/late_fuse_explicitRelation/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done



#  在晚期融合中不使用相关性来控制情感线索的使用，直接暴力全部的使用情感线索 （Gemini的文本情感线索）
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/gemini_emotion_clues/parese_text_clues/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/late_fuse_emotion_clues_GeminiText/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5 \
#     --weight 0.5
#     done
# done

# 0909:测试超参数：权重
# for weight in 0.2 0.3 
# do
#     for seed in "24"
#     do
#         echo ${seed}
#         python    train.py \
#         --dataset twitter2015 \
#         --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#         --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#         --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_weight_fuse/twitter2015_weight${weight}/${seed} \
#         --BATCH_SIZE 16 \
#         --seed ${seed} \
#         --EPOCHS 20 \
#         --LEARNING_RATE 5e-5 \
#         --weight ${weight}
#     done
# done

# twitter2017
# for weight in 0.7 0.8 
# do
#     for seed in "24"
#     do
#         echo ${seed}
#         python    train.py \
#         --dataset twitter2017 \
#         --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#         --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#         --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_weight_fuse/twitter2017_weight${weight}/${seed} \
#         --BATCH_SIZE 16 \
#         --seed ${seed} \
#         --EPOCHS 20 \
#         --LEARNING_RATE 5e-5 \
#         --weight ${weight}
#     done
# done


# debug: 
# # 8-19: 调整融合方式  "twitter2017"  
# for dataset in "twitter2015"  "twitter2017"  
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python    train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/relation_weight_fuse/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 25 \
#     --LEARNING_RATE 5e-5
#     done
# done

# # 8-17:调整晚期融合结果
# for dataset in  "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/Late_fuse_8_17_v2/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 20 \
#     --LEARNING_RATE 5e-5
#     done
# done




# 8-15：考虑晚期融合的结果
# for dataset in  "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset} \
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/Late_fuse/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 15 \
#     --LEARNING_RATE 5e-5
#     done
# done


# # 新的双路融合实验：调整后的情感线索支路+图文理解组合输入
# for dataset in  "twitter2015" "twitter2017"
# do
#     for seed in "24"
#     do
#     echo ${dataset}
#     echo ${seed}
#     python  train.py \
#     --dataset ${dataset}\
#     --data_dir /data/lzy1211/code/A2II/instructBLIP/CoT_data_process/emotion_clue/  \
#     --img_feat_dir /data/lzy1211/code/A2II/instructBLIP/img_data \
#     --output_dir /data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-ThreeInfo/two_infuse/${dataset}/${seed} \
#     --BATCH_SIZE 16 \
#     --seed ${seed} \
#     --EPOCHS 15 \
#     --LEARNING_RATE 5e-5
#     done
# done
