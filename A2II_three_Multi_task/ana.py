import logging
import torch
import torch.utils.data as Data
from torch.optim import AdamW
import argparse
import os
import numpy as np
from tqdm import tqdm, trange
import random
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from CoT_Model_LateFuse import MyFlanT5

from DataProcessor_emotion_clues import MyDataset
# from DataProcessor_unified import MyDataset

from sklearn.metrics import precision_recall_fscore_support
import wandb
from torch import nn
from tool import warmup_linear,parse_sequences,compute_metrics
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2017', type=str, help=', ')
    parser.add_argument('--data_dir', default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data", type=str)
    parser.add_argument('--img_dir', default='/data/lzy1211/code/twitterImage', type=str)
    parser.add_argument('--img_feat_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--blip_path', default='/data/lzy1211/code/model/instructblip-flan-t5-xl', type=str,
                        help='InstructBLIP 模型路径（含Flan-T5与Q-Former）')
    parser.add_argument('--output_dir', default="/data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-t", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--EPOCHS', default=2, type=int)
    parser.add_argument('--LEARNING_RATE', default=1e-5, type=float)
    parser.add_argument("--weight",
                    default=0.5,
                    type=float,
                    help="text weight")
    parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%  of training.")
    parser.add_argument("--max_seq_len",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--run_name",
                        default="pytorch_model",
                        type=str,
                        help="运行的名称")
    parser.add_argument("--output_model_name",
                        default="pytorch_model",
                        type=str,
                        help="保存权重的名称")
    parser.add_argument("--output_log_name",
                        default="log",
                        type=str,
                        help="日志的名称")  
    opt = parser.parse_args()

    return opt


def move_to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor


def evaluate_single(model, dataset,weight, logger, save_path=None):
    """单样本推理"""
    model.eval()
    device = next(model.parameters()).device

    all_preds, all_labels, meta_records = [], [], []

    for i in tqdm(range(len(dataset)), desc="Single-sample Testing"):
        # 取单条样本

        sample = dataset[i]
        (input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
        aspect,image,labels,senti_label,relation_label,raw_meta) = sample

        # === 张量转设备 ===
        input_ids = move_to_device(input_ids.unsqueeze(0), device)
        input_attention_mask = move_to_device(input_attention_mask.unsqueeze(0), device)
        input_multi_ids = move_to_device(input_multi_ids.unsqueeze(0), device)
        input_multi_attention_mask = move_to_device(input_multi_attention_mask.unsqueeze(0), device)
        
        labels = move_to_device(labels.unsqueeze(0), device)
        senti_label = move_to_device(senti_label, device)
        relation_label = move_to_device(relation_label, device)

        with torch.no_grad():
            outputs= model.generate(input_ids=input_ids,
                            input_multi_ids=input_multi_ids,
                            attention_mask=input_attention_mask,  
                            input_multi_attention_mask=input_multi_attention_mask,            
                            text=aspect,
                            images=image,
                            relation=relation_label,
                            weight=weight
                        )

        # === 解析预测结果 ===
        preds = parse_sequences(outputs)
        all_preds.append(preds[0])
        all_labels.append(senti_label)

        # === 记录样本信息 ===
        sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
        meta_records.append({
            "text": raw_meta["text"],
            "aspect": raw_meta["aspect"],
            "image": raw_meta["image"],
            "gold": sentiment_map[str(senti_label)],
            "pred": sentiment_map[str(preds[0])],
            "relation": raw_meta["relation"],
            "text_emo":raw_meta["text_clue"],
            "img_emo":raw_meta["image_emotion"]
        })

    # === 计算总体指标 ===
    senti_result = compute_metrics(all_preds, all_labels)
    result = {"senti_acc": senti_result["acc"], "senti_f1": senti_result["f1"]}

    # === 保存详细样本结果 ===
    if save_path:
        df = pd.DataFrame(meta_records)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved detailed predictions to {save_path}")

    return result


def main(args):
    model_path = '/data/lzy1211/code/model/flan-t5-base/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 创建输出目录与日志 ===
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_model_file = os.path.join(args.output_dir, f"{args.output_model_name}.bin")
    output_logger_file = os.path.join(args.output_dir, f'{args.output_log_name}.txt')

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        filename=output_logger_file
    )
    logger = logging.getLogger(__name__)

    # === 初始化模型与数据 ===
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    # t5_model = T5ForConditionalGeneration.from_pretrained(model_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(args)

    test_data=f"{args.data_dir}/{args.dataset}/test.json"
    test_dataset=MyDataset(args,test_data,tokenizer,type='ana')
    save_path = os.path.join(args.output_dir, f"single_result.csv")

    model = MyFlanT5(model_path=model_path,
        blip_path=args.blip_path,
        tokenizer=tokenizer)
    model.to(device)
    model_state_dict = torch.load(args.output_model_file)
    model.load_state_dict(model_state_dict)

    logger.info("***** Running single-sample evaluation *****")
    logger.info("  Num examples = %d", len(test_dataset))

    results = evaluate_single(model, test_dataset,args.weight, logger, save_path)

    logger.info("***** Evaluation Results *****")
    for k, v in results.items():
        logger.info(f"  {k} = {v}")


if __name__ == "__main__":
    args = get_parser()
    main(args)
