import imp
import logging
import torch
import torch.utils.data as Data
from torch.optim import AdamW
import argparse
import os
from datasets import Dataset
import pickle
import json
import numpy as np
from tqdm import tqdm, trange
import random
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration,DataCollatorForSeq2Seq
from Test_T5_Model import MyFlanT5
from DataProcessor_test import Test_Dataset
from rel_model import Rel_inference
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
# import wandb
from transformers import RobertaTokenizer, RobertaModel
from itertools import cycle
from torch import nn


"""_summary_
对q-former和我们方法在相关性部分做比较实验
1. 在相关性数据集中测试q-former和我们的方法的性能比较
2. 在twitter数据集上测试使用q-former和我们方法对masc任务性能的差异
"""
def post_Rel_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_ids,input_mask,img_id,img_feat,relation_label,input_pooler_outputs=batch    
    
    input_ids=list(map(list, zip(*input_ids)))
    input_mask=list(map(list, zip(*input_mask)))

    input_ids = torch.tensor(input_ids,dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask,dtype=torch.long).to(device)

    
    relation_label=relation_label.to(device).long()

    img_feat = img_feat.clone().detach().to(input_ids.device)
    input_pooler_outputs = input_pooler_outputs.clone().detach().to(input_ids.device)
            
    return input_ids,input_mask,img_id,img_feat,relation_label,input_pooler_outputs

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(y_true, y_pred, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro

# 在相关性test上评测
def evaluate_on_Rel_test(model,Re_test_dataloader, logger):
    model.eval()
    rel_examples = 0
    rel_acc=0
    true_label_list = []
    pred_label_list = []
    # RELation
    for batch in tqdm(Re_test_dataloader, desc="Evaluating-Relation"):
        input_ids,input_mask,img_id,img_feat,relation_label,input_pooler_outputs=post_Rel_dataloader(batch)
        with torch.no_grad():
            Rel_score=model(rel_inputs_id=input_ids,
                            rel_inputs_mask=input_mask,
                            img_feat=img_feat,
                            rel_label=relation_label)
            Rel_score=Rel_score.detach().cpu().numpy()
            relation_pred = np.argmax(Rel_score, axis=1)
            tmp_rel_accuracy=np.sum(relation_pred == relation_label.cpu().numpy()) 
            true_label_list.extend(relation_label.cpu().numpy())
            pred_label_list.extend(relation_pred)
            rel_acc += tmp_rel_accuracy
            rel_examples+= input_ids.size()[0]

    rel_acc = rel_acc/ rel_examples
    test_rel_precision, test_rel_recall, test_rel_F_score = macro_f1(true_label_list, pred_label_list)

    result = {
        'Test_rel_acc':rel_acc,
        'Test_rel_precision':test_rel_precision,
        'Test_rel_recall':test_rel_recall,
        'Test_rel_F_score':test_rel_F_score}
    logger.info("***** Test Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

# q-former 相关性测试
def evaluate_on_Rel_test_QFormer(model,Re_test_dataloader, logger):
    model.eval()
    rel_examples = 0
    rel_acc=0
    true_label_list = []
    pred_label_list = []
    # RELation
    for batch in tqdm(Re_test_dataloader, desc="Evaluating-Relation"):
        input_ids,input_mask,img_id,img_feat,relation_label,input_pooler_outputs=post_Rel_dataloader(batch)
        with torch.no_grad():
            Rel_score=model(rel_inputs_id=input_ids,
                            rel_inputs_mask=input_mask,
                            img_feat=img_feat,
                            rel_label=relation_label,
                            input_pooler_output=input_pooler_outputs)
            Rel_score=Rel_score.detach().cpu().numpy()
            relation_pred = np.argmax(Rel_score, axis=1)
            tmp_rel_accuracy=np.sum(relation_pred == relation_label.cpu().numpy()) 
            true_label_list.extend(relation_label.cpu().numpy())
            pred_label_list.extend(relation_pred)
            rel_acc += tmp_rel_accuracy
            rel_examples+= input_ids.size()[0]

    rel_acc = rel_acc/ rel_examples
    test_rel_precision, test_rel_recall, test_rel_F_score = macro_f1(true_label_list, pred_label_list)

    result = {
        'Test_rel_acc':rel_acc,
        'Test_rel_precision':test_rel_precision,
        'Test_rel_recall':test_rel_recall,
        'Test_rel_F_score':test_rel_F_score}
    logger.info("***** Test Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))


if __name__ == "__main__":
    # args=get_parser()
    # main(args)

    # 设置参数
    # output_dir='/public/home/ghfu/lzy/code/instructBLIP/results/twitter2015-a2ii-first_work/'
    output_dir='/public/home/ghfu/lzy/code/instructBLIP/results/twitter2015/'
    output_model_file = os.path.join(output_dir, f"pytorch_model.bin")
    output_logger_file=os.path.join(output_dir,f'log.txt')
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = logging.INFO,
                filename=output_logger_file)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path='/public/home/ghfu/lzy/model/flan-t5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    Roberta_tokenizer=RobertaTokenizer.from_pretrained('/public/home/ghfu/lzy/model/roberta-base')
    t5_model=T5ForConditionalGeneration.from_pretrained(model_path)
    Re_data_dir='/public/home/ghfu/lzy/code/instructBLIP/relation_data'
    batch_size=8
    max_seq_length=128
    img_dir=f"/public/home/ghfu/lzy/code/instructBLIP/img_feat/twitter2017.pkl"
    img_feat={}
    try:
        with open(img_dir, 'rb') as f:  # 以二进制模式打开文件
            img_feat = pickle.load(f)  # 反序列化
    except Exception as e:
        print(f"读取 .pkl 文件时出错: {e}")
    # 加载数据
    Re_test_data=f"{Re_data_dir}/test.json"
    Re_test_dataset=Test_Dataset(Re_test_data,tokenizer,Roberta_tokenizer,img_feat, max_seq_len= max_seq_length)
    Re_test_dataloader= Data.DataLoader(dataset=Re_test_dataset,shuffle=True, batch_size=batch_size,num_workers=0)


    # 加载我们的方法模型
    # Load a trained model that you have fine-tuned
    # model_state_dict = torch.load(output_model_file)
    # model = MyFlanT5(model_path=t5_model)
    # model.load_state_dict(model_state_dict)
    # model.to(device)
    # 进行测试
    # evaluate_on_Rel_test(model,Re_test_dataloader, logger)

    model_state_dict = torch.load(output_model_file)
    model = Rel_inference(model_path=t5_model)
    model.load_state_dict(model_state_dict)
    model.to(device)

    # 进行测试
    evaluate_on_Rel_test_QFormer(model,Re_test_dataloader, logger)
    



