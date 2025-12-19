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
from DataProcessor import MyDataset
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
def post_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
    relation_input_mask,img_feat,relation_label=batch
    
    # RR
    relation_input_ids=list(map(list, zip(*relation_input_ids)))
    relation_input_mask=list(map(list, zip(*relation_input_mask)))
    relation_input_ids = torch.tensor(relation_input_ids,dtype=torch.long).to(device)
    relation_input_mask = torch.tensor(relation_input_mask,dtype=torch.long).to(device)

    
    relation_label=relation_label.to(device).long()

    img_feat = img_feat.clone().detach().to(relation_input_ids.device)

    # ss
        
    input_re_ids = input_re_ids.clone().detach().long().to(device)
    input_re_attention_mask = input_re_attention_mask.clone().detach().float().to(device)
    input_ir_ids = input_ir_ids.clone().detach().long().to(device)
    input_ir_attention_mask = input_ir_attention_mask.clone().detach().float().to(device)
    
    labels=labels.to(device).long()

    input_pooler_outputs = input_pooler_outputs.clone().detach().to(input_re_ids.device)
    input_hidden_states = input_hidden_states.clone().detach().to(input_re_ids.device)

            
    return input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
    relation_input_mask,img_feat,relation_label

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
        input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
            relation_input_mask,img_feat,relation_label= post_dataloader(batch)
        with torch.no_grad():
            Rel_score=model(rel_inputs_id=relation_input_ids,
                        rel_inputs_mask=relation_input_mask,
                            img_feat=img_feat,
                            rel_label=relation_label)
            Rel_score=Rel_score.detach().cpu().numpy()
            relation_pred = np.argmax(Rel_score, axis=1)
            pred_label_list.extend(relation_pred)

    output_path='/public/home/ghfu/lzy/code/instructBLIP/A2II_first_work/analysis/twitter_rel_pred/our_pred.pkl'
    with open(output_path, "wb") as f:  # 以二进制写模式打开文件
        pickle.dump(pred_label_list, f)  # 将列表序列化并写入文件
    
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(pred_label_list, f, ensure_ascii=False, indent=2) 
    print("done")
    

# q-former 相关性测试
def evaluate_on_Rel_test_QFormer(model,Re_test_dataloader, logger):
    model.eval()
    rel_examples = 0
    rel_acc=0
    true_label_list = []
    pred_label_list = []
    # RELation
    for batch in tqdm(Re_test_dataloader, desc="Evaluating-Relation"):
        input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
            relation_input_mask,img_feat,relation_label= post_dataloader(batch)
        with torch.no_grad():
            Rel_score=model(rel_inputs_id=relation_input_ids,
                        rel_inputs_mask=relation_input_mask,
                            img_feat=img_feat,
                            rel_label=relation_label,
                            input_pooler_output=input_pooler_outputs)
            Rel_score=Rel_score.detach().cpu().numpy()
            relation_pred = np.argmax(Rel_score, axis=1)
            pred_label_list.extend(relation_pred)

    output_path='/public/home/ghfu/lzy/code/instructBLIP/A2II_first_work/analysis/twitter_rel_pred/qformer_pred.pkl'
    with open(output_path, "wb") as f:  # 以二进制写模式打开文件
        pickle.dump(pred_label_list, f)  # 将列表序列化并写入文件
    print("done")


if __name__ == "__main__":
    # =get_parser()
    # main()

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
    data_dir='/public/home/ghfu/lzy/code/instructBLIP/img_data'
    dataset='twitter2015'
    batch_size=8
    max_seq_length=128
    img_dir=f"/public/home/ghfu/lzy/code/instructBLIP/img_feat/{dataset}.pkl"
    img_feat={}
    try:
        with open(img_dir, 'rb') as f:  # 以二进制模式打开文件
            img_feat = pickle.load(f)  # 反序列化
    except Exception as e:
        print(f"读取 .pkl 文件时出错: {e}")
    # 加载数据
    test_data=f"{data_dir}/{dataset}/test.pkl"
    test_dataset=MyDataset(test_data,tokenizer,Roberta_tokenizer,img_feat, max_seq_len= max_seq_length)
    test_dataloader= Data.DataLoader(dataset=test_dataset,shuffle=True, batch_size=batch_size,num_workers=0)


    # # 加载我们的方法模型
    # # Load a trained model that you have fine-tuned
    # model_state_dict = torch.load(output_model_file)
    # model = MyFlanT5(model_path=t5_model)
    # model.load_state_dict(model_state_dict)
    # model.to(device)
    # # 进行测试
    # evaluate_on_Rel_test(model,test_dataloader, logger)

    model_state_dict = torch.load(output_model_file)
    model = Rel_inference(model_path=t5_model)
    model.load_state_dict(model_state_dict)
    model.to(device)

    # 进行测试
    evaluate_on_Rel_test_QFormer(model,test_dataloader, logger)
    



