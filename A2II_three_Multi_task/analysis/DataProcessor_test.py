import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import logging
import torch
import json

class Test_Dataset(Data.Dataset):
    def __init__(self,data_dir,tokenizer,Roberta_tokenizer,img_feat,max_seq_len=64):
        self.tokenizer=tokenizer
        self.Roberta_tokenizer=Roberta_tokenizer
        self.max_seq_len=max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.qformer_examples=self.creat_qformer_examples('/public/home/ghfu/lzy/code/instructBLIP/img_data/twitter2017/train.pkl')
        self.img_examples=img_feat
        self.number = len(self.examples)
        self.relation_label_list=self.get_relation_labels()

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        # qformer_item=self.qformer_examples[index]
        return self.transform(line)   
    def get_relation_labels(self):
        return ["0","1"]
    def creat_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=json.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples


    def creat_qformer_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples

    def transform(self,line):
        max_input_len =self.max_seq_len
        value=line

        input_texts = value["text"]
        input_aspects = value["aspect"]
        # img
        img_id=os.path.splitext(os.path.basename(value["ImageID"]))[0]
        img_feat=self.img_examples[img_id]
        input_pooler_outputs={}
        for item in self.qformer_examples:
            qformer_examples_imgid=os.path.splitext(os.path.basename(item["image"]))[0]
            if qformer_examples_imgid==img_id and item['aspect']==input_aspects:
                input_pooler_outputs = item["pooler_output"]
                break

        # sentiment_label = value["sentiment"]
        relation_label=-1
        relation_label_map = {label : i for i, label in enumerate(self.relation_label_list)}

        # relation:
        input_ids=self.Roberta_tokenizer(input_texts.lower(),input_aspects.lower())['input_ids']   #  <s>text_a</s></s>text_b</s>
        input_mask=[1]*len(input_ids)
        padding_id = [1]*(max_input_len-len(input_ids)) #<pad> :1
        padding_mask=[0]*(max_input_len-len(input_ids)) 

        input_ids += padding_id
        input_mask += padding_mask
        
        
        # rel
        rel=value['relation_s'] 
        # 改进 将关系类型分为3类
        if rel:
            if rel=='2' or rel=='3' :
                rel='1'
            relation_label=relation_label_map[rel]


        return input_ids,input_mask,img_id,img_feat,relation_label,input_pooler_outputs