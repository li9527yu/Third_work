import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import json
import pickle
import numpy as np
import os
import logging
import torch

class ReDataset(Data.Dataset):
    def __init__(self,data_dir,tokenizer,img_feat,max_seq_len=64):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.img_examples=img_feat
        self.examples=self.creat_examples(data_dir)
        self.number = len(self.examples)
        self.relation_label_list=self.get_relation_labels()

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        return self.transform(line)   

    def creat_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=json.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples

    def get_relation_labels(self):
        return ["0","1"]

    def transform(self,line):
        max_input_len =self.max_seq_len
        value=line

        input_texts = value["text"]
        input_aspects = value["aspect"]
        inputs=input_texts+input_aspects
        relation_label=-1
        relation_label_map = {label : i for i, label in enumerate(self.relation_label_list)}

        # relation:
        input_text = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        input_ids = input_text["input_ids"].squeeze(0)
        input_mask = input_text["attention_mask"].squeeze(0)


        
        # img
        img_id=os.path.splitext(os.path.basename(value["ImageID"]))[0]
        img_feat=self.img_examples[img_id]
        # rel
        rel=value['relation_s'] 
        # 改进 将关系类型分为3类
        if rel:
            if rel=='2' or rel=='3' :
                rel='1'
            relation_label=relation_label_map[rel]


        return input_ids,input_mask,img_id,img_feat,relation_label
