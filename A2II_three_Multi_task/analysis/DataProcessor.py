import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import logging
import torch

class MyDataset(Data.Dataset):
    def __init__(self,data_dir,tokenizer,Roberta_tokenizer,img_feat,max_seq_len=64):
        self.tokenizer=tokenizer
        self.Roberta_tokenizer=Roberta_tokenizer
        self.max_seq_len=max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.img_examples=img_feat
        self.number = len(self.examples)

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        return self.transform(line)   

    def creat_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples

    def transform(self,line):
        max_input_len =self.max_seq_len
        value=line

        instruction_related = "Definition: Combining information from image and the following sentence to identify the sentiment of aspect in the sentence."
        instruction_irrelevant = "Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."

        input_texts = value["text"]
        input_aspects = value["aspect"]
        input_hidden_states = value["hidden_state"]
        input_pooler_outputs = value["pooler_output"]
        output_labels = value["sentiment"]

        """
        相关性部分
        """
        relation_label=-1

        # relation:
        original_text=input_texts.replace(input_aspects,"$T$")
        relation_input_ids=self.Roberta_tokenizer(original_text.lower(),input_aspects.lower())['input_ids']   #  <s>text_a</s></s>text_b</s>
        relation_input_mask=[1]*len(relation_input_ids)
        padding_id = [1]*(max_input_len-len(relation_input_ids)) #<pad> :1
        padding_mask=[0]*(max_input_len-len(relation_input_ids)) 

        relation_input_ids += padding_id
        relation_input_mask += padding_mask
        
        # img
        img_id=os.path.splitext(os.path.basename(value["image"]))[0]
        img_feat=self.img_examples[img_id]
        """
        情感部分
        """
        input_re = f'{instruction_related} Sentence: {input_texts} aspect: {input_aspects} OPTIONS: -positive -neutral -negative output:' 
        input_ir = f'{instruction_irrelevant} Sentence: {input_texts} aspect: {input_aspects} OPTIONS: -positive -neutral -negative output:'

        model_inputs_re = self.tokenizer(input_re, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        model_inputs_ir = self.tokenizer(input_ir, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")

        # 获取 tokenized 的 input_ids 和 attention_mask
        input_re_ids = model_inputs_re["input_ids"].squeeze(0)
        input_re_attention_mask = model_inputs_re["attention_mask"].squeeze(0)
        input_ir_ids = model_inputs_ir["input_ids"].squeeze(0)
        input_ir_attention_mask = model_inputs_ir["attention_mask"].squeeze(0)

        input_hidden_states = torch.tensor(input_hidden_states).to(input_re_ids.device)  # 确保在同一个设备上
        input_pooler_outputs = torch.tensor(input_pooler_outputs).to(input_re_ids.device)  # 确保在同一个设备上

        # 处理标签
        model_inputs_labels = self.tokenizer(output_labels, padding='max_length', truncation=True, max_length=5, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        return input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels,relation_input_ids,\
                relation_input_mask,img_feat,relation_label
