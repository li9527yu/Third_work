import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import json

def get_rel(relation):
    semantic_rel,emotional_rel=relation.split(',')
    semantic_relation_label,emotional_relation_label=0,0
    if 'irrelevant' in semantic_rel:
        semantic_relation_label=0
    else:
        semantic_relation_label=1

    if 'irrelevant' in emotional_rel:
        emotional_relation_label=0
    else:
        emotional_relation_label=1
   
    return semantic_relation_label,emotional_relation_label
class MyDataset(Data.Dataset):
    def __init__(self,args,data_dir,img_data_dir,tokenizer):
        self.tokenizer=tokenizer
        self.max_seq_len=args.max_seq_len
        self.examples=self.creat_examples(data_dir)
        self.img_examples=self.creat_img_examples(img_data_dir)
        self.number = len(self.examples)
        self.instruction_text = (
            "QA: Based on the sentence, identify the sentiment of the given aspect."
        )
        # self.instruction_mm = (
        #     "QA: Combining information from the image and the sentence, "
        #     "identify the sentiment of the given aspect."
        # )
        self.instruction_mm = (
            "QA: Analyze the image emotion and sentence. Explain the reasoning" "first, then identify the sentiment of the aspect."
        )   
        self.instruction_imgemo = "QA: Describe the image sentiment and the reason."

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        img_line=self.img_examples[index]
        return self.transform_three(line,img_line)   
        # return self.transform_three(line,img_line)   

    def creat_examples(self,data_dir):
        with open(data_dir,"r") as f:
            dict=json.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples

    def creat_img_examples(self,data_dir):
        with open(data_dir,"rb") as f:
            dict=pickle.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples
 
    def transform_three(self,line,img_line):

        input_hidden_states = img_line["hidden_state"]
        input_pooler_outputs = img_line["pooler_output"]
        value=line
        text = value["text"]
        aspect = value["aspect"]
        output_labels = value["label"]
        
        
        # image_caption = value["image_caption"]
        image_emotion= value["image_emotion"]

        sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
        sentiment = sentiment_map[str(output_labels)]
        text_sentiment = value['textual_clues_parsed']['polarity']
        
        relation_s=1
        

        inputs_text = (
            f"{self.instruction_text} "
            f"Sentence: {text} "
            f"Aspect: {aspect} "
            f"OPTIONS: -positive -neutral -negative "
            f"OUTPUT:"
        )
        model_inputs_text = self.tokenizer(
            inputs_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        input_ids_text = model_inputs_text["input_ids"].squeeze(0)
        attention_mask_text = model_inputs_text["attention_mask"].squeeze(0)

        inputs_image = (
            f"{self.instruction_mm} "
            f"Sentence: {text} "
            f"Image Emotion: {image_emotion} "
            f"Aspect: {aspect} "
            f"OUTPUT:"
        )
        model_inputs_image = self.tokenizer(
            inputs_image,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )

        input_ids_image = model_inputs_image["input_ids"].squeeze(0)
        attention_mask_image = model_inputs_image["attention_mask"].squeeze(0)


        

        # 处理标签
        labels_text = self.tokenizer(
            text_sentiment,
            padding="max_length",
            truncation=True,
            max_length=8,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        labels_text[labels_text == self.tokenizer.pad_token_id] = -100

        
        # —— 主任务标签（multimodal）
        target_mm_str = f"{image_emotion} Therefore, the sentiment is {sentiment}."
        labels_mm = self.tokenizer(
            target_mm_str,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        labels_mm[labels_mm == self.tokenizer.pad_token_id] = -100

 
        return input_ids_text,input_ids_image,attention_mask_text,\
            attention_mask_image,input_hidden_states,input_pooler_outputs,labels_text,labels_mm,output_labels,relation_s
    


 