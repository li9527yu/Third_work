import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import json
import torch

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
        self.images_feature_path = os.path.join('/data/lzy1211/code/A2II/instructBLIP/reason_data/data/',args.dataset, 'images_feature')
        self.img_examples=self.creat_img_examples(img_data_dir)
        self.number = len(self.examples)

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

    def get_image_feature(self, image_id):
        image_feature = np.load(os.path.join(self.images_feature_path, image_id[:-4] + '.npz'))['embedding']
        return image_feature
    
    def transform(self,line,img_line):
        max_input_len =self.max_seq_len
        input_hidden_states = img_line["hidden_state"]
        input_pooler_outputs = img_line["pooler_output"]
        value=line
        text = value["text"]
        aspect = value["aspect"]
        output_labels = value["label"]
        relation = value["relation"]
        aspect_Context= value["aspect_Context"]
        image_caption = value["image_caption"]
        image_emotion= value["image_emotion"]
        text_description= value["text_description"]
        sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
        sentiment = sentiment_map[str(output_labels)]

        # instruction_related = "Definition: Combining information from image and the following sentence to identify the sentiment of aspect in the sentence."
        # instruction_irrelevant = "Definition: Based solely on the information in the following sentence to identify the sentiment of aspect in the sentence."
        instruction=f"Definition: Based on the information in the following sentences and the image to identify the sentiment towards {aspect} in the text."
        contain_konw=f"Text: {text}; Text Description: {text_description};"
        semantic_relation_label,emotional_relation_label=get_rel(relation)
        if semantic_relation_label==1:
            contain_konw=contain_konw+ f"Image Caption: {image_caption};"
        if emotional_relation_label==1:
            contain_konw=contain_konw+ f"Image Emotion: {image_emotion};"
        
        if semantic_relation_label==0 and emotional_relation_label==0:
            irrelevant_instruction = f"Definition: Based solely on the information in the following sentence to identify the sentiment towards {aspect} in the text."
            inputs= f'{irrelevant_instruction} {contain_konw} Aspect: {aspect},{aspect_Context}. OPTIONS: -positive -neutral -negative. Output:'
        else:
            inputs= f'{instruction} {contain_konw} Aspect: {aspect},{aspect_Context}. OPTIONS: -positive -neutral -negative. Output:'
    
        model_inputs = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        # 获取 tokenized 的 input_ids 和 attention_mask
        input_ids = model_inputs["input_ids"].squeeze(0)
        input_attention_mask = model_inputs["attention_mask"].squeeze(0)

        # 处理标签
        model_inputs_labels = self.tokenizer(sentiment, padding='max_length', truncation=True, max_length=10, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,output_labels

 
    def transform_three(self,line,img_line):
        max_input_len =self.max_seq_len

        # 图像特征
        # image_id = line['image']
        # image_feature= self.get_image_feature(image_id)  # np (1,196,768)
        # input_hidden_states=torch.from_numpy(image_feature)
        # input_pooler_outputs=0

        input_hidden_states = img_line["hidden_state"]
        input_pooler_outputs = img_line["pooler_output"]
        value=line
        text = value["text"]
        aspect = value["aspect"]
        output_labels = value["label"]
        relation = value["relation"]

        text_emotion= value["textual_clues_parsed"]
        evidence = text_emotion.get("evidence", "").strip()

        sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
        sentiment = sentiment_map[str(output_labels)]

        instruction_related = "QA: Combining information from image and the following sentences to identify the sentiment of aspect."
        instruction_irrelevant = "QA: Based solely on the following sentences to identify the sentiment of aspect."
        
        semantic_relation_label,emotional_relation_label=get_rel(relation)
      
        # 考虑文本、图像的情感线索组合
        if semantic_relation_label==0 and emotional_relation_label==0:
            relation_s=0
            inputs=f"{instruction_irrelevant} Sentence: {text} Sentence Emotion: {evidence} Aspect: {aspect} OPTIONS: -positive -neutral -negative OUTPUT:"
        elif semantic_relation_label==1 and emotional_relation_label==0:
            relation_s=1
            inputs=f"{instruction_related} Sentence: {text} Sentence Emotion: {evidence} Aspect: {aspect} OPTIONS: -positive -neutral -negative OUTPUT:"
        else:
            relation_s=1
            inputs=f"{instruction_related} Sentence: {text} Sentence Emotion: {evidence} Aspect: {aspect} OPTIONS: -positive -neutral -negative OUTPUT:"

        model_inputs = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        input_ids = model_inputs["input_ids"].squeeze(0)
        input_attention_mask = model_inputs["attention_mask"].squeeze(0)
 
        # 处理标签
        model_inputs_labels = self.tokenizer(sentiment, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        return input_ids,input_attention_mask,input_hidden_states,input_pooler_outputs,labels,output_labels,relation_s

 
 