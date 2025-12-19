import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
from PIL import Image
import pickle
import numpy as np
import os
import json

class MyDataset(Data.Dataset):
    def __init__(self,args,data_dir,tokenizer,type='train'):
        self.tokenizer=tokenizer
        self.max_seq_len=args.max_seq_len
        self.img_dir=args.img_dir
        self.dataset=args.dataset
        self.type=type
        self.examples=self.creat_examples(data_dir)
        self.number = len(self.examples)

    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line=self.examples[index]
        
        return self.transform_three(line)   
    
    def creat_examples(self,data_dir):
        with open(data_dir,"r") as f:
            dict=json.load(f)
        examples=[]
        for item in tqdm(dict,desc="CreatExample"):
            examples.append(item)
        return examples

    def transform_three(self,line):
        max_input_len =self.max_seq_len

        value=line
        text = value["text"]
        aspect = value["aspect"]
        output_labels = value["label"]
        relation = value["relation"]
        text_clue= value["text_clue"]
        image_emotion= value["image_emotion"]
        sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
        relation_map = {
            "the semantic relevance is relevant, the emotional relevance is irrelevant": 1,
            "the semantic relevance is irrelevant, the emotional relevance is irrelevant":0,
            "the semantic relevance is relevant, the emotional relevance is relevant": 2}
        # 0: 无关；1是语义相关；2是情感相关

        sentiment = sentiment_map[str(output_labels)]
        relation_s=relation_map[relation]

        # gemini的文本情感线索
        # text_emotion= value["textual_clues_parsed"]
        # text_clue = text_emotion.get("evidence", "").strip()
        
        image_path =f"{self.img_dir}/{self.dataset}_images/{line['image'].split('/')[-1]}"
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.getcwd(), image_path)
        image = Image.open(image_path).convert("RGB")

        instruction_related = "QA: Combining information from image and the following sentences to identify the sentiment of aspect."

        # relation_s=1

        inputs_text=f"{instruction_related} Sentence: {text} Sentence Emotion: {text_clue} Aspect: {aspect} OPTIONS: -positive -neutral -negative OUTPUT:"
        inputs_image=f"{instruction_related} Sentence: {text} Image Emotion: {image_emotion} Aspect: {aspect} OPTIONS: -positive -neutral -negative OUTPUT:"
        
        model_inputs_text = self.tokenizer(inputs_text, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        input_ids = model_inputs_text["input_ids"].squeeze(0)
        input_attention_mask = model_inputs_text["attention_mask"].squeeze(0)

        model_inputs_image = self.tokenizer(inputs_image, padding='max_length', truncation=True, max_length=max_input_len, return_tensors="pt")
        input_multi_ids = model_inputs_image["input_ids"].squeeze(0)
        input_multi_attention_mask = model_inputs_image["attention_mask"].squeeze(0)

        # 处理标签
        model_inputs_labels = self.tokenizer(sentiment, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        model_inputs_labels['input_ids'][model_inputs_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        labels = model_inputs_labels["input_ids"].squeeze(0)

        if self.type=='ana':
            return input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
                aspect,image,labels,output_labels,relation_s,line
        else:
            return input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
                    aspect,image,labels,output_labels,relation_s


    