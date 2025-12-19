# DataProcessor.py
import torch
import torch.utils.data as Data
from tqdm import tqdm
import json
import pickle
from PIL import Image
import os


class MyDataset(Data.Dataset):
    """
    端到端数据加载类
    ---------------------------------------
    task_type='relation'  图文相关性预训练
        输入：文本（含<asp>方面<asp>）、图像、关系标签
    task_type='sentiment' 情感分类
        输入：指令 + 文本（含<asp>方面<asp>）+ 图像 + 情感标签
    ---------------------------------------
    数据文件格式（json/pkl均可）:
        {
          "text": "The <asp>room<asp> is clean but small.",
          "aspect": "room",
          "image": "/path/to/image.jpg",
          "relation": 1,
          "sentiment": "positive"
        }
    """
    # train_path, tokenizer,dataset=args.dataset,img_dir=args.img_dir, max_seq_len=args.max_seq_length, task_type=args.task_type
    def __init__(self,data_path,tokenizer,dataset,img_dir, max_seq_len=128, task_type="sentiment"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.task_type = task_type
        self.dataset=dataset
        self.img_dir=img_dir
        # 自动识别文件格式
        if data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        elif data_path.endswith(".pkl"):
            with open(data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            raise ValueError("data_path must end with .json or .pkl")

        self.number = len(self.data)
        self.relation_map={"0":0,"1":1,"2":1,"3":1}
    def __len__(self):
        return self.number

    def __getitem__(self, index):
        item = self.data[index]

        # ============ 通用字段 ============
        text = item["text"]
        aspect = item.get("aspect", "")
        
        # ============ Stage-1 图文关系任务 ============
        if self.task_type == "relation":
            image_path =f"{self.img_dir}/{self.dataset}_images/{item['ImageID'].split('/')[-1]}"
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.getcwd(), image_path)
            image = Image.open(image_path).convert("RGB")

            # 若文本中未显式带<asp>标签，可在此自动插入
            if "<asp>" not in text and aspect:
                text = text.replace("$T$", f"<asp>{aspect}<asp>")
            relation_label = self.relation_map[item.get("relation_s", "0")]
            rel_prompt = f"Determine whether the image is relevant to the aspect '{aspect}'. Sentence: {text}"
            return image, rel_prompt, torch.tensor(relation_label, dtype=torch.long)

        # ============ Stage-2 情感分类任务 ============
        elif self.task_type == "sentiment":
            image_path =f"{self.img_dir}/{self.dataset}_images/{item['image'].split('/')[-1]}"
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.getcwd(), image_path)
            image = Image.open(image_path).convert("RGB")
                # 若文本中未显式带<asp>标签，可在此自动插入
            if "<asp>" not in text and aspect:
                text = text.replace(aspect, f"<asp>{aspect}<asp>")
            sentiment_label = item.get("label", "neutral")
            sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
            sentiment = sentiment_map[str(sentiment_label)]
            # 指令模板
            instruction = (
                "Definition: Identify the sentiment polarity of the aspect in the sentence, "
                "considering the information from the image if it is relevant."
            )
            input_text = f"{instruction}\nSentence: {text}\nOPTIONS: -positive -neutral -negative\nAnswer:"

            model_inputs = self.tokenizer(
                input_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            input_ids = model_inputs["input_ids"].squeeze(0)
            attention_mask = model_inputs["attention_mask"].squeeze(0)

            # 标签转为 T5 可监督形式
            label_inputs = self.tokenizer(
                sentiment_label,
                padding="max_length",
                truncation=True,
                max_length=16,
                return_tensors="pt",
            )
            label_ids = label_inputs["input_ids"].squeeze(0)
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100

            return image, text, input_ids, attention_mask, label_ids,sentiment

        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
