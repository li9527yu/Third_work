# DataProcessor.py
import torch
import torch.utils.data as Data
from tqdm import tqdm
import json
from PIL import Image
import os


class ECCLDataset(Data.Dataset):
    """
    ECCL 预训练数据集
    -----------------------------
    data 格式（eccl_*.json）示例：
    {
        "text": "The <asp>room<asp> is clean but small.",
        "aspect": "room",
        "image": "/data/.../twitter2017_images/123.jpg",
        "relation_s": 1,
        "text_emotion": "positive",
        "image_emotion": "positive",
        "eccl_label": 1      # 1=情感一致(正样本), 0=不一致/无关(负样本)
    }
    -----------------------------
    __getitem__ 返回：
        images: PIL.Image
        texts : str  (用于喂给 InstructBLIP Q-Former 的文本)
        labels: float tensor (0. or 1.)
    """

    def __init__(self, data_path, dataset, img_dir):
        """
        Args:
            data_path: eccl_train.json / eccl_val.json 路径
            dataset  : 'twitter2015' or 'twitter2017'
            img_dir  : 图像根路径，例如 /data/.../twitterImage
        """
        self.data_path = data_path
        self.dataset = dataset
        self.img_dir = img_dir

        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.number = len(self.data)
        print(f"[ECCLDataset] Loaded {self.number} samples from {self.data_path}")

    def __len__(self):
        return self.number

    def __getitem__(self, index):
        item = self.data[index]

        text = item["text"]
        aspect = item.get("aspect", "")

        # 若原始 text 中是 $T$ 占位符，则自动替换为 <asp>aspect<asp>
        if "$T$" in text and aspect:
            text = text.replace("$T$", f"<asp>{aspect}<asp>")

        # 构造图像路径（通常 image 字段已经是完整路径，如果不是则按你原来的路径拼）
        img_field = item["image"]
        # 如果是全路径直接用，如果只是文件名则拼 dataset_images
        if os.path.isabs(img_field):
            image_path = img_field
        else:
            image_name = os.path.basename(img_field)
            image_path = os.path.join(self.img_dir, f"{self.dataset}_images", image_name)

        image = Image.open(image_path).convert("RGB")

        eccl_label = float(item["eccl_label"])
        label_tensor = torch.tensor(eccl_label, dtype=torch.float)

        return image, text, label_tensor


def eccl_collate_fn(batch):
    """
    ECCL 任务专用 collate_fn
    输入 batch: List[(image, text, label)]
    输出:
        images: List[PIL.Image]
        texts : List[str]
        labels: Tensor[B]
    """
    images, texts, labels = zip(*batch)
    labels = torch.stack(labels)  # [B]
    return list(images), list(texts), labels
