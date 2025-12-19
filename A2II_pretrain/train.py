# train.py
import logging
import torch
import torch.utils.data as Data
from torch.optim import AdamW
import argparse
import os
import numpy as np
from tqdm import tqdm, trange
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

from T5_Model import MyFlanT5
from DataProcessor import MyDataset
from sklearn.metrics import accuracy_score, f1_score
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch
import numpy as np

# ===============================
# 参数定义
# ===============================
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2017', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--img_dir', default='/data/lzy1211/code/twitterImage', type=str)
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--task_type', default='sentiment', type=str, choices=['relation', 'sentiment'])
    parser.add_argument('--pretrained_relation_path', default=None, type=str,
                        help='Stage-2 时加载的关系预训练权重路径')
    parser.add_argument('--model_path', default='/data/lzy1211/code/model/flan-t5-base/', type=str,
                        help='t5 路径')
    parser.add_argument('--blip_path', default='/data/lzy1211/code/model/instructblip-flan-t5-xl', type=str,
                        help='InstructBLIP 模型路径（含Flan-T5与Q-Former）')
    parser.add_argument('--BATCH_SIZE', default=8, type=int)
    parser.add_argument('--LEARNING_RATE', default=5e-5, type=float)
    parser.add_argument('--EPOCHS', default=3, type=int)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--run_name', default='A2II_train', type=str)
    parser.add_argument('--use_qformer_online', action='store_true',
                        help='若指定，则启用端到端Q-Former特征计算（不使用离线pkl特征）')
    parser.add_argument('--freeze_vision', action='store_true', help='是否冻结ViT视觉模型')
    parser.add_argument('--freeze_qformer', action='store_true', help='是否冻结Q-Former')
    return parser.parse_args()


# ===============================
# 通用函数
# ===============================
class EarlyStopping:
    def __init__(self, patience=5, mode="min", delta=1e-4):
        """
        mode = "min" -> 监控 loss
        mode = "max" -> 监控 f1
        patience: 连续多少 epoch 没提升就停止
        delta: 最小改善幅度
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta

        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def step(self, metric_value):
        """
        输入当前 epoch 的 loss 或 f1
        返回是否 early_stop
        """

        if self.best_score is None:
            self.best_score = metric_value
            return False

        if self.mode == "min":   # loss 越低越好
            improvement = self.best_score - metric_value
        else:                    # f1 越高越好
            improvement = metric_value - self.best_score

        if improvement > self.delta:
            # 有改善
            self.best_score = metric_value
            self.counter = 0
        else:
            # 无改善
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def post_dataloader(batch, device):
    images, text, labels = batch

    # input_ids 和 attention_mask 由 tokenizer 编码，直接转设备
    input_ids = input_ids.long().to(device)
    attention_mask = attention_mask.long().to(device)
    if isinstance(labels, torch.Tensor) and labels.dim() == 0:
        # 单个 int 标签
        labels = labels.long().to(device)
    else:
        labels = labels.to(device)

    return images, input_ids, attention_mask, labels

def collate_fn(batch):
    elem = batch[0]
    if len(elem) == 3:  # relation
        images, texts, labels = zip(*batch)
        labels = torch.stack(labels)
        return list(images), list(texts), labels
    else:  # sentiment
        images, texts, input_ids, attention_mask, labels,sentiments = zip(*batch)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        return list(images), list(texts), input_ids, attention_mask, labels,list(sentiments)

# ===============================
# 训练阶段一：图文关系预训练
# ===============================
@torch.no_grad()
def evaluate_relation(model, val_loader, device, logger):
    """
    评估图文关系分类准确率与 F1。
    """
    model.eval()
    preds, labels_true = [], []

    for batch in tqdm(val_loader, desc="Evaluating Relation"):
        images, texts, labels = batch
        labels = labels.to(device)

        outputs = model(
            images=images,
            text=texts,
            labels=labels,
            task_type="relation"
        )
        logits = outputs["logits"]
        preds_batch = torch.argmax(logits, dim=-1)
        preds.extend(preds_batch.cpu().numpy())
        labels_true.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_true, preds)
    f1 = f1_score(labels_true, preds, average="macro")
    logger.info(f"📊 Relation Eval — Acc={acc:.4f}, F1={f1:.4f}")

    return {"acc": acc, "f1": f1}


def train_relation(args, model, train_loader,val_loader, optimizer, logger, device):
    logger.info("===== Stage 1: Relation Pretraining =====")
    t_total = len(train_loader) * args.EPOCHS
    warmup_steps = int(0.1 * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total
    )
    logger.info(f"[Relation] Total Steps={t_total}, Warmup={warmup_steps}")
    early_stopper = EarlyStopping(patience=3, mode="max")

    best_f1 = 0.0
    model.train()
    for epoch in trange(args.EPOCHS, desc="Relation Epoch"):
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training Relation"):
            images, text, labels =batch
            outputs = model(
                images=images,
                text=text,
                labels=labels.to(device),
                task_type='relation'
            )
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()   
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"[Epoch {epoch}] Relation Loss: {avg_loss:.4f}")

                # ---------- Validation ----------
        val_result = evaluate_relation(model, val_loader, device, logger)
        acc, f1 = val_result["acc"], val_result["f1"]

        # ---------- Save best ----------
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.output_model_file )
            logger.info(f"✅ New best Relation model saved (F1={f1:.4f})")

        # ---------- Early Stop 判断 ----------
        if early_stopper.step(f1):
            logger.info(f"📌 Early stopping triggered at epoch {epoch}! Best F1={best_f1:.4f}")
            break
    logger.info(f"Best Relation F1: {best_f1:.4f}")
    return best_f1



# ===============================
# 训练阶段二：情感分类微调
# ===============================

@torch.no_grad()
def evaluate(model, data_loader, logger, device, max_new_tokens: int = 16):
    """
    通用生成式评估函数
    ---------------------------------
    Args:
        model: MyFlanT5 模型（已包含 generate 方法）
        data_loader: 验证或测试 DataLoader
        tokenizer: T5Tokenizer
        logger: logging.Logger
        device: torch.device
        max_new_tokens: 每次生成的最大新 token 数（默认 8）
    Returns:
        result: dict(acc, f1, preds, trues)
    """
    model.eval()
    pred_texts = []
    y_true=[]
    for batch in tqdm(data_loader, desc="Evaluating"):
        images, texts, input_ids, attention_mask, labels,sentiments = batch
        input_ids = input_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)
        labels = labels.to(device)

        # ===== 使用模型 generate() 推理 =====
        outputs = model.generate(
            images=images,
            text=texts,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False
        )
        pred_texts.extend(outputs)
        y_true.extend(sentiments)

    # ===== 清洗输出（去除空白和大小写）=====
    pred_texts = [p.strip().lower() for p in pred_texts]
    # ===== 统一标签映射 =====
    label_map = {"positive": 1, "negative": -1, "neutral": 0}
    def map_label(x): return label_map.get(x, 0)
    y_pred = [map_label(p) for p in pred_texts]

    # ===== 过滤非法样本 =====
    valid_idx = [i for i in range(len(y_true)) if y_true[i] in [-1, 0, 1]]
    if len(valid_idx) == 0:
        logger.warning("No valid samples found for evaluation!")
        return {"acc": 0.0, "f1": 0.0}

    y_true = np.array(y_true)[valid_idx]
    y_pred = np.array(y_pred)[valid_idx]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    result = {
        "acc": acc,
        "f1": f1
    }

    logger.info(f"📊 Evaluation Results — Acc={acc:.4f}, Macro-F1={f1:.4f}")
    return result


def train_sentiment(args, model, train_loader, val_loader, optimizer, logger, device):
    """
    端到端多模态情感分类微调
    - 输入: image(list of PIL), text(list of str for Q-Former), input_ids, attention_mask, labels
    - 模型: Q-Former + Flan-T5 (online特征)
    """
    logger.info("===== Stage 2: Sentiment Fine-tuning =====")
    best_f1 = 0.0
    t_total = len(train_loader) * args.EPOCHS   # 总步数

    warmup_steps = int(0.1 * t_total)   # 或根据 warmup_ratio 动态计算
    logger.info(f"Total steps = {t_total}, warmup = {warmup_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total
    )

    early_stopper = EarlyStopping(patience=5, mode="max")   

    for epoch in trange(args.EPOCHS, desc="Sentiment Epoch"):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            images, texts, input_ids, attention_mask, labels,sentiments = batch

            # 将 T5 的输入移到 GPU
            input_ids = input_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            labels = labels.to(device)

            outputs = model(
                images=images,        # List[PIL.Image]，内部 processor 自动处理
                text=texts,           # List[str]，供 Q-Former 文本输入
                input_ids=input_ids,  # 指令化 token 序列
                attention_mask=attention_mask,
                labels=labels,
                task_type="sentiment"
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()   
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"[Epoch {epoch}] Sentiment Loss: {avg_loss:.4f}")

        # ===================== Dev 评估 =====================
        logger.info("***** Running evaluation on Dev Set*****")
        model.eval()
        dev_result=evaluate(model,val_loader, logger, device)
        # ========== 保存最佳模型 ==========
        if dev_result['f1'] >= best_f1:
            best_f1 = dev_result['f1']
            
            torch.save(model.state_dict(), args.output_model_file)
            logger.info(f"New best model saved (F1={dev_result['f1']:.4f})")

        # ---------- Early Stop 判断 ----------
        if early_stopper.step(dev_result["f1"]):
            logger.info(f"📌 Early stopping triggered at epoch {epoch}! Best F1={best_f1:.4f}")
            break
    logger.info(f"Best F1: {best_f1:.4f}")


# ===============================
# 主程序
# ===============================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{args.task_type}_train.log")
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=log_path
    )
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    logger.info(args)
    # 添加特殊 token
    special_tokens_dict = {'additional_special_tokens': ['<asp>']}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    # 初始化模型
    model = MyFlanT5(
        model_path=args.model_path,
        blip_path=args.blip_path,
        tokenizer=tokenizer,
        task_type=args.task_type,
        use_qformer_online=args.use_qformer_online,
        freeze_vision=args.freeze_vision,
        freeze_qformer=args.freeze_qformer
    ).to(device)

    # ============ 加载数据 ============
    if args.task_type=="relation":
        args.output_model_file = os.path.join(args.output_dir, "relation_pretrained.bin")
        train_path = f"{args.data_dir}//train.json"
        val_path = f"{args.data_dir}/dev.json"
    else:
        args.output_model_file = os.path.join(args.output_dir, "sentiment_finetuned.bin")
        train_path = f"{args.data_dir}/{args.dataset}/train.json"
        val_path = f"{args.data_dir}/{args.dataset}/val.json"
        test_path = f"{args.data_dir}/{args.dataset}/test.json"
    
    train_dataset = MyDataset(train_path, tokenizer,dataset=args.dataset,img_dir=args.img_dir, max_seq_len=args.max_seq_length, task_type=args.task_type)
    val_dataset = MyDataset(val_path, tokenizer,dataset=args.dataset,img_dir=args.img_dir, max_seq_len=args.max_seq_length, task_type=args.task_type)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 优化器
    # --------- optimizer param groups ---------
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # --------- AdamW optimizer ---------
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # optimizer = AdamW(model.parameters(), lr=args.LEARNING_RATE)
    
    # ============ Stage-1：关系预训练 ============
    if args.task_type == 'relation':
        train_relation(args, model, train_loader,val_loader, optimizer, logger, device)

    # ============ Stage-2：情感微调 ============
    elif args.task_type == 'sentiment':
        # 可选加载Stage-1权重
        if args.pretrained_relation_path and os.path.exists(args.pretrained_relation_path):
            logger.info(f"Loading relation-pretrained weights: {args.pretrained_relation_path}")
            state_dict = torch.load(args.pretrained_relation_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        train_sentiment(args, model, train_loader, val_loader, optimizer, logger, device)

        test_dataset = MyDataset(test_path, tokenizer,dataset=args.dataset,img_dir=args.img_dir, max_seq_len=args.max_seq_length, task_type=args.task_type)
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=args.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        # =====================Test 评估 =====================
        logger.info("***** Running evaluation on Test Set*****")
        model.eval()
        model_state_dict = torch.load(args.output_model_file)
        model.load_state_dict(model_state_dict)
        test_results=evaluate(model,test_loader, logger, device)
        logger.info("***** Test Eval results *****")
        for key in sorted(test_results.keys()):
            logger.info("  %s = %s", key, str(test_results[key]))


if __name__ == "__main__":
    args = get_parser()
    main(args)
