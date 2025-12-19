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
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from T5_Model import MyFlanT5_ECCL
from DataProcessor import ECCLDataset, eccl_collate_fn
from sklearn.metrics import accuracy_score, f1_score


# ===============================
# 参数定义
# ===============================
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2017', type=str)
    parser.add_argument('--data_dir', default='/data/lzy1211/code/pretrain/data/eecl_data', type=str,
                        help='ECCL 数据根目录，内部组织为 {dataset}/eccl_train.json 等')
    parser.add_argument('--img_dir', default='/data/lzy1211/code/twitterImage', type=str)
    parser.add_argument('--output_dir', default='./eccl_outputs', type=str)

    parser.add_argument('--model_path', default='/data/lzy1211/code/model/flan-t5-base/', type=str,
                        help='Flan-T5 路径')
    parser.add_argument('--blip_path', default='/data/lzy1211/code/model/instructblip-flan-t5-xl', type=str,
                        help='InstructBLIP 模型路径（含 Vision + Q-Former）')

    parser.add_argument('--BATCH_SIZE', default=8, type=int)
    parser.add_argument('--LEARNING_RATE', default=5e-5, type=float)
    parser.add_argument('--EPOCHS', default=10, type=int)
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='若>0，则以step为主，epoch作为上限')
    parser.add_argument('--warmup_ratio', default=0.06, type=float,
                        help='warmup 步数比例 (例如 0.06)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--run_name', default='ECCL_pretrain', type=str)

    parser.add_argument('--use_qformer_online', action='store_true', default=True,
                        help='ECCL 必须使用在线 Q-Former 特征')
    parser.add_argument('--freeze_vision', action='store_true', default=True)
    parser.add_argument('--freeze_qformer', action='store_true', default=False)
    return parser.parse_args()


# ===============================
# 通用函数
# ===============================
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_eccl(model, val_loader, device, logger):
    """
    在验证集上评估 ECCL 对比预训练任务的效果（二分类：情感一致 vs 不一致）
    返回: acc, macro-F1
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(val_loader, desc="Evaluating ECCL"):
        images, texts, labels = batch
        labels = labels.to(device)

        outputs = model(
            images=images,
            text=texts,
            labels=None,       # 评估时不算 loss
            task_type='eccl'
        )
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels).astype(int)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    logger.info(f"📊 ECCL Eval — Acc={acc:.4f}, Macro-F1={f1:.4f}")
    return acc, f1


def train_eccl(args, model, train_loader, val_loader, optimizer, scheduler, logger, device):
    logger.info("===== ECCL Contrastive Pretraining =====")
    best_f1 = 0.0
    global_step = 0

    for epoch in trange(args.EPOCHS, desc="ECCL Epoch"):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            images, texts, labels = batch
            labels = labels.to(device)

            outputs = model(
                images=images,
                text=texts,
                labels=labels,
                task_type='eccl'
            )
            loss = outputs["loss"]
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            if 0 < args.max_steps <= global_step:
                break

        avg_loss = total_loss / len(train_loader)
        logger.info(f"[Epoch {epoch}] ECCL Loss: {avg_loss:.4f}")

        # ---------- Validation ----------
        acc, f1 = evaluate_eccl(model, val_loader, device, logger)

        # ---------- Save best ----------
        if f1 >= best_f1:
            best_f1 = f1
            save_path = os.path.join(args.output_dir, "eccl_pretrained.bin")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ New best ECCL model saved (F1={f1:.4f})")

        if 0 < args.max_steps <= global_step:
            break

    logger.info(f"Best ECCL Macro-F1: {best_f1:.4f}")
    return best_f1


# ===============================
# 主程序
# ===============================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "eccl_pretrain.log")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=log_path
    )
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)

    logger.info(args)

    # ===== Tokenizer =====
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    special_tokens_dict = {'additional_special_tokens': ['<asp>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # ===== Model =====
    model = MyFlanT5_ECCL(
        model_path=args.model_path,
        blip_path=args.blip_path,
        tokenizer=tokenizer,
        task_type="eccl",
        use_qformer_online=args.use_qformer_online,
        freeze_vision=args.freeze_vision,
        freeze_qformer=args.freeze_qformer
    ).to(device)

    # ===== Data =====
    train_path = os.path.join(args.data_dir, args.dataset, "eccl_train.json")
    val_path = os.path.join(args.data_dir, args.dataset, "eccl_val.json")

    train_dataset = ECCLDataset(train_path, dataset=args.dataset, img_dir=args.img_dir)
    val_dataset = ECCLDataset(val_path, dataset=args.dataset, img_dir=args.img_dir)

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        collate_fn=eccl_collate_fn
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        collate_fn=eccl_collate_fn
    )

    # ===== Optimizer + Warmup Scheduler =====
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.LEARNING_RATE,
        betas=(0.9, 0.999)
    )

    # 计算训练总步数
    t_total = len(train_loader) * args.EPOCHS if args.max_steps <= 0 else args.max_steps
    warmup_steps = int(args.warmup_ratio * t_total)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total
    )

    # ===== Start ECCL Pretraining =====
    train_eccl(args, model, train_loader, val_loader, optimizer, scheduler, logger, device)


if __name__ == "__main__":
    args = get_parser()
    main(args)
