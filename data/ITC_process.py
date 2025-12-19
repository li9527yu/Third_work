import json
import random
from tqdm import tqdm


# -----------------------------
# 情感方向映射
# -----------------------------
def emo_to_dir(e):
    e = str(e).lower()
    if "pos" in e:
        return 1
    if "neg" in e:
        return -1
    return 0   # neutral / unknown


# =============================
# Step 1: 单样本正负标签（基础 ECCL）
# =============================
def base_eccl_label(text_dir, img_dir, relation_s):
    """
    relation_s: 1/2/3 → aspect-related, 0 → unrelated
    """

    # ---- C：aspect-unrelated = 强负样本 ----
    if relation_s == 0:
        return 0

    # ---- A：情绪冲突：唯一真正的冲突负样本 ----
    if text_dir * img_dir == -1:
        return 0

    # ---- B：文本 neutral + 图像 emotional（负样本来源 B）----
    if text_dir == 0 and img_dir != 0:
        return 0

    # ---- 情感接近 / 图像增强 / 图像来源 → 正样本 ----
    return 1


# =============================
# Step 2: 跨样本情绪冲突负样本（负样本来源 A）
# =============================
def build_cross_negative(all_data):
    """
    从不同样本强行拼接出冲突图文，使其成为 hard negative。
    """
    neg_pairs = []
    emo_groups = {1: [], -1: [], 0: []}

    # 先按文本情绪分组
    for item in all_data:
        text_dir = emo_to_dir(item["text_emotion"])
        emo_groups[text_dir].append(item)

    for item in all_data:
        t_dir = emo_to_dir(item["text_emotion"])
        # 冲突情绪方向样本
        opposite_dir = -t_dir

        # neutral 文本不做 cross conflict（neutral 不存在“反方向”）
        if t_dir == 0:
            continue

        candidates = emo_groups.get(opposite_dir, [])
        if len(candidates) == 0:
            continue

        neg_target = random.choice(candidates)

        neg_pairs.append({
            "text": item["text"],
            "image": neg_target["image"],
            "aspect": item["aspect"],
            "relation_s": 1,   # 强制设为 related（为了形成纯冲突负样本）
            "text_emotion": item["text_emotion"],
            "image_emotion": neg_target["image_emotion"],
            "eccl_label": 0,
            "type": "cross_conflict"
        })

    return neg_pairs


# =============================
# Step 3: 整体 ECCL 构造
# =============================
def build_eccl_dataset(data_path, save_path):
    dataset = json.load(open(data_path, "r", encoding="utf-8"))
    eccl_data = []

    # ---- 先构造基础 ECCL 样本 ----
    for item in tqdm(dataset, desc=f"Base ECCL from {data_path}"):

        text_em = item["text_emotion"]
        img_em = item["image_emotion"]
        relation_s = item["relation_s"]

        text_dir = emo_to_dir(text_em)
        img_dir = emo_to_dir(img_em)

        base_label = base_eccl_label(text_dir, img_dir, relation_s)

        eccl_data.append({
            "text": item["text"],
            "aspect": item["aspect"],
            "image": item["image"],
            "relation_s": relation_s,
            "text_emotion": text_em,
            "image_emotion": img_em,
            "eccl_label": base_label,
            "type": "base"
        })

    # ---- 加入跨样本冲突负样本（来源 A）----
    cross_negs = build_cross_negative(eccl_data)
    print(f"[Cross-negatives] Added {len(cross_negs)} samples")
    eccl_data.extend(cross_negs)

    # ---- 保存 ----
    json.dump(eccl_data, open(save_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved {len(eccl_data)} ECCL samples to: {save_path}")


# =============================
# 执行（twitter2015 + twitter2017 + train/val/test）
# =============================
datasets = ["twitter2015", "twitter2017"]
splits = ["train", "val", "test"]

for d in datasets:
    for s in splits:
        build_eccl_dataset(
                    data_path=f"/data/lzy1211/code/pretrain/data/emotion_gemini/{d}/{s}.json",
                    save_path=f"/data/lzy1211/code/pretrain/data/eecl_data/{d}/eccl_{s}.json"
                )

