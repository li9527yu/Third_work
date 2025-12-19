import json
import os
from collections import Counter
import matplotlib.pyplot as plt

def analyze_relation_distribution(train_path, dev_path):
    def load_json(path):
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def analyze_dataset(data, name="dataset"):
        relation_labels = [int(item.get("relation_s", 0)) for item in data]
        # 原始四类统计
        counter_raw = Counter(relation_labels)
        # 二类（相关/无关）统计
        relation_binary = ["related" if r in [1, 2, 3] else "unrelated" for r in relation_labels]
        counter_binary = Counter(relation_binary)

        total = sum(counter_binary.values())
        related_ratio = counter_binary["related"] / total if total else 0
        unrelated_ratio = counter_binary["unrelated"] / total if total else 0

        print(f"\n===== {name.upper()} Relation Label Statistics =====")
        print(f"Raw relation_s counts: {dict(counter_raw)}")
        print(f"Binary related/unrelated: {dict(counter_binary)}")
        print(f"Total samples: {total}")
        print(f"Related ratio: {related_ratio:.3f}, Unrelated ratio: {unrelated_ratio:.3f}")

        return counter_binary, related_ratio, unrelated_ratio

    # 读取数据
    train_data = load_json(train_path)
    dev_data = load_json(dev_path)

    # 分析分布
    train_counter, train_rel, train_unrel = analyze_dataset(train_data, "train")
    dev_counter, dev_rel, dev_unrel = analyze_dataset(dev_data, "dev")

    # 汇总打印
    print("\n===== SUMMARY =====")
    print(f"Train related ratio: {train_rel:.3f}, Dev related ratio: {dev_rel:.3f}")
    print(f"Train-Unrelated:Related = {train_counter['unrelated']}:{train_counter['related']}")
    print(f"Dev-Unrelated:Related = {dev_counter['unrelated']}:{dev_counter['related']}")

    # 画图
    plt.figure(figsize=(6, 4))
    plt.bar(["Train-Unrelated", "Train-Related", "Dev-Unrelated", "Dev-Related"],
            [train_counter["unrelated"], train_counter["related"], dev_counter["unrelated"], dev_counter["related"]],
            color=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"])
    plt.title("Relation_s Label Distribution (Binary)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# === 使用示例 ===
train_path = "/data/lzy1211/code/pretrain/data/AECR/train.json"
dev_path   = "/data/lzy1211/code/pretrain/data/AECR/dev.json"
analyze_relation_distribution(train_path, dev_path)
