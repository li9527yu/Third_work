import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ===============================
# 1. 测试阶段评估与记录保存
# ===============================
def evaluate(model, test_dataloader, logger, save_path="test_results.csv"):
    model.eval()
    pred_sequence, senti_labels, meta_records = [], [], []

    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids, input_attention_mask, input_hidden_states, \
        input_pooler_outputs, labels, senti_label, relation_label, raw_meta = post_dataloader(batch)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                input_hidden_states=input_hidden_states,
                labels=labels,
                relation=relation_label,
            )

        preds = parse_sequences(outputs)
        pred_sequence.extend(preds)
        senti_labels.extend(senti_label.detach().cpu().numpy())

        for i, meta in enumerate(raw_meta):
            tclue = meta.get("textual_clues_parsed", {})
            iclue = meta.get("img_clue_parsed", {})
            meta_records.append({
                "text": meta.get("text", ""),
                "aspect": meta.get("aspect", ""),
                "relation": meta.get("relation", ""),
                "gold": int(senti_label[i].item()),
                "pred": preds[i],
                "text_polarity": tclue.get("polarity", None),
                "img_polarity": iclue.get("polarity", None),
                "text_conf": tclue.get("confidence", None),
                "img_conf": iclue.get("confidence", None),
            })

    df = pd.DataFrame(meta_records)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved detailed predictions to {save_path}")
    return df


# ===============================
# 2. 混淆矩阵绘制
# ===============================
def plot_confusion_matrix(y_true, y_pred, labels=(0, 1, 2), label_names=( "neutral","positive", "negative"), save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title("Normalized Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ===============================
# 3. 置信度与准确率曲线
# ===============================
def plot_confidence_curve(df, conf_col="text_conf", save_path=None):
    df["correct"] = (df["gold"] == df["pred"])
    bins = np.linspace(0, 1, 11)
    df["conf_bin"] = pd.cut(df[conf_col], bins)
    conf_acc = df.groupby("conf_bin")["correct"].mean()

    plt.figure(figsize=(6, 4))
    conf_acc.plot(marker="o")
    plt.title(f"{conf_col} vs Accuracy")
    plt.xlabel("Confidence Bin")
    plt.ylabel("Accuracy")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return conf_acc


# ===============================
# 4. 错误分析主函数
# ===============================
def analyze_errors(result_path, save_dir="analysis_output"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(result_path)

    # ---------- 基础映射 ----------
    # label_map = {0: "neutral", 1: "positive", 2: "negative"}
    # df["gold"] = df["gold"].map(label_map)
    # df["pred"] = df["pred"] 

    # ---------- 准确性与错误标记 ----------
    df["correct"] = df["gold"] == df["pred"]

    # ---------- 整体准确率 ----------
    acc = df["correct"].mean()
    print(f"Overall accuracy: {acc:.4f} ({df['correct'].sum()}/{len(df)})")

    # ---------- 混淆矩阵 ----------
    plot_confusion_matrix(df["gold"], df["pred"], save_path=os.path.join(save_dir, "confusion_matrix.png"))

    # ---------- 相关性影响 ----------
    if "relation" in df.columns:
        df["relation_type"] = df["relation"].apply(
            lambda x: "relevant" if "relevant" in str(x).lower() else "irrelevant"
        )
        rel_acc = df.groupby("relation_type")["correct"].mean()
        print("\nAccuracy by relation type:")
        print(rel_acc)
        rel_acc.plot(kind="bar", title="Accuracy by Relation Type")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(save_dir, "relation_acc.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # ---------- 文图极性冲突分析 ----------
    if "text_polarity" in df.columns and "img_polarity" in df.columns:
        df["conflict"] = (df["text_polarity"] != df["img_polarity"]).astype(int)
        conflict_acc = df.groupby("conflict")["correct"].mean()
        print("\nAccuracy by text-image polarity conflict:")
        print(conflict_acc)
        sns.barplot(x=conflict_acc.index, y=conflict_acc.values)
        plt.xticks([0, 1], ["No Conflict", "Conflict"])
        plt.title("Accuracy by Text-Image Polarity Conflict")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(save_dir, "conflict_acc.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # ---------- 置信度 vs 准确率 ----------
    for col in ["text_conf", "img_conf"]:
        if col in df.columns and df[col].notnull().any():
            conf_acc = plot_confidence_curve(df, conf_col=col,
                                             save_path=os.path.join(save_dir, f"{col}_curve.png"))
            print(f"\n{col} bins accuracy:")
            print(conf_acc)

    # ---------- 错误类别分布 ----------
    wrong_df = df[~df["correct"]]
    wrong_df["err_type"] = df.apply(lambda x: f"{x['gold']}→{x['pred']}", axis=1)
    err_counts = wrong_df["err_type"].value_counts()
    print("\nTop-10 Error Transitions:")
    print(err_counts.head(10))

    plt.figure(figsize=(8, 4))
    sns.barplot(x=err_counts.index[:10], y=err_counts.values[:10])
    plt.xticks(rotation=45)
    plt.title("Top Error Type Distribution")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, "error_type_distribution.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # ---------- 导出典型错误样例 ----------
    wrong_df.to_csv(os.path.join(save_dir, "wrong_cases.csv"), index=False)
    print(f"\nSaved {len(wrong_df)} wrong samples to {save_dir}/wrong_cases.csv")

    return df, wrong_df


# ===============================
# 示例调用
# ===============================
if __name__ == "__main__":
    # evaluate() 之后已经生成 test_results.csv
    output_dir='/data/lzy1211/code/A2II/instructBLIP/results/A2II-ThreeRel-Gemini/Img_Cules/twitter2015/24'
    result_path = os.path.join(output_dir,"single_result.csv")
    analyze_errors(result_path,output_dir)
