# 情感分类错误分析脚本
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os 

output_dir='/data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-twitter2017/24'
# 1. 读取预测结果
pred_path=os.path.join(output_dir,'pred.json')
with open(pred_path, 'r') as f:
    predictions = json.load(f)

# 2. 提取真实标签和预测标签
true_labels = [item['gold'] for item in predictions]
pred_labels = [item['pred'] for item in predictions]

# 3. 构建混淆矩阵
classes = sorted(set(true_labels + pred_labels))
cm = confusion_matrix(true_labels, pred_labels, labels=classes)

# 4. 混淆矩阵可视化
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 5. 分类报告
report = classification_report(true_labels, pred_labels, target_names=classes)
print('Classification Report:\n')
print(report)

# 6. 详细错误类型分析
errors = []
for item in predictions:
    if item['pred'] != item['gold']:
        errors.append(item)

# 7. 错误统计
error_types = {'neutral_to_negative': 0, 'neutral_to_positive': 0,
               'negative_to_neutral': 0, 'negative_to_positive': 0,
               'positive_to_neutral': 0, 'positive_to_negative': 0}
for e in errors:
    key = f"{e['gold']}_to_{e['pred']}"
    if key in error_types:
        error_types[key] += 1

print('\nError Type Statistics:')
for et, count in error_types.items():
    print(f"{et}: {count}")

# 8. 输出错误样本
with open(os.path.join(output_dir,'error_samples.json'), 'w') as f:
    json.dump(errors, f, ensure_ascii=False, indent=2)
print(f"\nTotal Errors: {len(errors)} (saved to 'error_samples.json')")
