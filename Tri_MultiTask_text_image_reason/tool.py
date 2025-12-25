import json
import numpy as np
# pip install accelerate
from sklearn.metrics import f1_score



def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)

sentiment_map = {'0': 'neutral', '1': 'positive', '2': 'negative'}
def parse_sequences(pred_sequences):
    senti_preds,srel_preds,erel_preds= [],[],[]
    for seq in pred_sequences:
        seq = seq.lower().replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
        sentiment=seq.split('.')[0]
        if 'negative' in sentiment:
            pred = 2
        elif 'positive' in sentiment:
            pred = 1
        else:
            pred = 0
        
        senti_preds.append(pred)


    return np.array(senti_preds)

def parse_mm_sequences(pred_sequences):
    """
    解析长序列中的情感标签。
    格式预期："... Therefore, the sentiment is positive."
    """
    senti_preds = []
    
    # 建立一个标准化的映射
    # 注意：这里要确保 pred 的数字对应关系与你评估脚本中的 sentiment_map 一致
    # 0: neutral, 1: positive, 2: negative
    
    for seq in pred_sequences:
        # 1. 预处理：转小写并去除特殊字符
        seq = seq.lower().strip()
        
        # 2. 策略 A：查找特定的连接词 "the sentiment is" 之后的内容
        anchor = "the sentiment is"
        if anchor in seq:
            # 获取锚点之后的所有内容
            sentiment_part = seq.split(anchor)[-1]
        else:
            # 如果没找到锚点（模型可能没按格式生成），则扫描整句
            sentiment_part = seq

        # 3. 关键词匹配
        # 优先级：先判断明确的极性，最后默认为中性
        if 'negative' in sentiment_part:
            pred = 2
        elif 'positive' in sentiment_part:
            pred = 1
        else:
            pred = 0 # 默认中性
        
        senti_preds.append(pred)

    return np.array(senti_preds)


# 获取相关性标签
def get_rel_data(dataset):
    path1=f'/data/lzy1211/code/A2II/instructBLIP/two_relation_inference_output/{dataset}/test_result.json'
    path2=f'/data/lzy1211/code/A2II/instructBLIP/two_relation_inference_output/{dataset}/train_result.json'

    
    with open(path1,'r') as f:
        rel1=json.load(f)
    f.close()
    with open(path2,'r') as f:
        rel2=json.load(f)
    f.close()

    rel=rel1+rel2
    return rel


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x