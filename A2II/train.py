import logging
import torch
import torch.utils.data as Data
from torch.optim import AdamW
import argparse
import os
from datasets import Dataset
import pickle
import json
import numpy as np
from tqdm import tqdm, trange
import random
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration,DataCollatorForSeq2Seq
from T5_Model import MyFlanT5
from DataProcessor import MyDataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import wandb
# 标签映射
label_map = {"positive": 0, "negative": 1, "neutral": 2}
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2015', type=str, help=', ')
    parser.add_argument('--data_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--img_feat_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--output_dir', default="/data/lzy1211/code/A2II/instructBLIP/results/test1", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--EPOCHS', default=2, type=int)
    parser.add_argument('--LEARNING_RATE', default=5e-5, type=float)
    parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%  of training.")
    parser.add_argument("--max_seq_length",
                        default=150,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--run_name",
                        default="pytorch_model",
                        type=str,
                        help="运行的名称")
    parser.add_argument("--output_model_name",
                        default="pytorch_model",
                        type=str,
                        help="保存权重的名称")
    parser.add_argument("--output_log_name",
                        default="log",
                        type=str,
                        help="日志的名称")  
    opt = parser.parse_args()

    return opt

def post_dataloader(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels=batch    
        
    input_re_ids = input_re_ids.clone().detach().long().to(device)
    input_re_attention_mask = input_re_attention_mask.clone().detach().float().to(device)
    input_ir_ids = input_ir_ids.clone().detach().long().to(device)
    input_ir_attention_mask = input_ir_attention_mask.clone().detach().float().to(device)
    
    labels=labels.to(device).long()

    input_pooler_outputs = input_pooler_outputs.clone().detach().to(input_re_ids.device)
    input_hidden_states = input_hidden_states.clone().detach().to(input_re_ids.device)

            
    return input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels

def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(y_true, y_pred, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro

# 手动裁剪每个序列，直到第一个 </s> 之前的位置
def crop_sequences(ids, eos_token_id):
    cropped_ids = []
    for seq in ids:
        eos_index = (seq == eos_token_id).nonzero(as_tuple=True)[0][0]  # 找到 </s> token 的位置
        if eos_index.numel() > 0:  # 如果找到了 </s> token
            cropped_ids.append(seq[:eos_index.item()])
        else:
            cropped_ids.append(seq)  # 如果没有找到 eos token，返回原始序列
    return torch.stack(cropped_ids)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def evaluate_on_test(model,test_dataloader,tokenizer,eos_token_id, logger):
    model.eval()
    nb_eval_examples = 0
    test_senti_acc=0
    senti_true_label_list = []
    senti_pred_label_list = []
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels= post_dataloader(batch)
        with torch.no_grad():
            outputs= model(input_ids=input_re_ids,
                            attention_mask     = input_re_attention_mask, 
                            input_ir_ids   = input_ir_ids, 
                            input_ir_attention_mask     = input_ir_attention_mask,                 
                            input_hidden_state    = input_hidden_states,
                            input_pooler_output=input_pooler_outputs,
                            labels=labels)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        cropped_predicted_ids = crop_sequences(predicted_ids, eos_token_id)
        cropped_labels = crop_sequences(labels, eos_token_id)
        predicted_labels = tokenizer.batch_decode(cropped_predicted_ids, skip_special_tokens=True)
        true_labels = tokenizer.batch_decode(cropped_labels, skip_special_tokens=True)
        senti_true_label_list.append(true_labels)
        senti_pred_label_list.append(predicted_labels)
        # 计算准确率
        test_senti_acc += sum([1 for pred, true in zip(predicted_labels, true_labels) if pred == true])
        nb_eval_examples += labels.size(0)  

    test_senti_acc=test_senti_acc/nb_eval_examples
    senti_true_label = np.concatenate(senti_true_label_list)
    senti_pred_outputs = np.concatenate(senti_pred_label_list)
    true_labels_numeric = [label_map[label] for label in senti_true_label]
    predictions_numeric = [label_map[label] for label in senti_pred_outputs]
    test_senti_precision, test_senti_recall, test_senti_F_score = macro_f1(true_labels_numeric, predictions_numeric)

    result = {
        'Test_senti_acc':test_senti_acc,
        'Test_senti_precision':test_senti_precision,
        'Test_senti_recall':test_senti_recall,
        'Test_senti_F_score':test_senti_F_score}
    logger.info("***** Test Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    wandb.log(
        {
            "test_acc": test_senti_acc,
            "test_macro_f1": test_senti_F_score,
            'Test_senti_precision':test_senti_precision,
            'Test_senti_recall':test_senti_recall,
        })

def main(args):
    wandb.login()
    # wandb初始化
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="A2II-sweep",
        config=args,
    )
    # args.BATCH_SIZE=wandb.config.BATCH_SIZE
    # args.LEARNING_RATE=wandb.config.LEARNING_RATE
    # args.EPOCHS=wandb.config.EPOCHS
    args.output_dir=f'{args.output_dir}/A2II-sweep-{args.BATCH_SIZE}-{args.LEARNING_RATE}-{args.EPOCHS}/'
    model_path='/data/lzy1211/code/model/flan-t5-base/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_model_file = os.path.join(args.output_dir, f"{args.output_model_name}.bin")
    output_logger_file=os.path.join(args.output_dir,f'{args.output_log_name}.txt')
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = logging.INFO,
                filename=output_logger_file)
    logger = logging.getLogger(__name__)
    logger.info(args)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    eos_token_id = tokenizer.eos_token_id  
    t5_model=T5ForConditionalGeneration.from_pretrained(model_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 处理数据
    train_data=f"{args.data_dir}/{args.dataset}/train.pkl"
    val_data=f"{args.data_dir}/{args.dataset}/val.pkl"
    test_data=f"{args.data_dir}/{args.dataset}/test.pkl"
    train_dataset=MyDataset(train_data,tokenizer, max_seq_len= args.max_seq_length)
    val_dataset=MyDataset(val_data,tokenizer, max_seq_len= args.max_seq_length)
    test_dataset=MyDataset(test_data,tokenizer, max_seq_len= args.max_seq_length)

    train_dataloader= Data.DataLoader(dataset=train_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    val_dataloader= Data.DataLoader(dataset=val_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    test_dataloader= Data.DataLoader(dataset=test_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)

    train_number=train_dataset.number
    num_train_steps = int( train_number / args.BATCH_SIZE * args.EPOCHS)

    model = MyFlanT5(model_path=t5_model)
    model.to(device)

    # Prepare optimizer
    # optimizer BertAdam
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
   
    
    optimizer_t5 = AdamW(optimizer_grouped_parameters,
                            lr=args.LEARNING_RATE)
    max_senti_acc = 0.0
    best_epoch=-1
    global_step = 0
    # train
    logger.info("*************** Running training ***************")
    for train_idx in trange(int(args.EPOCHS), desc="Epoch"):
        logger.info("************************************************** Epoch: "+ str(train_idx) + " *************************************************************")
        logger.info("  Num examples = %d",  train_number) 
        logger.info("  Batch size = %d", args.BATCH_SIZE)
        logger.info("  Num steps = %d", num_train_steps)
        
        ### train
        model.train()
        senti_l=0
        for batch in tqdm(train_dataloader, desc="Iteration"):
            input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels= post_dataloader(batch)
            outputs= model(input_ids=input_re_ids,
                            attention_mask     = input_re_attention_mask, 
                            input_ir_ids   = input_ir_ids, 
                            input_ir_attention_mask     = input_ir_attention_mask,                 
                            input_hidden_state    = input_hidden_states,
                            input_pooler_output=input_pooler_outputs,
                            labels=labels)
            output_loss=outputs['loss']
            output_loss.backward()
            lr_this_step = args.LEARNING_RATE * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
            for param_group in optimizer_t5.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_t5.step()
            optimizer_t5.zero_grad()
            global_step += 1
            senti_l+=output_loss.item()
        
        senti_l=senti_l/global_step
        logger.info("sentiment_loss:%s",senti_l)
        wandb.log(
            {
                "train_loss":senti_l
            })
        # ## dev
        model.eval()
        logger.info("***** Running evaluation on Dev Set*****")
        logger.info("  SA Num examples = %d", val_dataset.number) #len(eval_examples)
        logger.info("  Batch size = %d", args.BATCH_SIZE)


        nb_eval_examples = 0
        senti_acc=0
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            input_re_ids,input_re_attention_mask,input_ir_ids,input_ir_attention_mask,input_hidden_states,input_pooler_outputs,labels= post_dataloader(batch)
            with torch.no_grad():
                outputs= model(input_ids=input_re_ids,
                                attention_mask     = input_re_attention_mask, 
                                input_ir_ids   = input_ir_ids, 
                                input_ir_attention_mask     = input_ir_attention_mask,                 
                                input_hidden_state    = input_hidden_states,
                                input_pooler_output=input_pooler_outputs,
                                labels=labels)
                 # 获取预测结果
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                cropped_predicted_ids = crop_sequences(predicted_ids, eos_token_id)
                cropped_labels = crop_sequences(labels, eos_token_id)
                predicted_labels = tokenizer.batch_decode(cropped_predicted_ids, skip_special_tokens=True)
                true_labels = tokenizer.batch_decode(cropped_labels, skip_special_tokens=True)
                # 计算准确率
                senti_acc += sum([1 for pred, true in zip(predicted_labels, true_labels) if pred == true])
                nb_eval_examples += labels.size(0)

                
        senti_acc=senti_acc/nb_eval_examples
        result = {
            'nb_eval_examples':nb_eval_examples,         
            'Dev_senti_acc':senti_acc,             
            }        
        logger.info("***** Dev Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        wandb.log(
            {
                "val_acc": senti_acc
            })
        # 保存 模型
                # save model
        if senti_acc >= max_senti_acc:
            max_senti_acc=senti_acc
            best_epoch=train_idx
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("best_epoch: %d",best_epoch)

    logger.info("best_epoch: %d",best_epoch)

    model.eval()
    logger.info("***** Running evaluation on Test Set*****")
    logger.info("  Num examples = %d", test_dataset.number) #len(eval_examples)
    logger.info("  Batch size = %d", args.BATCH_SIZE)
    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model.load_state_dict(model_state_dict)
    evaluate_on_test(model,test_dataloader,tokenizer,eos_token_id, logger)
    wandb.finish()



if __name__ == "__main__":
    args=get_parser()
    # sweep_configuration = {
    #     "method": "grid",
    #     "name": "A2II-sweep",
    #     "metric": {"goal": "maximize", "name": "test_senti_F_score"},
    #     "parameters": {
    #         "BATCH_SIZE": {"values": [8,16,32]},
    #         "LEARNING_RATE": {"values": [1e-6,1e-5,2e-5]},
    #         "EPOCHS": {"values": [1,2]}
    #     },
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="A2II-sweep")
    # wandb.agent(sweep_id, function=main(args), count=3)
    args=get_parser()
    main(args)