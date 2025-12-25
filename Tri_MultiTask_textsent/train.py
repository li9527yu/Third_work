import logging
import torch
import torch.utils.data as Data
from torch.optim import AdamW
import argparse
import os
import numpy as np
from tqdm import tqdm, trange
import random
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 晚期融合
from CoT_Model_LateFuse import MyFlanT5

from DataProcessor_emotion_clues import MyDataset


from sklearn.metrics import precision_recall_fscore_support
import wandb
from torch import nn
from tool import warmup_linear,parse_sequences,compute_metrics

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2017', type=str, help=', ')
    parser.add_argument('--data_dir', default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data", type=str)
    parser.add_argument('--img_feat_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--output_dir', default="/data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-t", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--EPOCHS', default=2, type=int)
    parser.add_argument('--LEARNING_RATE', default=1e-5, type=float)
    parser.add_argument("--weight",
                    default=0.1,
                    type=float,
                    help="text weight")
    parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%  of training.")
    parser.add_argument("--max_seq_len",
                        default=128,
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
    input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
        input_hidden_states,input_pooler_outputs,\
            labels_text,labels,senti_label,relation_label=batch
      
    input_ids = input_ids.clone().detach().long().to(device)
    input_attention_mask = input_attention_mask.clone().detach().float().to(device)

    input_multi_ids = input_multi_ids.clone().detach().long().to(device)
    input_multi_attention_mask = input_multi_attention_mask.clone().detach().float().to(device)
    
    labels=labels.to(device).long()
    labels_text=labels_text.to(device).long()
    senti_label=senti_label.to(device).long()

    relation_label=relation_label.to(device).long()

    input_pooler_outputs = input_pooler_outputs.clone().detach().to(input_ids.device)
    input_hidden_states = input_hidden_states.clone().detach().to(input_ids.device)
      
    return input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
        input_hidden_states,input_pooler_outputs,labels_text,labels,senti_label,relation_label

 
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

# 验证集计算指标
def evaluate(model,test_dataloader,weight, logger):
    model.eval()
    pred_sequence,senti_labels=[],[]
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
            input_hidden_states,input_pooler_outputs,\
                labels_text,labels,senti_label,relation_label= post_dataloader(batch)
        with torch.no_grad():
            outputs= model.generate(input_ids=input_ids,
                            input_multi_ids=input_multi_ids,
                            attention_mask=input_attention_mask,  
                            input_multi_attention_mask=input_multi_attention_mask,            
                            input_hidden_states    = input_hidden_states,
                            relation=relation_label,
                            weight=weight
                        )
            pred_sequence.extend(outputs)
            senti_labels.extend(senti_label.detach().cpu().numpy())

    senti_preds=parse_sequences(pred_sequence)
    senti_result = compute_metrics(senti_preds, senti_labels)
    result = {     
        f'senti_acc':senti_result['acc'],    
        f'senti_f1':senti_result['f1'],  
   
    }      
    return result



def train(args,model,train_dataset,val_dataset,optimizer_t5,weight,logger):

    max_senti_acc,max_f1 = 0.0,0.0
    best_epoch=-1
    Rel_global_step,global_step = 0,0
    train_number=train_dataset.number
    num_train_steps = int( train_number / args.BATCH_SIZE * args.EPOCHS)
    train_dataloader= Data.DataLoader(dataset=train_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    val_dataloader= Data.DataLoader(dataset=val_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)

    # # train
    logger.info("*************** Running training ***************")
    for train_idx in trange(int(args.EPOCHS), desc="Epoch"):
        logger.info("************************************************** Epoch: "+ str(train_idx) + " *************************************************************")
        logger.info("  Num examples = %d",  train_number) 
        logger.info("  Batch size = %d", args.BATCH_SIZE)
        logger.info("  Num steps = %d", num_train_steps)
        
        ### train
        model.train()
        tot_l,senti_l, senti_text_l = 0.0, 0.0, 0.0
        for step,data in enumerate(tqdm(train_dataloader,desc="Iteration")):
            batch=data
            # Sentiment
            input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
            input_hidden_states,input_pooler_outputs,\
                labels_text,labels,senti_label,relation_label= post_dataloader(batch)
            total_loss, loss_mm, loss_text = model(input_ids=input_ids,
                            input_multi_ids=input_multi_ids,
                            attention_mask=input_attention_mask,  
                            input_multi_attention_mask=input_multi_attention_mask,            
                            input_hidden_states    = input_hidden_states,
                            labels=labels,
                            labels_text=labels_text,
                            relation=relation_label,
                        
                            weight=weight
                        )
 
            total_loss.backward()
            lr_this_step = args.LEARNING_RATE * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
            for param_group in optimizer_t5.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_t5.step()
            optimizer_t5.zero_grad()
            global_step += 1
            tot_l+=total_loss.item()
            senti_l+=loss_mm.item()
            senti_text_l += loss_text.item()

        senti_l=senti_l/global_step
        senti_text_l=senti_text_l/global_step
        tot_l=tot_l/global_step
        logger.info("total_loss:%s",tot_l)
        logger.info("multi_sentiment_loss:%s",senti_l)
        logger.info("text_sentiment_loss:%s",senti_text_l)

        # Dev evaluation
        model.eval()
        logger.info("***** Running evaluation on Dev Set*****")
        logger.info("  SA Num examples = %d", val_dataset.number) #len(eval_examples)
        logger.info("  Batch size = %d", args.BATCH_SIZE)
        dev_result=evaluate(model,val_dataloader,weight, logger)
        logger.info("***** Dev Eval results *****")
        for key in sorted(dev_result.keys()):
            logger.info("  %s = %s", key, str(dev_result[key]))
 
        # save model
        if dev_result['senti_f1'] >= max_f1:
            max_f1=dev_result['senti_f1']
            best_epoch=train_idx
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  
            torch.save(model_to_save.state_dict(), args.output_model_file)
            logger.info("best_epoch: %d",best_epoch)
            logger.info("dev_max_f1: %d",max_f1)

        # if dev_result['senti_acc'] >= max_senti_acc:
        #     max_senti_acc=dev_result['senti_acc']
        #     best_epoch=train_idx
        #     # Save a trained model
        #     model_to_save = model.module if hasattr(model, 'module') else model  
        #     torch.save(model_to_save.state_dict(), args.output_model_file)
        #     logger.info("best_epoch: %d",best_epoch)

    logger.info("best_epoch: %d",best_epoch)


def main(args):
    # wandb初始化
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="A2II-CoT",
    #     config=args,
    # )
    # args.output_dir=f'{args.output_dir}/A2II-CoT-{args.BATCH_SIZE}-{args.LEARNING_RATE}-{args.EPOCHS}/'

    model_path='/data/lzy1211/code/model/flan-t5-base/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_model_file = os.path.join(args.output_dir, f"{args.output_model_name}.bin")
    output_logger_file=os.path.join(args.output_dir,f'{args.output_log_name}.txt')
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt = '%m/%d/%Y %H:%M:%S',
                level = logging.INFO,
                filename=output_logger_file)
    logger = logging.getLogger(__name__)

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    eos_token_id = tokenizer.eos_token_id  
    # t5_model=T5ForConditionalGeneration.from_pretrained(model_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info(args)
    

    train_data=f"{args.data_dir}/{args.dataset}/train.json"
    train_img_data=f"{args.img_feat_dir}/{args.dataset}/train.pkl"

    val_data=f"{args.data_dir}/{args.dataset}/val.json"
    val_img_data=f"{args.img_feat_dir}/{args.dataset}/val.pkl"

    test_data=f"{args.data_dir}/{args.dataset}/test.json"
    test_img_data=f"{args.img_feat_dir}/{args.dataset}/test.pkl"

    train_dataset=MyDataset(args,train_data,train_img_data,tokenizer)
    val_dataset=MyDataset(args,val_data,val_img_data,tokenizer)
    test_dataset=MyDataset(args,test_data,test_img_data,tokenizer)

    # model
    model = MyFlanT5(model_path=model_path,tokenizer=tokenizer)
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer_t5 = AdamW(optimizer_grouped_parameters,
                            lr=args.LEARNING_RATE)

    # train
    train(args,model,train_dataset,val_dataset,optimizer_t5,args.weight,logger)
    
    # Test
    model.eval()
    logger.info("***** Running evaluation on Test Set*****")
    logger.info("  Num examples = %d", test_dataset.number) #len(eval_examples)
    logger.info("  Batch size = %d", args.BATCH_SIZE)
    test_dataloader= Data.DataLoader(dataset=test_dataset,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0)
    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(args.output_model_file)
    model.load_state_dict(model_state_dict)
    test_results=evaluate(model,test_dataloader,args.weight,logger)
    logger.info("***** Test Eval results *****")
    for key in sorted(test_results.keys()):
        logger.info("  %s = %s", key, str(test_results[key]))
 



if __name__ == "__main__":
 
    args=get_parser()
    main(args)