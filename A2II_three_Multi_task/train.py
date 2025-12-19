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
from CoT_Model_Soft import MyFlanT5
# from CoT_Model_LateFuse import MyFlanT5

from DataProcessor_emotion_clues import MyDataset
# from DataProcessor_unified import MyDataset

from sklearn.metrics import precision_recall_fscore_support
import wandb
from torch import nn
from tool import warmup_linear,parse_sequences,compute_metrics

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter2017', type=str, help=', ')
    parser.add_argument('--data_dir', default="/data/lzy1211/code/A2II/instructBLIP/CoT_data_process/CoT_data", type=str)
    parser.add_argument('--img_dir', default='/data/lzy1211/code/twitterImage', type=str)
    parser.add_argument('--img_feat_dir', default="/data/lzy1211/code/A2II/instructBLIP/img_data", type=str)
    parser.add_argument('--blip_path', default='/data/lzy1211/code/model/instructblip-flan-t5-xl', type=str,
                        help='InstructBLIP 模型路径（含Flan-T5与Q-Former）')
    parser.add_argument('--output_dir', default="/data/lzy1211/code/A2II/instructBLIP/results/A2II-CoT-t", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--BATCH_SIZE', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--EPOCHS', default=2, type=int)
    parser.add_argument('--LEARNING_RATE', default=1e-5, type=float)
    parser.add_argument("--weight",
                    default=0.5,
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
                aspect,image,labels,senti_label,relation_label=batch
      
    input_ids = input_ids.clone().detach().long().to(device)
    input_attention_mask = input_attention_mask.clone().detach().float().to(device)

    input_multi_ids = input_multi_ids.clone().detach().long().to(device)
    input_multi_attention_mask = input_multi_attention_mask.clone().detach().float().to(device)
    
    labels=labels.to(device).long()
    senti_label=senti_label.to(device).long()  
    relation_label=relation_label.to(device).long()

    aspect=list(aspect)
    image=list(image)
      
    return input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
        aspect,image,labels,senti_label,relation_label

def custom_collate_fn(batch):
    # 1. 分离 PIL.Image 对象和其余的 Tensor/可堆叠对象
    # 假设您的 Dataset 返回的顺序是：
    # (input_ids, input_multi_ids, input_attention_mask, input_multi_attention_mask, aspect, image, labels, output_labels, relation_label)

    # 将批次数据进行转置 (transpose)，以便按类型收集所有样本
    transposed_batch = list(zip(*batch))

    # 图像和 aspect_prompt 是不能被 default_collate 处理的
    images = list(transposed_batch[5]) # 假设 image 在第6位 (索引5)
    aspects = list(transposed_batch[4]) # 假设 aspect 在第5位 (索引4)

    # 剩下的是可以被 default_collate 处理的 Tensor 或数字
    # 我们对剩余的元素使用 default_collate 进行堆叠
    collated_tensors = []
    for i, samples in enumerate(transposed_batch):
        if i not in [4, 5]:  # 排除 aspect (4) 和 image (5)
            # 使用 default_collate 堆叠 Tensors
            collated_tensors.append(torch.utils.data._utils.collate.default_collate(samples))
        # 注意：aspect 和 image 已经保留为 Python list (或 PIL.Image list)

    # collated_tensors 的顺序是: [input_ids, input_multi_ids, input_attention_mask, input_multi_attention_mask, labels, output_labels, relation_label]
    # 加上 aspect, image 后，返回的顺序应该和 Dataset 的输出顺序一致

    # 重新组合所有元素，保持原来的顺序
    final_batch = []
    tensor_idx = 0
    for i in range(len(transposed_batch)):
        if i == 4:
            final_batch.append(aspects) # aspect 仍然是 list of str
        elif i == 5:
            final_batch.append(images) # image 仍然是 list of PIL.Image
        else:
            final_batch.append(collated_tensors[tensor_idx])
            tensor_idx += 1

    return tuple(final_batch)
# 验证集计算指标
def evaluate(model,test_dataloader,weight, logger):
    model.eval()
    pred_sequence,senti_labels=[],[]
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
        aspect,image,labels,senti_label,relation_label= post_dataloader(batch)
        with torch.no_grad():
            outputs= model.generate(input_ids=input_ids,
                            input_multi_ids=input_multi_ids,
                            attention_mask=input_attention_mask,  
                            input_multi_attention_mask=input_multi_attention_mask,            
                            text=aspect,
                            images=image,
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

    max_senti_acc = 0.0
    best_epoch=-1
    Rel_global_step,global_step = 0,0
    train_number=train_dataset.number
    num_train_steps = int( train_number / args.BATCH_SIZE * args.EPOCHS)
    train_dataloader= Data.DataLoader(dataset=train_dataset,shuffle=True,collate_fn=custom_collate_fn, batch_size=args.BATCH_SIZE,num_workers=0)
    val_dataloader= Data.DataLoader(dataset=val_dataset,shuffle=True,collate_fn=custom_collate_fn, batch_size=args.BATCH_SIZE,num_workers=0)

    # # train
    logger.info("*************** Running training ***************")
    for train_idx in trange(int(args.EPOCHS), desc="Epoch"):
        logger.info("************************************************** Epoch: "+ str(train_idx) + " *************************************************************")
        logger.info("  Num examples = %d",  train_number) 
        logger.info("  Batch size = %d", args.BATCH_SIZE)
        logger.info("  Num steps = %d", num_train_steps)
        
        ### train
        model.train()
        senti_l=0
        main_loss_sum = 0.0
        aim_loss_sum  = 0.0
        for step,data in enumerate(tqdm(train_dataloader,desc="Iteration")):
            batch=data
            # Sentiment
            input_ids,input_multi_ids,input_attention_mask,input_multi_attention_mask,\
        aspect,image,labels,senti_label,relation_label= post_dataloader(batch)
            
            output_loss, main_loss,loss_aim,aim_logits= model(input_ids=input_ids,
                            input_multi_ids=input_multi_ids,
                            attention_mask=input_attention_mask,  
                            input_multi_attention_mask=input_multi_attention_mask,            
                            text=aspect,
                            images=image,
                            labels=labels,
                            relation=relation_label,
                            weight=weight
                        )
            # output_loss=outputs['loss']
            output_loss.backward()
            lr_this_step = args.LEARNING_RATE * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
            for param_group in optimizer_t5.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_t5.step()
            optimizer_t5.zero_grad()
            global_step += 1
            senti_l+=output_loss.item()
            main_loss_sum += main_loss.item() if main_loss is not None else 0
            aim_loss_sum  += loss_aim.item() if loss_aim  is not None else 0

        senti_l=senti_l/global_step
        avg_main_loss  = main_loss_sum  / global_step
        avg_aim_loss   = aim_loss_sum   / global_step

        logger.info(f"[Epoch {train_idx}] total_loss = {senti_l:.4f}, "
                f"sentiment_loss = {avg_main_loss:.4f}, aim_loss = {avg_aim_loss:.4f}")
        # wandb.log(
        #     {
        #         "train_loss":senti_l
        #     })
        
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
        if dev_result['senti_acc'] >= max_senti_acc:
            max_senti_acc=dev_result['senti_acc']
            best_epoch=train_idx
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  
            torch.save(model_to_save.state_dict(), args.output_model_file)
            logger.info("best_epoch: %d",best_epoch)

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
    val_data=f"{args.data_dir}/{args.dataset}/val.json"
    test_data=f"{args.data_dir}/{args.dataset}/test.json"


    train_dataset=MyDataset(args,train_data,tokenizer)
    val_dataset=MyDataset(args,val_data,tokenizer)
    test_dataset=MyDataset(args,test_data,tokenizer)

    # model
    model = MyFlanT5(model_path=model_path,
        blip_path=args.blip_path,
        tokenizer=tokenizer)
    
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
    test_dataloader= Data.DataLoader(dataset=test_dataset,shuffle=True,collate_fn=custom_collate_fn, batch_size=args.BATCH_SIZE,num_workers=0)
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