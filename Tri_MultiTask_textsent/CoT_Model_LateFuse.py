import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaModel, AutoConfig,InstructBlipConfig,InstructBlipQFormerAttention,InstructBlipQFormerEmbeddings
# from modeling_utils import BertSelfEncoder, BertCrossEncoder_AttnMap, BertPooler, BertLayerNorm



class MyFlanT5(nn.Module):
    def __init__(self,model_path,tokenizer,model_name='/public/home/ghfu/lzy/model/instructblip-flan-t5-xl'):
        super().__init__()
        self.language_model=T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer=tokenizer
        # 在输入部分添加 MLP 层
        # 图像特征的映射
        self.imagetext_projection = nn.Linear(768, 768)
        # self.imagetext_projection = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.Linear(768, 768),
        #     nn.Dropout(0.1)
        # )

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.init_linear_weight()

    def init_linear_weight(self):
        # # 初始化
        nn.init.normal_(self.imagetext_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.imagetext_projection.bias)
        # for module in self.imagetext_projection:
        #     # 只有当层是 Linear 层时才进行参数初始化
        #     if isinstance(module, nn.Linear):
        #         nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #         if module.bias is not None:
        #             nn.init.zeros_(module.bias)

 
    def get_embeds(self, input_ids):
        """
        获取输入的嵌入向量
        :param input_ids: 输入的token IDs
        :return: 对应的嵌入向量
        """
        # 调用父类的 get_input_embeddings 方法来获取嵌入层
        input_embeddings = self.language_model.get_input_embeddings()
        
        # 通过嵌入层将 token IDs 转换为嵌入向量
        input_embeds = input_embeddings(input_ids)  # shape: [bs, seq_len, embedding_dim]
        
        return input_embeds
 
    def forward(self,input_ids,input_multi_ids,attention_mask,input_multi_attention_mask,\
                input_hidden_states,relation,weight,labels_text,decoder_input_ids=None, decoder_attention_mask=None, labels=None):


        inputs_embeds = self.get_embeds(input_ids)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels_text, 
            output_hidden_states=True,
            return_dict=True
        )
        logits_text=outputs.logits 

        imagetext_query=self.imagetext_projection(input_hidden_states)
        # query=input_hidden_state
        imagetext_query_attention_mask = torch.ones(
            imagetext_query.size()[:-1], dtype=torch.long, device=imagetext_query.device
        )
         # 多模态的instruction
        multi_inputs_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([imagetext_query, multi_inputs_embeds.to(imagetext_query.device)], dim=1)  # [bs, 32+128,768]
        multi_attention_mask = torch.cat(
            [imagetext_query_attention_mask, input_multi_attention_mask.to(imagetext_query_attention_mask.device)], dim=1
        )
        multi_outputs = self.language_model(
            inputs_embeds=multi_inputs_embeds,
            attention_mask=multi_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
 
        # # ===== 返回总 loss 
        # weight=0.5
        fused_logits = weight* logits_text+ (1-weight) * multi_outputs.logits
        # Auxiliary loss on TEXT branch
        if labels_text is not None:
            loss_text = self.loss_fct(
                logits_text.view(-1, logits_text.size(-1)),
                labels_text.view(-1),
            )

        # Main loss on FUSED logits
        if labels is not None:
            loss_mm = self.loss_fct(
                fused_logits.view(-1, fused_logits.size(-1)),
                labels.view(-1),
            )

        if (loss_text is not None) and (loss_mm is not None):
            total_loss = loss_mm + 0.5 * loss_text
        elif loss_mm is not None:
            total_loss = loss_mm
        elif loss_text is not None:
            total_loss = loss_text
        else:
            total_loss = None

        return total_loss, loss_mm, loss_text

    
    
    def generate(self, input_ids, input_multi_ids, attention_mask, input_multi_attention_mask, input_hidden_states, relation, weight, max_new_tokens=16):
        
        # ---------------- Step 1: 文本分支独立生成 ----------------
        inputs_embeds = self.get_embeds(input_ids)
        outputs_text = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_scores=True,            # 必须开启，为了算分
            return_dict_in_generate=True   # 必须开启，为了拿到 dict
        )

        # ---------------- Step 2: 多模态分支独立生成 ----------------
        # 准备多模态分支的 Embedding (同前)
        imagetext_query = self.imagetext_projection(input_hidden_states)
        imagetext_query_attention_mask = torch.ones(imagetext_query.size()[:-1], dtype=torch.long, device=imagetext_query.device)
        multi_inputs_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([imagetext_query, multi_inputs_embeds.to(imagetext_query.device)], dim=1)
        multi_attention_mask = torch.cat([imagetext_query_attention_mask, input_multi_attention_mask.to(imagetext_query.device)], dim=1)

        outputs_multi = self.language_model.generate(
            inputs_embeds=multi_inputs_embeds,
            attention_mask=multi_attention_mask,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )

        # ---------------- Step 3: 计算置信度并融合 ----------------
        # 辅助函数：计算生成序列的平均对数概率作为置信度
        def compute_confidence(outputs):
            # sequences: [batch_size, seq_len] (包含了起始符)
            sequences = outputs.sequences
            # scores: tuple of len(seq_len-1), each is [batch_size, vocab_size]
            scores = outputs.scores 
            
            batch_size = sequences.shape[0]
            confidences = []
            
            for i in range(batch_size):
                # 取出当前样本生成的 token id (去掉开头的 pad/bos)
                # T5 的 sequence 通常以 0 (pad) 开头，scores 对应的是从第2个词开始
                # 这里的逻辑需要根据 T5 具体行为微调，通常 sequences[:, 1:] 对应 scores
                generated_tokens = sequences[i, 1:] 
                
                token_probs = []
                for step, token_id in enumerate(generated_tokens):
                    if step >= len(scores): break # 防止越界
                    if token_id == self.tokenizer.pad_token_id: break # 遇到 pad 停止
                    if token_id == self.tokenizer.eos_token_id: break # 遇到 eos 停止
                    
                    # 当前步的 Logits -> Softmax -> Prob
                    step_logits = scores[step][i]
                    step_probs = torch.softmax(step_logits, dim=-1)
                    token_prob = step_probs[token_id].item()
                    token_probs.append(token_prob)
                
                # 计算几何平均 (Geometric Mean) 或者 简单的算术平均
                # 几何平均通常更稳健： pow(prod(probs), 1/n)
                if len(token_probs) == 0:
                    confidences.append(0.0)
                else:
                    confidences.append(np.mean(token_probs)) # 这里用简单平均即可
            
            return outputs.sequences, confidences

        # 分别计算
        seqs_text, conf_text = compute_confidence(outputs_text)
        seqs_multi, conf_multi = compute_confidence(outputs_multi)

        final_labels = []
        
        # ---------------- Step 4: 决策比对 ----------------
        batch_size = len(conf_text)
        for i in range(batch_size):
            # 也可以在这里引入 weight 参数： 
            # score_text = conf_text[i] * weight
            # score_multi = conf_multi[i] * (1 - weight)
            
            score_text = conf_text[i]
            score_multi = conf_multi[i]

            if score_text > score_multi:
                chosen_ids = seqs_text[i]
            else:
                chosen_ids = seqs_multi[i]
                
            # 解码单个结果
            decoded = self.tokenizer.decode(chosen_ids, skip_special_tokens=True)
            final_labels.append(decoded)

        return final_labels   
        


    # def generate(self,input_ids,input_multi_ids,attention_mask,input_multi_attention_mask,input_hidden_states,relation,weight,decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        
    #     # 获取预测结果
       
        
    #     inputs_embeds = self.get_embeds(input_ids)

    #     # 多模态的instruction # 获取预测结果
    #     imagetext_query=self.imagetext_projection(input_hidden_states)
    #     # query=input_hidden_state
    #     imagetext_query_attention_mask = torch.ones(
    #         imagetext_query.size()[:-1], dtype=torch.long, device=imagetext_query.device
    #     )
    #      # 多模态的instruction
    #     multi_inputs_embeds = self.get_embeds(input_multi_ids)
    #     multi_inputs_embeds = torch.cat([imagetext_query, multi_inputs_embeds.to(imagetext_query.device)], dim=1)  # [bs, 32+128,768]
    #     multi_attention_mask = torch.cat(
    #         [imagetext_query_attention_mask, input_multi_attention_mask.to(imagetext_query_attention_mask.device)], dim=1
    #     )
 
    #     outputs = self.language_model.generate(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         max_new_tokens=16,
    #         do_sample=False ,
    #         output_scores=True,
    #         return_dict_in_generate=True)
    #     multi_outputs = self.language_model.generate(
    #         inputs_embeds=multi_inputs_embeds,
    #         attention_mask=multi_attention_mask,
    #         max_new_tokens=16,
    #         do_sample=False,
    #         output_scores=True,
    #         return_dict_in_generate=True
    #     )
    #     # weight=0.5
    #     fused_token_ids = []
    #     for step_logits_single, step_logits_multi in zip(outputs.scores, multi_outputs.scores):
    #         fused_logits =weight* step_logits_single +  (1 - weight)  * step_logits_multi
    #         fused_token_ids.append(fused_logits.argmax(dim=-1))  # [batch_size]

    #     # [batch, seq_len]
    #     fused_token_ids = torch.stack(fused_token_ids, dim=1)

    #     # === 解码 ===
    #     predicted_labels = self.tokenizer.batch_decode(
    #         fused_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #     )

    #     # /,multi_predicted_labels
    #     return predicted_labels
        