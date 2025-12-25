import torch
import torch.nn.functional as F
import torch.nn as nn
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

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.init_linear_weight()

    def init_linear_weight(self):
        # # 初始化
        nn.init.normal_(self.imagetext_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.imagetext_projection.bias)


 
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

    
    
    

    def generate(self, input_ids, input_multi_ids, attention_mask, input_multi_attention_mask, input_hidden_states, relation, weight, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        """
        Late Fusion 实现：
        1. 纯文本分支独立生成 -> 得到 答案A 和 置信度ScoreA
        2. 多模态分支独立生成 -> 得到 答案B 和 置信度ScoreB
        3. 比较 (ScoreA * weight) 和 (ScoreB * (1-weight))，选择分数高对应的答案
        """
        
        # ==================== 1. 准备 Embedding (与你之前的逻辑一致) ====================
        
        # --- 纯文本分支输入 ---
        inputs_embeds = self.get_embeds(input_ids)

        # --- 多模态分支输入 ---
        imagetext_query = self.imagetext_projection(input_hidden_states)
        imagetext_query_attention_mask = torch.ones(
            imagetext_query.size()[:-1], dtype=torch.long, device=imagetext_query.device
        )
        
        multi_inputs_embeds_part = self.get_embeds(input_multi_ids)
        # 拼接 Q-Former 输出 和 文本 Embedding
        multi_inputs_embeds = torch.cat([imagetext_query, multi_inputs_embeds_part.to(imagetext_query.device)], dim=1)
        multi_attention_mask = torch.cat(
            [imagetext_query_attention_mask, input_multi_attention_mask.to(imagetext_query_attention_mask.device)], dim=1
        )

        # ==================== 2. 独立生成 (Independent Generation) ====================

        # --- 分支 A: 纯文本生成 ---
        # 注意：必须设置 output_scores=True 和 return_dict_in_generate=True 才能拿到概率
        outputs_text = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False,       # 贪婪搜索 (Greedy Search)
            output_scores=True,
            return_dict_in_generate=True
        )

        # --- 分支 B: 多模态生成 ---
        outputs_multi = self.language_model.generate(
            inputs_embeds=multi_inputs_embeds,
            attention_mask=multi_attention_mask,
            max_new_tokens=16,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

        # ==================== 3. 计算置信度 (Confidence Calculation) ====================
        
        # 辅助函数：计算生成序列的概率得分
        # compute_transition_scores 会返回每一步生成的 log_softmax 值
        # 我们对这些值求和，然后取 exp，得到整个句子的生成概率 (0~1之间)
        
        # 处理 Text 分支
        transition_scores_text = self.language_model.compute_transition_scores(
            outputs_text.sequences, outputs_text.scores, normalize_logits=True
        )
        # sum(dim=1) 把整个句子的所有 token 的 log 概率加起来
        log_prob_text = torch.sum(transition_scores_text, dim=1) 
        prob_text = torch.exp(log_prob_text) # 转换回 0-1 的概率值

        # 处理 Multi 分支
        transition_scores_multi = self.language_model.compute_transition_scores(
            outputs_multi.sequences, outputs_multi.scores, normalize_logits=True
        )
        log_prob_multi = torch.sum(transition_scores_multi, dim=1)
        prob_multi = torch.exp(log_prob_multi)

        # ==================== 4. 决策级融合 (Late Fusion Decision) ====================

        batch_size = input_ids.shape[0]
        final_token_ids = []

        # 遍历 Batch 中的每一个样本进行比较
        for i in range(batch_size):
            # 获取当前样本的纯文本概率和多模态概率
            p_text = prob_text[i].item()
            p_multi = prob_multi[i].item()
            
            # 加权比较
            score_text_weighted = p_text * weight
            score_multi_weighted = p_multi * (1 - weight)
            
            # 谁的分数高，就选用谁生成的 token 序列
            if score_text_weighted > score_multi_weighted:
                final_token_ids.append(outputs_text.sequences[i])
            else:
                final_token_ids.append(outputs_multi.sequences[i])
        
        # 将 list 堆叠回 tensor
        # 注意：由于两个分支生成的长度可能不一样，直接 stack 可能会报错
        # T5 generate 的输出通常会有 padding，但为了保险，建议这里直接 decode，不再变回 tensor
        
        # ==================== 5. 解码 (Decoding) ====================
        
        # 这里的 final_token_ids 是一个包含 Tensor 的列表，长度不一定一致
        # tokenizer.batch_decode 可以处理 list of tensors
        predicted_labels = self.tokenizer.batch_decode(
            final_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return predicted_labels
        