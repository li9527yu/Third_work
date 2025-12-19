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
        self.language_projection = nn.Linear(768, 768)
        self.imagetext_projection = nn.Linear(768, 768)
        self.subjectivity_head = nn.Linear(768, 2)
        self.init_linear_weight()

    def init_linear_weight(self):
        # # 初始化
        nn.init.normal_(self.language_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.language_projection.bias)

        nn.init.normal_(self.imagetext_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.imagetext_projection.bias)

        nn.init.normal_(self.subjectivity_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.subjectivity_head.bias)

 
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
                input_hidden_states,relation,subject_label,weight,decoder_input_ids=None, decoder_attention_mask=None, labels=None):

        # outputs是 文本情感线索  multi是图像情感线索
        query=self.language_projection(input_hidden_states)
        # query=input_hidden_state
        query_attention_mask = torch.ones(
            query.size()[:-1], dtype=torch.long, device=query.device
        )
        relation_inputs_embeds = self.get_embeds(input_ids)
        inputs_embeds = torch.cat([query, relation_inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
        attention_mask = torch.cat(
            [query_attention_mask, attention_mask.to(query_attention_mask.device)], dim=1
        )
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )

        # 
        encoder_feat = outputs.encoder_last_hidden_state
        text_global_feat = torch.mean(encoder_feat, dim=1)

        subj_logits = self.subjectivity_head(text_global_feat)
        loss_fct_aux = nn.CrossEntropyLoss()
        loss_subj = loss_fct_aux(subj_logits, subject_label)
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
 
        # # ===== 返回总 loss（可选加 consistency loss）
        # total_loss =  outputs.loss +  multi_outputs.loss
        # weight=0.5
        fused_logits = weight* outputs.logits + (1-weight) * multi_outputs.logits
        

        total_loss,main_loss = None,None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            main_loss = loss_fct(
                fused_logits.view(-1, fused_logits.size(-1)),
                labels.view(-1)
            )
            total_loss = main_loss+0.5*loss_subj
        return total_loss,main_loss,loss_subj
    
    
    

    def generate(self,input_ids,input_multi_ids,attention_mask,input_multi_attention_mask,input_hidden_states,relation,weight,decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        
        # 获取预测结果
       
        query=self.language_projection(input_hidden_states)
        # query=input_hidden_state
        query_attention_mask = torch.ones(
            query.size()[:-1], dtype=torch.long, device=query.device
        )
        relation_inputs_embeds = self.get_embeds(input_ids)
        inputs_embeds = torch.cat([query, relation_inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
        attention_mask = torch.cat(
            [query_attention_mask, attention_mask.to(query_attention_mask.device)], dim=1
        )
       
        # 多模态的instruction # 获取预测结果
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
 
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False ,
            output_scores=True,
            return_dict_in_generate=True)
        multi_outputs = self.language_model.generate(
            inputs_embeds=multi_inputs_embeds,
            attention_mask=multi_attention_mask,
            max_new_tokens=16,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )
        # weight=0.5
        fused_token_ids = []
        for step_logits_single, step_logits_multi in zip(outputs.scores, multi_outputs.scores):
            fused_logits =  weight* step_logits_single + (1 - weight) * step_logits_multi
            fused_token_ids.append(fused_logits.argmax(dim=-1))  # [batch_size]

        # [batch, seq_len]
        fused_token_ids = torch.stack(fused_token_ids, dim=1)

        # === 解码 ===
        predicted_labels = self.tokenizer.batch_decode(
            fused_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # 获取 subjectivity 预测
        # 从 text_global_feat 计算 subjectivity logits
        # text_global_feat = torch.mean(outputs.encoder_last_hidden_state, dim=1)  # 聚合特征
        # subj_logits = self.subjectivity_head(text_global_feat)
        # predicted_subject_label = torch.argmax(subj_logits, dim=-1)  # 获取预测的 subject_label
        return predicted_labels
        