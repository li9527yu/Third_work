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
        # 两路模型融合后特征的映射
        self.fused_projection = nn.Linear(768, 768)
        # self.selector=nn.Linear(768, 2)
        self.init_linear_weight()

    def init_linear_weight(self):
        # # 初始化
        nn.init.normal_(self.language_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.language_projection.bias)

        # nn.init.normal_(self.fused_projection.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.fused_projection.bias)
        # nn.init.normal_(self.feat_linear.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.feat_linear.bias)

 
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
    def mean_pooling(self, hidden_states, attention_mask):
        """
        hidden_states: [B, seq_len, H]
        attention_mask: [B, seq_len]
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        sum_hidden = (hidden_states * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        return sum_hidden / sum_mask  # [B, H]

    def compute_fusion_weight(self,relation_tensor):
        """
        relation_tensor: Tensor, shape [bs, 1] or [bs], int values in {0,1,2,3}
        return: fusion_weight tensor, shape [bs, 1], float values in {0.0, 1.0}
        """
        if relation_tensor.dim() == 2:
            relation_tensor = relation_tensor.squeeze(1)

        fusion_weight = torch.zeros_like(relation_tensor, dtype=torch.float)
        fusion_weight[relation_tensor != 0] = 0.5
        return fusion_weight.unsqueeze(1)  # shape [B,1]


    def forward(self,input_ids,input_multi_ids,attention_mask,input_multi_attention_mask,input_hidden_states,relation,weight,decoder_input_ids=None, decoder_attention_mask=None, labels=None):

       
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
 
        fused_repr = self.mean_pooling(outputs.encoder_last_hidden_state, attention_mask) 
        
         # 多模态的instruction
        multi_inputs_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([query, multi_inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
        multi_attention_mask = torch.cat(
            [query_attention_mask, input_multi_attention_mask.to(query_attention_mask.device)], dim=1
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
        mm_repr = self.mean_pooling(multi_outputs.encoder_last_hidden_state, multi_attention_mask) 


        # # 将两个表示进行融合
        # if fusion_weight is None:
        #     fusion_weight = torch.ones(fused_repr.size(0), 1).to(fused_repr.device)  # 默认全为1
        # elif fusion_weight.dim() == 1:
        #     fusion_weight = fusion_weight.unsqueeze(1)  # [B, 1]

        # fused_final = (1 - fusion_weight) * fused_repr + fusion_weight * mm_repr  # [B, H]

        # fusion_weight = self.compute_fusion_weight(relation)  # shape [bs, 1]
        # fused_final = (1 - fusion_weight) * fused_repr + fusion_weight * mm_repr
        
        fused_final =   fused_repr + mm_repr  # [B, H]
        final_query = self.fused_projection(fused_final).unsqueeze(1)  # [B, 1, H]
        final_inputs_embeds = torch.cat([final_query, relation_inputs_embeds.to(query.device)], dim=1)
        final_attention_mask = torch.cat(
            [torch.ones((query.size(0), 1), dtype=torch.long, device=query.device), attention_mask], dim=1
        )

        final_outputs = self.language_model(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True
        )

        # ===== 返回总 loss（可选加 consistency loss）
        total_loss = final_outputs.loss + 0.5 *outputs.loss + 0.5*multi_outputs.loss
        # total_loss = final_outputs.loss + outputs.loss + multi_outputs.loss
        return total_loss
    
    
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
        outputs = self.language_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        fused_repr = self.mean_pooling(outputs.last_hidden_state, attention_mask)  
        # outputs = self.language_model(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     output_hidden_states=True,
        #     return_dict=True
        # )
 
        # fused_repr = self.mean_pooling(outputs.encoder_last_hidden_state, attention_mask) 
        

        # 多模态的instruction # 获取预测结果
        multi_inputs_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([query, multi_inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
        multi_attention_mask = torch.cat(
            [query_attention_mask, input_multi_attention_mask.to(query_attention_mask.device)], dim=1
        )
        multi_outputs = self.language_model.encoder(
            inputs_embeds=multi_inputs_embeds,
            attention_mask=multi_attention_mask,
            return_dict=True
        )
        mm_repr = self.mean_pooling(multi_outputs.last_hidden_state, multi_attention_mask)


        # multi_outputs = self.language_model(
        #     inputs_embeds=multi_inputs_embeds,
        #     attention_mask=multi_attention_mask,
        #     output_hidden_states=True,
        #     return_dict=True
        # )
        # mm_repr = self.mean_pooling(multi_outputs.encoder_last_hidden_state, multi_attention_mask)
          
        
        fused_final =   fused_repr + mm_repr  # [B, H]
        final_query = self.fused_projection(fused_final).unsqueeze(1)  # [B, 1, H]
        final_inputs_embeds = torch.cat([final_query, relation_inputs_embeds.to(query.device)], dim=1)
        final_attention_mask = torch.cat(
            [torch.ones((query.size(0), 1), dtype=torch.long, device=query.device), attention_mask], dim=1
        )
        final_outputs = self.language_model.generate(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            max_new_tokens=16,
            do_sample=False
        )

        predicted_labels = self.tokenizer.batch_decode(
            final_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        # /,multi_predicted_labels
        return predicted_labels
        
