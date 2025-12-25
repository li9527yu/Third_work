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
 
    def forward(self,input_ids,input_multi_ids,\
                attention_mask,input_multi_attention_mask,\
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
        loss_text = outputs.loss

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
        loss_mm = multi_outputs.loss

        
        # # ===== 返回总 loss 
        # weight=0.5
        
        total_loss = loss_mm + 0.5 * loss_text

        return total_loss, loss_mm, loss_text

    
    
    
    def generate(self, input_ids, input_multi_ids, attention_mask, input_multi_attention_mask, 
                 input_hidden_states, weight,relation, max_new_tokens=128):
        
        # 1. 准备文本分支输入嵌入
        inputs_embeds = self.get_embeds(input_ids)

        # 2. 准备多模态分支输入嵌入 (投影图像特征 + 拼接文本)
        imagetext_query = self.imagetext_projection(input_hidden_states)
        query_attention_mask = torch.ones(
            imagetext_query.size()[:-1], dtype=torch.long, device=imagetext_query.device
        )
        
        multi_inputs_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([imagetext_query, multi_inputs_embeds.to(imagetext_query.device)], dim=1)
        multi_attention_mask = torch.cat(
            [query_attention_mask, input_multi_attention_mask.to(query_attention_mask.device)], dim=1
        )

        # 3. 分别调用 generate
        # 注意：这里我们让两路独立生成，因为它们的 Label 长度/结构已经不同了
        # 文本分支预测短标签，多模态分支预测“理由+标签”
        
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=16, # 文本分支很短
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

        multi_outputs = self.language_model.generate(
            inputs_embeds=multi_inputs_embeds,
            attention_mask=multi_attention_mask,
            max_new_tokens=max_new_tokens, # 图像分支较长，需要更多 token
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

        # 4. 解码结果
        # 文本分支结果 (用于参考)
        text_predicted = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # 多模态分支结果 (包含推理过程)
        mm_predicted = self.tokenizer.batch_decode(
            multi_outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # 5. 融合策略逻辑
        # 既然两路任务不同，直接做 Logit 融合在数学上会因为长度不一而变得非常复杂。
        # 在多任务架构下，通常直接取主任务（多模态分支）的结果，因为它已经通过辅助任务被强化了。
        # 如果你一定要融合，建议提取 mm_predicted 中的情感词，与 text_predicted 做投票。
        
        return mm_predicted, text_predicted
    
  