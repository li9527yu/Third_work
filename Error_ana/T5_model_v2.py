import torch
import torch.nn.functional as F
import torch.nn as nn
# from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaModel, AutoConfig,InstructBlipConfig,InstructBlipQFormerAttention,InstructBlipQFormerEmbeddings
# from modeling_utils import BertSelfEncoder, BertCrossEncoder_AttnMap, BertPooler, BertLayerNorm



class MyFlanT5(nn.Module):
    def __init__(self,model_path,tokenizer,model_name='/public/home/ghfu/lzy/model/instructblip-flan-t5-xl'):
        super().__init__()
        self.language_model=model_path
        # self.language_model.resize_token_embeddings(len(tokenizer))
        self.tokenizer=tokenizer
        # 在输入部分添加 MLP 层
        self.language_projection = nn.Linear(768, 768)
        # self.selector=nn.Linear(768, 2)
        self.init_linear_weight()

    def init_linear_weight(self):
        # # 初始化
        nn.init.normal_(self.language_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.language_projection.bias)

        # nn.init.normal_(self.feat_linear.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.feat_linear.bias)

    # 定义获取嵌入层的函数
    def get_relation_embeds(self,input_ids):

        embeddings = self.word_embeddings(input_ids)
        embeddings = embeddings.to(self.layernorm.weight.dtype)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
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

    def forward(self,input_ids,attention_mask,input_hidden_states,relation,decoder_input_ids=None, decoder_attention_mask=None, labels=None):

        # 处理relation输入（确保是Tensor）
        if not isinstance(relation, torch.Tensor):
            relation = torch.tensor(relation, device=input_hidden_states.device)
        
        # 确保relation形状正确 [batch_size]
        relation = relation.view(-1).to(input_hidden_states.device)
        if relation.size(0) != input_hidden_states.size(0):
            relation = relation.expand(input_hidden_states.size(0))
        
        # 对图像特征按样本进行条件屏蔽
        with torch.no_grad():
            # 创建屏蔽掩码 [bs, 1, 1] 用于特征维度广播
            mask = (relation == 0).view(-1, 1, 1).float()
            
            # 先投影特征（保留计算图）
            projected = self.language_projection(input_hidden_states)
            
            # 对需要屏蔽的样本归零
            query = projected * (1 - mask)  # relation=0的样本完全归零
            
            # 创建注意力掩码
            query_attention_mask = (1 - mask).squeeze(-1).long().expand(-1, query.size(1))  # [bs, seq_len]

        # query=self.language_projection(input_hidden_states)
        # # query=input_hidden_state
        # query_attention_mask = torch.ones(
        #     query.size()[:-1], dtype=torch.long, device=query.device
        # )
        inputs_embeds = self.get_embeds(input_ids)
        # inputs_embeds = torch.cat([query, inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]

        inputs_embeds = torch.cat([inputs_embeds[:, :2, :], query, inputs_embeds[:, 2:, :]], dim=1)
        # [bs,160(32+128),768]

        text_attention_mask = attention_mask.to(query_attention_mask.device)
        attention_mask = torch.cat(
            [text_attention_mask[:, :2], query_attention_mask, text_attention_mask[:, 2:]],
            dim=1
        )
        # attention_mask = torch.cat(
        #     [query_attention_mask, attention_mask.to(query_attention_mask.device)], dim=1
        # )
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self,input_ids,attention_mask,input_hidden_states,relation,decoder_input_ids=None, decoder_attention_mask=None, labels=None):
         # 与forward相同的relation处理逻辑
        if not isinstance(relation, torch.Tensor):
            relation = torch.tensor(relation, device=input_hidden_states.device)
        
        relation = relation.view(-1).to(input_hidden_states.device)
        if relation.size(0) != input_hidden_states.size(0):
            relation = relation.expand(input_hidden_states.size(0))
        
        with torch.no_grad():
            mask = (relation == 0).view(-1, 1, 1).float()
            projected = self.language_projection(input_hidden_states)
            query = projected * (1 - mask)
            query_attention_mask = (1 - mask).squeeze(-1).long().expand(-1, query.size(1))
        
        # query=self.language_projection(input_hidden_states)
        # # query=input_hidden_state
        # query_attention_mask = torch.ones(
        #     query.size()[:-1], dtype=torch.long, device=query.device
        # )
        inputs_embeds = self.get_embeds(input_ids)
        # inputs_embeds = torch.cat([query, inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
        
        inputs_embeds = torch.cat([inputs_embeds[:, :2, :], query, inputs_embeds[:, 2:, :]], dim=1)
        # [bs,160(32+128),768]

        text_attention_mask = attention_mask.to(query_attention_mask.device)
        attention_mask = torch.cat(
            [text_attention_mask[:, :2], query_attention_mask, text_attention_mask[:, 2:]],
            dim=1
        )
        # attention_mask = torch.cat(
        #     [query_attention_mask, attention_mask.to(query_attention_mask.device)], dim=1
        # )
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False 
        )
        predicted_labels=self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # 获取预测结果

        return predicted_labels
        
