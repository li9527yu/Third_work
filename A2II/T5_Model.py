import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration



class MyFlanT5(nn.Module):
    def __init__(self,model_path):
        super().__init__()
        self.language_model=model_path
        # 在输入部分添加 MLP 层
        self.language_projection = nn.Linear(768, 768)
        self.selector=nn.Linear(768, 2)
        self.init_linear_weight()

    def init_linear_weight(self):
        # 初始化
        nn.init.normal_(self.language_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.language_projection.bias)

        # nn.init.xavier_normal_(self.language_projection.weight)
        # nn.init.zeros_(self.language_projection.bias)
        # nn.init.xavier_normal_(self.selector.weight)
        # nn.init.zeros_(self.selector.bias)
    # 定义获取嵌入层的函数
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

    def forward(self, input_ids,input_ir_ids,input_hidden_state,input_pooler_output,attention_mask=None, input_ir_attention_mask=None,\
        decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        # 将input_pooler_output输出进行分类
        select_results=self.selector(input_pooler_output.squeeze(1)).to(input_ids.device)
        probabilities = F.softmax(select_results, dim=-1)
        #    获取每个样本的最大概率类别
        predicted_labels = torch.argmax(probabilities, dim=-1)
        final_input_ids = input_ir_ids * (predicted_labels.unsqueeze(1) == 0).float() + \
                 input_ids * (predicted_labels.unsqueeze(1) == 1).float()
        final_attention_mask = input_ir_attention_mask * (predicted_labels.unsqueeze(1) == 0).float() + \
                 attention_mask * (predicted_labels.unsqueeze(1) == 1).float()
        # 注意类型转化为long
        final_input_ids=final_input_ids.to(torch.long)
        final_attention_mask=final_attention_mask.to(torch.long)
        # 将筛选后的输入和query进行拼接
        query=self.language_projection(input_hidden_state)
        # query=input_hidden_state
        query_attention_mask = torch.ones(
            query.size()[:-1], dtype=torch.long, device=query.device
        )
        inputs_embeds = self.get_embeds(final_input_ids)
        inputs_embeds = torch.cat([query, inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
        attention_mask = torch.cat(
            [query_attention_mask, final_attention_mask.to(query_attention_mask.device)], dim=1
        )
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        return outputs
