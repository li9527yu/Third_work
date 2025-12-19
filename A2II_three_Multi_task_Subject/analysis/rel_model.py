import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration



class Rel_inference(nn.Module):
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

    def forward(self,rel_inputs_id,rel_inputs_mask,img_feat,rel_label, input_ids=None,input_ir_ids=None,input_hidden_state=None,input_pooler_output=None,\
        attention_mask=None, input_ir_attention_mask=None,\
        decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        # 将input_pooler_output输出进行分类
        select_results=self.selector(input_pooler_output.squeeze(1)).to(rel_inputs_id.device)
        
        return select_results
