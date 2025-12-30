import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration

class MyFlanT5(nn.Module):
    def __init__(self,model_path,tokenizer,feature_dim=768,model_name='/public/home/ghfu/lzy/model/instructblip-flan-t5-xl'):
        super().__init__()
        self.language_model=T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer=tokenizer
        # relation
        # 1. 特征压缩与融合
        # 假设 visual 和 text 输入进来时还是序列 [bs, seq_len, dim]，这里先做 Pooling
        self.visual_pooler = nn.AdaptiveAvgPool1d(1) # 或 Attention Pooling
        self.text_pooler = nn.AdaptiveAvgPool1d(1)
        
        # 2. 融合层
        self.joint_projector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim), # 加个 LN 稳定训练
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 3. 分层分类头
        self.semantic_classifier = nn.Linear(feature_dim, 2)      
        self.sentiment_align_classifier = nn.Linear(feature_dim, 2) 

        # 图像特征的映射
        self.visual_shared_adapter = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.imagetext_projection = nn.Linear(768, 768)
        self.relation_projection = nn.Linear(768, 768)
        # self.imagetext_projection = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.Linear(768, 768),
        #     nn.Dropout(0.1)
        # )

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.init_linear_weight()

    def compute_soft_fusion_weight(self, sem_logits, sent_align_logits):
        """
        【关键改进】使用 Softmax 概率计算可微的权重
        逻辑复刻：
        - P(Sem=0) -> 趋向 0.0
        - P(Sem=1) * P(Align=0) -> 趋向 0.3
        - P(Sem=1) * P(Align=1) -> 趋向 0.7
        """
        # 获取概率分布
        probs_sem = F.softmax(sem_logits, dim=-1)        # [bs, 2]
        probs_align = F.softmax(sent_align_logits, dim=-1) # [bs, 2]
        
        p_sem_relevant = probs_sem[:, 1]       # 语义相关的概率 P(Sem=1)
        p_sent_strong = probs_align[:, 1]      # 情感强相关的概率 P(Align=1)
        
        # 设计连续函数模拟你的离散逻辑：
        # Base Weight: 基础权重 0.3 (只要语义相关就有 0.3)
        # Bonus Weight: 额外权重 0.4 (如果情感也强相关，再加 0.4，总共 0.7)
        # Gating: 整个权重受 P(Sem=1) 控制
        
        base_val = 0.3
        bonus_val = 0.4 # 0.3 + 0.4 = 0.7
        
        # 公式：Weight = P(Sem) * [ 0.3 + P(Align) * 0.4 ]
        # 1. 如果 P(Sem) -> 0: Weight -> 0 (符合语义无关为0)
        # 2. 如果 P(Sem) -> 1, P(Align) -> 0: Weight -> 1 * [0.3 + 0] = 0.3 (符合弱相关)
        # 3. 如果 P(Sem) -> 1, P(Align) -> 1: Weight -> 1 * [0.3 + 0.4] = 0.7 (符合强相关)
        
        alpha = p_sem_relevant * (base_val + p_sent_strong * bonus_val)
        
        return alpha.unsqueeze(1) # [bs, 1] 用于广播乘法
    
    def _init_weights(self, module):
        """
        通用的权重初始化函数，用于 apply
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_linear_weight(self):
         # 初始化共享层
        self.visual_shared_adapter.apply(self._init_weights)
        
        # 初始化两个分支头
        self.imagetext_projection.apply(self._init_weights)
        self.relation_projection.apply(self._init_weights)
        
        # 2. 初始化你新加的相关性模块组件
        # joint_projector 是 Sequential，用 apply 最合适
        self.joint_projector.apply(self._init_weights)
        
        # 3. 初始化分类头
        self.semantic_classifier.apply(self._init_weights)
        self.sentiment_align_classifier.apply(self._init_weights)
        
        # 4. 初始化 Pooling 层 (如果有参数的话，虽然 AvgPool 没有参数，但为了统一可以写上，不报错)
        # self.visual_pooler.apply(self._init_weights)
    # def init_linear_weight(self):
    #     # # 初始化
    #     # nn.init.normal_(self.imagetext_projection.weight, mean=0.0, std=0.02)
    #     # nn.init.zeros_(self.imagetext_projection.bias)
    #     for module in self.imagetext_projection:
    #         # 只有当层是 Linear 层时才进行参数初始化
    #         if isinstance(module, nn.Linear):
    #             nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #             if module.bias is not None:
    #                 nn.init.zeros_(module.bias)
    
    def get_relevance_loss(self, sem_logits, sent_align_logits, sem_labels, sent_align_labels):
        loss_fct = nn.CrossEntropyLoss()
        # 1. 语义 Loss (全量计算)
        l_sem = loss_fct(sem_logits, sem_labels)
        # 2. 情感一致性 Loss (仅对语义相关的样本计算，避免噪声)
        mask = (sem_labels == 1)
        if mask.any():
            l_sent = loss_fct(sent_align_logits[mask], sent_align_labels[mask])
        else:
            l_sent = 0.0
            
        return l_sem + l_sent
    
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
    
    def masked_mean_pooling(self, hidden_states, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self,input_ids,input_multi_ids,attention_mask,input_multi_attention_mask,\
                input_hidden_states,relation_semantic,realtion_sentiment,weight,labels_text,decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        # 1. 先过共享底座 (这里实现了知识迁移！)
        # input_hidden_states [bs, 32, 768]
        shared_visual_feat = self.visual_shared_adapter(input_hidden_states)
        
        # 2. 分叉 - 辅助任务分支
        # 用 relation_query 去计算 Loss，梯度回传会更新 shared_visual_feat
        relation_query = self.relation_projection(shared_visual_feat)
        
        # 3. 分叉 - 主任务分支
        # T5 拿到的 imagetext_query 包含了辅助任务带来的语义增益
        imagetext_query = self.imagetext_projection(shared_visual_feat)
        # imagetext_query=self.imagetext_projection(input_hidden_states)
        # relation_query=self.relation_projection(input_hidden_states)
        inputs_embeds = self.get_embeds(input_ids)
        encoder_outputs = self.language_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_feats = encoder_outputs.last_hidden_state # [bs, seq_len, 768]
        
        # relation
        # permute for pooling: [bs, dim, seq_len]
        v_pooled = self.visual_pooler(relation_query.transpose(1, 2)).squeeze(2) # [bs, 768]
        
        t_pooled = self.masked_mean_pooling(text_feats, attention_mask) 
        
        # --- Step 2: Fusion ---
        combined = torch.cat([v_pooled, t_pooled], dim=-1)
        joint_feat = self.joint_projector(combined)
        
        # --- Step 3: Logits ---
        sem_logits = self.semantic_classifier(joint_feat)           # [bs, 2]
        sent_align_logits = self.sentiment_align_classifier(joint_feat) # [bs, 2]
        # 3. 计算辅助 Loss (Relevance Loss)
        if relation_semantic is not None:
            loss_aux = self.get_relevance_loss(sem_logits, sent_align_logits, relation_semantic, realtion_sentiment)
        else:
            loss_aux = 0.0

        outputs = self.language_model(
            encoder_outputs=encoder_outputs, # 传入刚才跑过的 encoder 结果
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels_text,
            return_dict=True
        )
        logits_text = outputs.logits

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

        total_loss = loss_mm + 0.5 * loss_text+0.1*loss_aux

        return total_loss, loss_mm, loss_text,loss_aux

    
    def generate(self,input_ids,input_multi_ids,attention_mask,input_multi_attention_mask,input_hidden_states,weight,decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        

        # 1.1 先过共享底座 (提取通用深度特征)
        # input_hidden_states: [bs, 32, 768]
        shared_visual_feat = self.visual_shared_adapter(input_hidden_states)
        
        # 1.2 主任务分支 (投影为 T5 可理解的特征)
        # 这就是 visual_feats_projected，用于后续拼接
        visual_feats_projected = self.imagetext_projection(shared_visual_feat)
        
        # 1.3 (可选) 辅助任务分支
        # 如果你想在推理时使用动态 alpha，就需要这一步；
        # 如果你推理时只用固定 weight，这一步可以不计算，节省算力。
        # relation_feat = self.relation_projection(shared_visual_feat) 
        text_inputs_embeds = self.get_embeds(input_ids)
        
        # 2. 获取 Text Encoder 特征 (用于计算 Alpha)
        encoder_outputs = self.language_model.encoder(
            inputs_embeds=text_inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_encoder_hidden = encoder_outputs.last_hidden_state
        
 
        outputs = self.language_model.generate(
            encoder_outputs=encoder_outputs, # 传入 encoder outputs
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        # Image Branch Generation
        imagetext_query_attention_mask = torch.ones(
            visual_feats_projected.size()[:-1], dtype=torch.long, device=visual_feats_projected.device
        )
        multi_text_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([visual_feats_projected, multi_text_embeds], dim=1)
        multi_attention_mask = torch.cat(
            [imagetext_query_attention_mask, input_multi_attention_mask.to(visual_feats_projected.device)], dim=1
        )
        
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
            fused_logits =weight* step_logits_single +  (1 - weight)  * step_logits_multi
            fused_token_ids.append(fused_logits.argmax(dim=-1))  # [batch_size]

        # [batch, seq_len]
        fused_token_ids = torch.stack(fused_token_ids, dim=1)

        # === 解码 ===
        predicted_labels = self.tokenizer.batch_decode(
            fused_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # /,multi_predicted_labels
        return predicted_labels
        