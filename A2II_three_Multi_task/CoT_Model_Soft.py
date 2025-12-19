# T5_Model_Combined.py
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from transformers import (
    T5ForConditionalGeneration,
    InstructBlipConfig,
    InstructBlipVisionModel,
    InstructBlipQFormerModel,
    InstructBlipProcessor,
)
from PIL import Image

class MyFlanT5(nn.Module):

    def __init__(
        self,
        model_path,
        blip_path, # 新增：用于加载InstructBLIP配置和权重
        tokenizer,
        use_qformer_online: bool = True, # 确保开启在线计算
        freeze_vision: bool = True,
        freeze_qformer: bool = False, # 默认不冻结Q-Former，以达到训练T5和Q-Former的目的
        num_query_tokens: Optional[int] = None,
        # model_name 参数在此处无用，已移除
    ):
        super().__init__()
        self.use_qformer_online = use_qformer_online
        
        # 假设 model_path 是 T5ForConditionalGeneration 的路径
        self.language_model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = tokenizer
        self.language_model.resize_token_embeddings(len(tokenizer))
        self._lm_name_or_path = blip_path
        
        # ===== 1) Vision + Q-Former (在线特征提取部分) =====
        self.processor = None
        self.vision_model = None
        self.qformer = None
        self.query_tokens = None
        
        if self.use_qformer_online:
            if blip_path is None:
                raise ValueError("When use_qformer_online=True, please pass a string path for blip_path (InstructBLIP checkpoint).")

            blip_config = InstructBlipConfig.from_pretrained(blip_path)
            self.vision_model = InstructBlipVisionModel(blip_config.vision_config)
            self.qformer = InstructBlipQFormerModel(blip_config.qformer_config)
            self.processor = InstructBlipProcessor.from_pretrained(blip_path)

            # learnable query tokens
            n_q = num_query_tokens if num_query_tokens is not None else blip_config.num_query_tokens
            h_q = blip_config.qformer_config.hidden_size
            self.query_tokens = nn.Parameter(torch.zeros(n_q, h_q))

            # 冻结选项
            if freeze_vision:
                for p in self.vision_model.parameters():
                    p.requires_grad = False
            # 默认不冻结Q-Former
            if freeze_qformer:
                for p in self.qformer.parameters():
                    p.requires_grad = False
            
            # 初始化 query_tokens
            nn.init.xavier_uniform_(self.query_tokens.data)


        # ===== 2) Projection Heads (来自第二个代码) =====
        # language_projection 用于主线 (text/relation)
        self.language_projection = nn.Linear(768, 768) 
        # imagetext_projection 用于辅助线 (multi/image-text)
        self.imagetext_projection = nn.Linear(768, 768)
        self.init_linear_weight()

        self.aim_head = nn.Linear(768, 3)
        nn.init.xavier_uniform_(self.aim_head.weight)
        nn.init.zeros_(self.aim_head.bias)

    def init_linear_weight(self):
        # 初始化 projection (使用第二个代码中的初始化方法)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        init_weights(self.language_projection)
        init_weights(self.imagetext_projection)

    
    # helper: LM input embeddings
    def get_embeds(self, input_ids):
        """
        获取输入的嵌入向量
        """
        return self.language_model.get_input_embeddings()(input_ids)
    
    # ====== Online path: compute Q-Former features from raw image + aspect text (来自第一个代码) ======
    def _compute_qformer_feats_online(self, images, aspect_prompt):
        """
        支持批处理版本：
        images: List[PIL.Image] 或单张 PIL.Image
        aspect_prompt: List[str] 或 str
        返回：hidden_state [B, n_q, 768] 和 pooler_output [B, 1, 768]
        """
        device = next(self.language_model.parameters()).device

        # 保证输入格式统一为列表
        if isinstance(images, (Image.Image, torch.Tensor)):
            images = [images]
        if isinstance(aspect_prompt, str):
            aspect_prompt = [aspect_prompt]

        assert len(images) == len(aspect_prompt), \
            f"Image/Text batch mismatch: {len(images)} vs {len(aspect_prompt)}"

        # Processor 自动支持 batch 图像
        inputs = self.processor(
            images=images,
            text=aspect_prompt,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Vision Encoder
        # Vision Model只有在vision_model不被冻结时才会产生梯度
        vision_outputs = self.vision_model(pixel_values=inputs["pixel_values"])
        image_embeds = vision_outputs[0]  # [B, L_img, D]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        # Q-Former
        B = image_embeds.shape[0]
        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
        qformer_attention_mask = torch.cat([query_attention_mask, inputs["qformer_attention_mask"]], dim=1)

        # Q-Former只有在qformer不被冻结时才会产生梯度
        q_out = self.qformer(
            input_ids=inputs["qformer_input_ids"],
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        hidden_state = q_out[0][:, : query_tokens.size(1), :]    # [B, n_q, 768] (Query states)
        # pooler_output = q_out[1].unsqueeze(1) if q_out[1].dim() == 2 else q_out[1]  # [B, 1, 768] (用于兼容第一个代码的relation任务，但本版本未使用)
        return hidden_state #, pooler_output

    def forward(
        self,
        input_ids,             # T5主线指令 Token IDs
        input_multi_ids,       # T5多模态指令 Token IDs
        attention_mask,        # T5主线 Attn Mask
        input_multi_attention_mask, # T5多模态 Attn Mask
        # input_hidden_states, # 移除：现在需要在线计算
        relation,              # 似乎是未使用参数
        weight,                # Logits 融合权重
        images,                # 新增：原始图像
        text,                  # 新增：用于Q-Former的aspect prompt文本
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,           # T5 Seq2Seq 标签
    ):
        device = next(self.language_model.parameters()).device

        # === 0) 在线计算 Q-Former 特征 (来自第一个代码的逻辑) ===
        if not self.use_qformer_online:
            raise ValueError("use_qformer_online=False, 无法执行在线特征提取。")
        
        # hidden_state 即为 Q-Former 的查询向量 [B, n_q, 768]
        input_hidden_states = self._compute_qformer_feats_online(images, text) 
        input_hidden_states = input_hidden_states.to(device) # 确保在正确设备上


        pooled_img_feat = input_hidden_states.mean(dim=1).detach()   # [B, 768]
        aim_logits = self.aim_head(pooled_img_feat)

        loss_aim = None
        if relation is not None:
            loss_aim = nn.CrossEntropyLoss()(aim_logits, relation.to(device))

        aim_probs = F.softmax(aim_logits, dim=-1)          # [B, 3]
        emo_rel_idx = 2                                    # 如果你的情感相关类别不是2，请改这里
        gate_img = aim_probs[:, emo_rel_idx]               # [B]
        # gate_img = gate_img.view(-1, 1, 1)                 # [B,1,1] 方便与 logits broadcast

        # 文本分支权重 = 1 - 图像分支权重
        # gate_text = 1.0 - gate_img                         # [B,1,1]
        # outputs是 文本情感线索/主线
        query = self.language_projection(input_hidden_states)
        query_attention_mask = torch.ones(
            query.size()[:-1], dtype=torch.long, device=query.device
        )
        # query = gate_img * query

        # 主线 T5 输入
        text_inputs_embeds = self.get_embeds(input_ids)
        inputs_embeds = torch.cat([query, text_inputs_embeds.to(query.device)], dim=1)
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

        # 多模态/辅助线 T5 输入 (来自第二个代码)
        imagetext_query = self.imagetext_projection(input_hidden_states)
        imagetext_query_attention_mask = torch.ones(
            imagetext_query.size()[:-1], dtype=torch.long, device=imagetext_query.device
        )
        # imagetext_query = gate_img * imagetext_query
        multi_inputs_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([imagetext_query, multi_inputs_embeds.to(imagetext_query.device)], dim=1)
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
        

        # fused_logits = weight * outputs.logits + (1 - weight) * multi_outputs.logits
        base_w = weight
        delta = 0.2

        dyn_w = base_w + delta * (gate_img - 0.5)
        dyn_w = dyn_w.clamp(0.0, 1.0)                 # [B]
        dyn_w = dyn_w.view(-1,1,1)                      

        fused_logits = (1-dyn_w) * outputs.logits + dyn_w * multi_outputs.logits  # [B, L_dec, V]
        main_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            main_loss = loss_fct(
                fused_logits.view(-1, fused_logits.size(-1)),
                labels.view(-1).to(device)
            )
        if main_loss is not None:
            if loss_aim is not None:
                total_loss = main_loss + 0.5 * loss_aim   # λ=0.5 可调
            else:
                total_loss = main_loss
        else:
            total_loss = None

        return total_loss, main_loss,loss_aim,aim_logits
    
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        input_multi_ids,
        attention_mask,
        input_multi_attention_mask,
        relation,         # 这里通常不用（只在 train 时使用）
        weight,           # 旧的融合同权重，不再用
        images,           # 原始图像
        text,             # aspect prompt
        max_new_tokens: int = 16,
        do_sample: bool = False,
    ):
        device = next(self.language_model.parameters()).device

        # === 0) 在线计算 Q-Former 特征 ===
        input_hidden_states = self._compute_qformer_feats_online(images, text)
        input_hidden_states = input_hidden_states.to(device)   # [B, n_q, 768]

        # === 1) AIM：获取图文相关性概率 ===
        pooled_img_feat = input_hidden_states.mean(dim=1).detach()         # [B, 768]
        aim_logits = self.aim_head(pooled_img_feat)               # [B, 3]
        aim_probs = F.softmax(aim_logits, dim=-1)                 # [B, 3]

        emo_rel_idx = 2                                           # 如果你的情感相关类别不是 2，请改这里
        gate_img = aim_probs[:, emo_rel_idx]         # [B,1,1]
        gate_text = 1.0 - gate_img                                # [B,1,1]

        # === 2) 主线 T5 输入（文本线） ===
        query = self.language_projection(input_hidden_states)
        query_attention_mask = torch.ones(
            query.size()[:-1], dtype=torch.long, device=query.device
        )
        # query = gate_img * query
        relation_inputs_embeds = self.get_embeds(input_ids)
        inputs_embeds = torch.cat([query, relation_inputs_embeds.to(query.device)], dim=1)
        attn_mask = torch.cat(
            [query_attention_mask, attention_mask.to(query_attention_mask.device)], dim=1
        )

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True
        )

        # === 3) 多模态分支 T5 输入（图像线） ===
        imagetext_query = self.imagetext_projection(input_hidden_states)
        imagetext_query_attention_mask = torch.ones(
            imagetext_query.size()[:-1], dtype=torch.long, device=imagetext_query.device
        )
        # imagetext_query = gate_img * imagetext_query
        
        multi_inputs_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([imagetext_query, multi_inputs_embeds.to(imagetext_query.device)], dim=1)
        multi_attention_mask = torch.cat(
            [imagetext_query_attention_mask, input_multi_attention_mask.to(imagetext_query_attention_mask.device)], dim=1
        )

        multi_outputs = self.language_model.generate(
            inputs_embeds=multi_inputs_embeds,
            attention_mask=multi_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True
        )

        # === 4) 使用 gating 进行 token-level 融合 ===
        base_w = weight
        delta = 0.2

        dyn_w = base_w + delta * (gate_img - 0.5)
        dyn_w = dyn_w.clamp(0.0, 1.0)                 # [B]
        dyn_w = dyn_w.view(-1,1,1)      
        fused_token_ids = []
        for step_logits_text, step_logits_img in zip(outputs.scores, multi_outputs.scores):
            # step_logits_text: [B,V]
            # step_logits_img:  [B,V]
            fused_logits = (1-dyn_w).squeeze(2).squeeze(1)[:,None] * step_logits_text + \
                        dyn_w.squeeze(2).squeeze(1)[:,None]  * step_logits_img

            fused_token_ids.append(fused_logits.argmax(dim=-1))   # [B]

        fused_token_ids = torch.stack(fused_token_ids, dim=1)     # [B, T_gen]

        # === 5) 解码 ===
        predicted_labels = self.tokenizer.batch_decode(
            fused_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return predicted_labels
