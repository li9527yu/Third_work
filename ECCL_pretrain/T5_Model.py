# T5_Model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    T5ForConditionalGeneration,
    InstructBlipConfig,
    InstructBlipVisionModel,
    InstructBlipQFormerModel,
    InstructBlipProcessor,
)
from PIL import Image


class MyFlanT5_ECCL(nn.Module):
    """
    专用于 ECCL（情感一致性对比学习）预训练的模型

    - 使用 Vision + Q-Former 提取多模态情感特征
    - 不再做原来的 relation_s 图文匹配任务
    - 不依赖 T5 解码器，只用到 T5 作为潜在后续微调的 backbone（目前 ECCL 只用 Q-Former 分支）

    Forward(task_type='eccl'):
        输入: images(list[PIL]) + text(list[str]) + labels(0/1)
        输出: {'loss', 'logits', 'probs'}
    """

    def __init__(
        self,
        model_path: str,
        blip_path: str,
        tokenizer,
        task_type: str = "eccl",
        use_qformer_online: bool = True,
        freeze_vision: bool = True,
        freeze_qformer: bool = False,
        num_query_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.use_qformer_online = use_qformer_online

        # ===== 1) Language model (Flan-T5)，此处主要是为了后续微调预留 =====
        self.language_model = T5ForConditionalGeneration.from_pretrained(model_path)
        self._lm_name_or_path = blip_path

        self.tokenizer = tokenizer
        self.language_model.resize_token_embeddings(len(tokenizer))

        # ===== 2) Vision + Q-Former (online feature) =====
        self.processor = None
        self.vision_model = None
        self.qformer = None
        self.query_tokens = None

        if self.use_qformer_online:
            if self._lm_name_or_path is None:
                raise ValueError("When use_qformer_online=True, blip_path must be a valid path.")

            blip_config = InstructBlipConfig.from_pretrained(self._lm_name_or_path)
            self.vision_model = InstructBlipVisionModel(blip_config.vision_config)
            self.qformer = InstructBlipQFormerModel(blip_config.qformer_config)
            self.processor = InstructBlipProcessor.from_pretrained(self._lm_name_or_path)

            # learnable query tokens
            n_q = num_query_tokens if num_query_tokens is not None else blip_config.num_query_tokens
            h_q = blip_config.qformer_config.hidden_size
            self.query_tokens = nn.Parameter(torch.zeros(n_q, h_q))

            # freeze options
            if freeze_vision:
                for p in self.vision_model.parameters():
                    p.requires_grad = False
            if freeze_qformer:
                for p in self.qformer.parameters():
                    p.requires_grad = False

        # ===== 3) 多模态情感特征投影 + ECCL 分类头 =====
        # 3.1 将 Q-Former query states 投影到共享语义空间（后续情感微调也可复用）
        self.language_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.LayerNorm(768)
        )

        # 3.2 ECCL 头：二分类（情感一致 vs 不一致/无关）
        self.eccl_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # 输出 logit
        )

        self.eccl_criterion = nn.BCEWithLogitsLoss()

        self.init_weights()

    def init_weights(self):
        # language_projection 的 Linear 使用 Xavier，LayerNorm 默认即可
        for m in self.language_projection:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # ECCL head 初始化
        for m in self.eccl_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # helper: LM input embeddings（目前 ECCL 不用，但保留接口）
    def get_embeds(self, input_ids):
        return self.language_model.get_input_embeddings()(input_ids)

    # ====== Online path: compute Q-Former features from raw image + text ======
    def _compute_qformer_feats_online(self, images, aspect_prompt):
        """
        支持批处理版本：
        images: List[PIL.Image] 或单张 PIL.Image
        aspect_prompt: List[str] 或 str
        """
        device = next(self.language_model.parameters()).device

        if isinstance(images, (Image.Image, torch.Tensor)):
            images = [images]
        if isinstance(aspect_prompt, str):
            aspect_prompt = [aspect_prompt]

        assert len(images) == len(aspect_prompt), \
            f"Image/Text batch mismatch: {len(images)} vs {len(aspect_prompt)}"

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
        vision_outputs = self.vision_model(pixel_values=inputs["pixel_values"])
        image_embeds = vision_outputs[0]  # [B, L_img, D]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        # Q-Former
        B = image_embeds.shape[0]
        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
        qformer_attention_mask = torch.cat([query_attention_mask, inputs["qformer_attention_mask"]], dim=1)

        q_out = self.qformer(
            input_ids=inputs["qformer_input_ids"],
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        hidden_state = q_out[0][:, : query_tokens.size(1), :]  # [B, n_q, 768]
        pooler_output = q_out[1].unsqueeze(1) if q_out[1].dim() == 2 else q_out[1]  # [B, 1, 768]
        return hidden_state, pooler_output

    def forward(
        self,
        images=None,
        text=None,
        labels=None,
        task_type: Optional[str] = None
    ):
        """
        当前版本只实现 task_type='eccl':
            输入: images + text + eccl_label(0/1)
            输出: {'loss', 'logits', 'probs'}
        """
        task_type = task_type or self.task_type
        device = next(self.language_model.parameters()).device113116
        

        if task_type != "eccl":
            raise ValueError(f"Current model only supports task_type='eccl', got {task_type}")

        if not self.use_qformer_online:
            raise ValueError("ECCL pretraining requires use_qformer_online=True.")

        if images is None or text is None:
            raise ValueError("ECCL pretraining requires both `images` and `text` inputs.")

        # ---- 1. Q-Former 多模态特征 ----
        input_hidden_state, _ = self._compute_qformer_feats_online(images, text)
        input_hidden_state = input_hidden_state.to(device)  # [B, n_q, 768]

        # ---- 2. 投影 + 池化成一个多模态情感向量 ----
        proj_queries = self.language_projection(input_hidden_state)  # [B, n_q, 768]
        mm_feat = proj_queries.mean(dim=1)  # [B, 768]

        # ---- 3. ECCL 头进行二分类 ----
        logits = self.eccl_head(mm_feat).squeeze(-1)   # [B]
        probs = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            labels = labels.view_as(logits).float().to(device)
            loss = self.eccl_criterion(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "probs": probs
        }
