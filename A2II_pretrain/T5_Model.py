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

class MyFlanT5(nn.Module):
    """
    Unified model for:
      - Stage-1: Relation-aware pretraining (task_type='relation') -> predict relevant/irrelevant from Q-Former pooler
      - Stage-2: Sentiment classification (task_type='sentiment')   -> A2II-like with selector + unified instruction

    It supports two feature sources:
      (A) Offline features: pass `input_hidden_state` and `input_pooler_output` (current DataProcessor output)
      (B) Online features : pass `images` + `aspect_prompt` -> compute Vision + Q-Former inside the model

    Args:
        model_path: str or T5ForConditionalGeneration
        task_type : 'relation' or 'sentiment'
        use_qformer_online: if True, initialize Vision+QFormer for online feature extraction
        freeze_vision: freeze vision backbone
        freeze_qformer: freeze Q-Former
        num_query_tokens: override config.num_query_tokens if not None
    """

    def __init__(
        self,
        model_path,
        blip_path,
        tokenizer,
        task_type: str = "sentiment",
        use_qformer_online: bool = True,
        freeze_vision: bool = True,
        freeze_qformer: bool = False,
        num_query_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.use_qformer_online = use_qformer_online

        # ===== 1) Language model (Flan-T5) =====
        if isinstance(model_path, str):
            self.language_model = T5ForConditionalGeneration.from_pretrained(model_path)
            self._lm_name_or_path = blip_path
        else:
            # allow passing a loaded T5 model instance (backward compatible with your train.py)
            self.language_model = model_path
            self._lm_name_or_path = blip_path

        # ===== 2) Vision + Q-Former (optional online path) =====
        self.processor = None
        self.vision_model = None
        self.qformer = None
        self.query_tokens = None
        self.tokenizer=tokenizer
        self.language_model.resize_token_embeddings(len(tokenizer))
        if self.use_qformer_online:
            # load InstructBLIP configs from same folder as LM if provided
            # (for xl/xxl checkpoints you used, they share a root dir)
            if self._lm_name_or_path is None:
                raise ValueError("When use_qformer_online=True, please pass a string path for model_path.")

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
                # still allow query_tokens to learn unless you also want to freeze them:
                # self.query_tokens.requires_grad_(False)

        # ===== 3) Selector / Projection =====
        # Project Q-Former query states into LM embedding space
        # self.language_projection = nn.Linear(768, 768)
        # # Selector over pooler_output: 0=use image-related instruction, 1=use text-only instruction
        self.selector = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        # # self.selector = nn.Linear(768, 2)
        self.language_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.LayerNorm(768)
        )

        self.init_linear_weight()

    def init_linear_weight(self):
        # 初始化 projection
        for m in self.language_projection:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # 初始化 selector
        for m in self.selector:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    # helper: LM input embeddings
    def get_embeds(self, input_ids):
        return self.language_model.get_input_embeddings()(input_ids)

    # ====== Online path: compute Q-Former features from raw image + aspect text ======
    def _compute_qformer_feats_online(self, images, aspect_prompt):
        """
        支持批处理版本：
        images: List[PIL.Image] 或单张 PIL.Image
        aspect_prompt: List[str] 或 str
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
            padding=True,                # ✅ 自动pad到batch最大长度
            truncation=True,             # ✅ 自动截断过长句子
            max_length=128,              # ⚙️ 可选（你数据中的句子一般不会太长）
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
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        images=None,    
        text=None,
        # ===== control =====
        task_type: Optional[str] = None,
    ):
        """
        Returns:
            - task_type='relation': {'loss', 'logits', 'probs'}
            - task_type='sentiment': HuggingFace T5 ModelOutput (supports outputs['loss'])
        """
        task_type = task_type or self.task_type
        device = next(self.language_model.parameters()).device

        # === 0) Prepare / ensure Q-Former features ===
        # If offline features are not provided, compute online if enabled.
        if self.use_qformer_online:
            if images is None:
                raise ValueError("Online Q-Former requires `images` and `text` when offline features are not provided.")
            input_hidden_state, input_pooler_output = self._compute_qformer_feats_online(images, text)
        else:
            raise ValueError("No Q-Former features provided and use_qformer_online=False.")    

        # Make sure tensors are on correct device & shapes
        if input_pooler_output.dim() == 2:
            input_pooler_output = input_pooler_output.unsqueeze(1)  # [B, 1, 768]
        input_hidden_state = input_hidden_state.to(device)
        input_pooler_output = input_pooler_output.to(device)

        # =============== 1) Relation Pretraining ===============
        if task_type == "relation":
            # Predict relevant/irrelevant 
            # CLS = input_hidden_state.mean(dim=1)   # [B, 768]
            # ---- 关键修改：projection 参与预训练 ----
            proj_hidden = self.language_projection(input_hidden_state)   # [B, n_q, 768]
            CLS = proj_hidden.mean(dim=1)                          # [B, 768]        
            rel_logits = self.selector(CLS)  # [B, 2]
            rel_probs = F.softmax(rel_logits, dim=-1)
            loss = None
            if labels is not None:
                # labels: 0=irrelevant, 1=relevant
                loss = F.cross_entropy(rel_logits, labels.long().to(device))
            return {"loss": loss, "logits": rel_logits, "probs": rel_probs}

        # =============== 2) Sentiment Classification (A2II-like) ===============
        elif task_type == "sentiment":
            """
            - 仍利用 Q-Former 的 query 表示作为前置查询 embedding，拼接到 T5 的 inputs_embeds 前端；
            - labels 为 T5 文本标签（包含 -100 mask），T5 自带 seq2seq loss。
            """
            if input_ids is None or attention_mask is None:
                raise ValueError("Sentiment task requires `input_ids` and `attention_mask` from tokenizer.")

            # 由 Q-Former 得到的多模态查询向量，映射到 LM 空间后拼接
            # query = self.language_projection(input_hidden_state)              # [B, n_q, 768]
            query = self.language_projection(input_hidden_state)  
            query_attention_mask = torch.ones(query.size()[:-1], dtype=torch.long, device=device)

            # 将原始 token ids 映射为 embedding，并和 query 拼接
            inputs_embeds = self.get_embeds(input_ids)                        # [B, L, 768]
            inputs_embeds = torch.cat([query, inputs_embeds.to(device)], dim=1)   # [B, n_q+L, 768]
            attn_mask = torch.cat([query_attention_mask, attention_mask.long().to(device)], dim=1)

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )
            return outputs

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")


    # ====== Generation for Sentiment Prediction ======
    @torch.no_grad()
    def generate(
        self,
        images,
        text,
        input_ids,
        attention_mask,
        max_new_tokens: int = 16,
        num_beams: int = 1,
        do_sample: bool = False,
    ):
        """
        Generate predicted sentiment labels (e.g. 'positive', 'negative', 'neutral')
        using Q-Former + T5 decoder generate().
        Args:
            images: list[PIL.Image] or batched image tensors
            text: list[str] (aspect-aware raw sentences, e.g. "The <asp>room<asp> is clean.")
            input_ids: tokenized instruction+sentence for T5
            attention_mask: corresponding attention mask
        Returns:
            predicted_labels: list[str]
        """
        device = next(self.language_model.parameters()).device

        # ---- Step 1. Get Q-Former multimodal features ----
        if not self.use_qformer_online:
            raise ValueError("generate() only supports online Q-Former feature extraction.")
        input_hidden_state, _ = self._compute_qformer_feats_online(images, text)

        # ---- Step 2. Project query and concatenate with T5 embeddings ----
        
        query = self.language_projection(input_hidden_state)  # [B, n_q, 768]
        query_attention_mask = torch.ones(query.size()[:-1], dtype=torch.long, device=device)
        inputs_embeds = self.get_embeds(input_ids)             # [B, L, 768]
        inputs_embeds = torch.cat([query, inputs_embeds.to(device)], dim=1)
        attn_mask = torch.cat([query_attention_mask, attention_mask.long().to(device)], dim=1)

        # ---- Step 3. Use T5 generate() ----
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample
        )

        # ---- Step 4. Decode tokens to text labels ----
        predicted_labels = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ) 

        return predicted_labels
