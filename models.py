import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class CausalLMClassifier(nn.Module):
    def __init__(self, lm_name: str, num_classes: int = 3):
        super().__init__()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.backbone = AutoModelForCausalLM.from_pretrained(
            lm_name,
            device_map="auto",
            quantization_config=bnb_config
        )
   
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        dtype = self.backbone.dtype  

        #  Trainable classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]  # [B, T, D]
        device = last_hidden.device

       # Get last token for each sequence
        last_token_index = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[
            torch.arange(last_hidden.shape[0], device=device),
            last_token_index,
            :
        ]  # [B, D]
        
        pooled = pooled.to(dtype=self.classifier[0].weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
