import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


class CausalLMClassifier(nn.Module):
    def __init__(self, lm_name: str, num_classes: int=3):
        super().__init__()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        lm = AutoModelForCausalLM.from_pretrained(
            lm_name,
            device_map="auto",
            quantization_config=bnb_config
        )

        # LoRa
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["query_key_value"]
        )
        self.backbone = get_peft_model(lm, lora_config)

        self.backbone.print_trainable_parameters()

        # classification head
        hidden_size = self.backbone.config.hidden_size
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
        last_hidden = outputs.hidden_states[-1]     # [B, T, D]
        last_token_index = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[
            torch.arange(last_hidden.shape[0]),
            last_token_index,
            :
        ]      # [B, D]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
    