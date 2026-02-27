import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

class RobertaClassifier(nn.Module):
    def __init__(self, hf_model_path: str, num_classes: int):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(hf_model_path)
        self.roberta = RobertaModel.from_pretrained(hf_model_path)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs["pooler_output"]
        pooled = self.drop(pooled)
        return self.fc(pooled)
