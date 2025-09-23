import torch
import torch.nn as nn

from transformers import (
    ViTPreTrainedModel,
    ViTConfig,
    ViTModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

class VitClassification(ViTPreTrainedModel):
    config_class = ViTConfig
    base_model_prefix = "vit"

    def __init__(self, config):
        super().__init__(config)

        # 1) load the raw ViT backbone
        self.vit = ViTModel.from_pretrained(config.name_or_path)

        # Freeze backbone initially (optional - good for large datasets)
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last few transformer blocks
        for param in self.vit.encoder.layer[-8:].parameters():
            param.requires_grad = True

        # 2) define the classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),  # was // 2
            nn.BatchNorm1d(config.hidden_size),  # was  // 2
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_labels), # was // 2 on first param
        )
        # Initialize only the classifier
        self._init_classifier()

    def _init_classifier(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def channel_norm(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        return (x - mean) / (std + 1e-8)


    def forward(self, pixel_values, labels=None):
        norm_values = self.channel_norm(pixel_values)
        outputs = self.vit(pixel_values=norm_values)
        pooled  = outputs.pooler_output           # [batch, hidden_size]
        logits  = self.classifier(pooled)         # [batch, num_labels]
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
