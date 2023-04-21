from sentence_transformers import util as st_utils
import numpy as np
import torch
from icecream import ic
from torch._C import device
import torch.nn.functional as F
import numpy as np


class RewardActionFunction(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        prompt_feature_dim,
        task_feature_dim,
        fusion_output_size,
        dropout_p,
        task_type="classification",
    ):
        super(RewardActionFunction, self).__init__()
        self.fusion = torch.nn.Linear(
            in_features=(prompt_feature_dim + task_feature_dim), 
            out_features=fusion_output_size
        )
        self.dense = torch.nn.Linear(
            in_features=fusion_output_size, 
            out_features=fusion_output_size
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size, 
            out_features=num_classes
        )
        self.num_classes = num_classes
        self.type = task_type
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, prompt_embedding, task_embedding, label=None):
        prompt_features = torch.nn.functional.relu(
            prompt_embedding
        )
        task_features = torch.nn.functional.relu(
            task_embedding
        )
        combined = torch.cat(
            [prompt_features, task_features]
        )
        fused = self.dropout(
            torch.nn.functional.relu(
            self.fusion(combined.float())
            )
        )
        dense_out = self.dense(fused)
        dense_out = self.dropout(dense_out)
        logits = self.fc(dense_out)
        logits = torch.nn.functional.tanh(logits)
        loss = (
            self.loss_fn(logits, label)
            if label is not None else label
        )
        return (logits, loss)