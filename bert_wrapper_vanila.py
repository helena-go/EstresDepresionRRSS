import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW


class BertWithFeatures(nn.Module):
    def __init__(self, feature_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, features, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        combined = torch.cat([cls_output, features], dim=1)  # [batch, 768+N]
        return self.classifier(combined)
    
class CustomDataset(Dataset):
    def __init__(self, X, y, _config = None):
        self.X = X
        self.y = y
        self.config = _config
        self.tokenizer = AutoTokenizer.from_pretrained(_config['model_name'])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        out = torch.tensor(self.tokenizer.encode(self.X[idx], max_length=self.config['max_length'], 
                            padding='max_length', add_special_tokens=True, 
                            truncation = 'longest_first'))
        if self.y:
            return {"text": out, "attention":(out!=1).float(), "label":torch.tensor(self.y[idx])}
        else:
            return {"text": out, "attention":(out!=1).float()}