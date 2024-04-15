import torch
from torch import nn
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim, 1))

    def forward(self, x):
        weights = torch.softmax(self.attention_weights, dim=0)
        # Apply weights and ensure output is [batch_size, 1]
        attended = torch.matmul(x, weights)  # This results in [batch_size, 1]
        return attended 

class RelationshipModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
        super(RelationshipModel, self).__init__()
        self.pathway1 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.pathway2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        self.attention = Attention(1024)
        self.classifier = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )

    def forward(self, combined_input):
        input1, input2 = combined_input[:, :768], combined_input[:, 768:1536]
        features1 = self.pathway1(input1)
        features2 = self.pathway2(input2)
        combined_features = torch.cat((features1, features2), dim=1)
        attended_features = self.attention(combined_features)
        return self.classifier(attended_features)


