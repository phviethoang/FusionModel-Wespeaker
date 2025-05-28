# wespeaker/models/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, fbank_dim, wavlm_dim, d_model, num_heads, dropout=0.0):
        super(CrossAttentionFusion, self).__init__()
        # Chiếu fbank và wavlm features về cùng chiều d_model
        self.fbank_query = nn.Linear(fbank_dim, d_model)
        self.wavlm_key = nn.Linear(wavlm_dim, d_model)
        self.wavlm_value = nn.Linear(wavlm_dim, d_model)
        self.wavlm_query = nn.Linear(wavlm_dim, d_model)
        self.fbank_key = nn.Linear(fbank_dim, d_model)
        self.fbank_value = nn.Linear(fbank_dim, d_model)
        

        for layer in [self.fbank_query, self.wavlm_key, self.wavlm_value, self.wavlm_query, self.fbank_key, self.fbank_value]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.attention1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.output_projection = nn.Linear(d_model * 2, d_model)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, fbank_features, frontend_features):
        B, T, _ = fbank_features.shape
        # Debug
        # Kiểm tra NaN/Inf trong input
        if torch.isnan(fbank_features).any() or torch.isinf(fbank_features).any():
            raise ValueError("NaN/Inf detected in fbank_features")
        if torch.isnan(frontend_features).any() or torch.isinf(frontend_features).any():
            raise ValueError("NaN/Inf detected in frontend_features")

        # Hướng 1: fbank -> wavlm
        query1 = self.fbank_query(fbank_features)
        key1 = self.wavlm_key(frontend_features)
        value1 = self.wavlm_value(frontend_features)
        
        # Ổn định attention scores
        attn_output1, _ = self.attention1(query1, key1, value1)
        attended1 = self.norm1(attn_output1 + query1)
        attended1 = self.dropout(attended1)

        # Hướng 2: wavlm -> fbank
        query2 = self.wavlm_query(frontend_features)
        key2 = self.fbank_key(fbank_features)
        value2 = self.fbank_value(fbank_features)
        
        attn_output2, _ = self.attention2(query2, key2, value2)
        attended2 = self.norm2(attn_output2 + query2)
        attended2 = self.dropout(attended2)

        # Kết hợp hai hướng
        combined = torch.cat((attended1, attended2), dim=-1)
        
        output = self.output_projection(combined)
        #Debug
        # Kiểm tra NaN/Inf trong output
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError("NaN/Inf detected in CrossAttentionFusion output: "+combined+ "become "+ output)

        return output