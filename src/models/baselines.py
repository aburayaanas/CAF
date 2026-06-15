import torch
import torch.nn as nn
import torch.nn.functional as F

class LateConcatenationFusion(nn.Module):
    """
    Standard Late Concatenation Fusion Baseline.
    Concatenates s_tab, s_soft, and s_hard into a 3-D vector and maps to prediction logit.
    No dynamic gating or weighting is performed.
    """
    def __init__(self):
        super().__init__()
        # 3 inputs (s_tab, s_soft, s_hard) -> 1 output logit
        self.classifier = nn.Linear(3, 1)

    def forward(self, s_tab, s_soft, s_hard):
        fused = torch.cat([s_tab, s_soft, s_hard], dim=1) # [Batch, 3]
        return self.classifier(fused)

class GatedMultimodalUnit(nn.Module):
    """
    Standard Gated Multimodal Unit (GMU) Baseline (Arevalo et al., 2016).
    Uses self-attention gating. The gate weights are learned directly from 
    the modalities themselves, rather than from an independent clinical confounder (X_gate).
    """
    def __init__(self):
        super().__init__()
        # Map concatenated modalities to 3 gate logits
        self.gate_layer = nn.Linear(3, 3)
        # Final classifier on the gated sum
        self.classifier = nn.Linear(1, 1)

    def forward(self, s_tab, s_soft, s_hard):
        concat_feats = torch.cat([s_tab, s_soft, s_hard], dim=1) # [Batch, 3]
        
        # Self-gated attention weights
        gate_weights = F.softmax(self.gate_layer(concat_feats), dim=1)
        
        # Weighted sum of inputs
        fused = (
            gate_weights[:, 0:1] * s_tab +
            gate_weights[:, 1:2] * s_soft +
            gate_weights[:, 2:3] * s_hard
        ) # [Batch, 1]
        
        return self.classifier(fused)

class BilinearTensorFusion(nn.Module):
    """
    Simplified Tensor Fusion Network (TFN) Baseline.
    Computes all bilinear cross-modal interactions (outer product equivalents) 
    between s_tab, s_soft, and s_hard to explicitly model multi-modal correlations.
    """
    def __init__(self):
        super().__init__()
        # Inputs: s_tab, s_soft, s_hard, (tab*soft), (tab*hard), (soft*hard), (tab*soft*hard)
        # Total of 7 features
        self.classifier = nn.Linear(7, 1)

    def forward(self, s_tab, s_soft, s_hard):
        # Bilinear cross terms
        tab_soft = s_tab * s_soft
        tab_hard = s_tab * s_hard
        soft_hard = s_soft * s_hard
        triple_term = s_tab * s_soft * s_hard
        
        # Concatenate 1st, 2nd, and 3rd order features
        fused = torch.cat([
            s_tab, s_soft, s_hard,
            tab_soft, tab_hard, soft_hard,
            triple_term
        ], dim=1) # [Batch, 7]
        
        return self.classifier(fused)
