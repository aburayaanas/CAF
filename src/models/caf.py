import torch
import torch.nn as nn
import torch.nn.functional as F

class ClinicalAttentionFramework(nn.Module):
    """
    Clinical Attention Framework (CAF) Gating & Classification Model.
    Specifically designed for small clinical pilot cohorts with low parameter count.
    
    1. Gating Network (12 parameters):
       Takes 3-dimensional nicotine/vaping profile (X_gate[:, 2:5]) -> Linear(3, 3) + Softmax -> 3-D Attention Weights [P_tab, P_soft, P_hard].
       
    2. Modality Weighting:
       Weighted features = [P_tab * s_tab, P_soft * s_soft, P_hard * s_hard].
       
    3. Classification Head (2 parameters):
       logit = exp(log_scale) * (w_tab + w_soft + w_hard) + bias
       This enforces that the gating network is the ONLY modality routing mechanism.
    """
    def __init__(self):
        super().__init__()
        # Gating Layer: 3 vaping features (Frequency, Duration, Nicotine) -> 3 outputs (softmax logits)
        self.gating = nn.Linear(3, 3)
        # Scaled summation classification head parameters: use unconstrained learned scale
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.classifier_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_gate, s_tab, s_soft, s_hard):
        """
        x_gate: [Batch, 5] - Vaping Profile (first two columns age/gender are ignored for gating)
        s_tab:  [Batch, 1] - Questionnaire Score (probability output of baseline classifier)
        s_soft: [Batch, 1] - Clinical Photo Rating (scaled to [0.0, 1.0])
        s_hard: [Batch, 1] - Radiographic Bone Loss Score (scaled to [0.0, 1.0])
        """
        # 1. Extract 3-D vaping habits (columns 2, 3, 4)
        x_gate_vaping = x_gate[:, 2:5]
        
        # 2. Compute attention routing weights
        gate_logits = self.gating(x_gate_vaping)
        probs = F.softmax(gate_logits, dim=1)  # [Batch, 3]
        
        p_tab = probs[:, 0:1]
        p_soft = probs[:, 1:2]
        p_hard = probs[:, 2:3]
        
        # 3. Attenuate modalities using routing weights
        w_tab = p_tab * s_tab
        w_soft = p_soft * s_soft
        w_hard = p_hard * s_hard
        
        # 4. Compute logit using unconstrained scale factor and bias
        logit = self.scale * (w_tab + w_soft + w_hard) + self.classifier_bias
        
        return logit, probs



