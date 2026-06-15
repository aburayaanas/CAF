# Presentation Notes: Figure 6 (PCA Gated Projection)

Use this guide to explain the PCA clustering in Figure 6. A statistics professor will immediately appreciate this explanation because it highlights the separation of concerns between **representation routing** (gating) and **classification** (the classifier head).

---

## 1. Why Gated Features Cluster by Vaping Status
In Figure 6:
*   **Plot A (Late Concatenation):** The input features are the raw modality scores: $[s_{tab}, s_{soft}, s_{hard}]$.
*   **Plot B (CAF):** The features are the **gated** modality scores: $[P_{tab} s_{tab}, P_{soft} s_{soft}, P_{hard} s_{hard}]$.

Because the gating network drives $P_{soft} \to 0$ for active vapers and $P_{soft} \approx 0.50$ for non-vapers, the gated feature vectors for these two groups are mathematically different:
*   **Vaper Feature Vector:** $\approx [0.90 \cdot s_{tab}, \mathbf{0.0}, 0.02 \cdot s_{hard}]$
*   **Non-Vaper Feature Vector:** $\approx [0.48 \cdot s_{tab}, \mathbf{0.50 \cdot s_{soft}}, 0.02 \cdot s_{hard}]$

Because the visual channel ($s_{soft}$) is completely shut off for vapers and active for non-vapers, the **largest source of variance** in the gated dataset is whether the visual feature is present or suppressed. Therefore, PCA (which projects features along directions of maximum variance) naturally separates patients by their **vaping status (routing pathway)** on the first principal component (PC1).

---

## 2. Why This is the Correct Behavior (The Defense)

### A. Gating is for Conflict Resolution, Not Classification
The gating network's job is **not** to classify healthy vs. diseased. Its job is to **resolve the conflict** (the vasoconstrictive masking effect) so that the classifier head receives clean, unconfounded data. 
By grouping vapers together and non-vapers together in the feature space, the gating network has successfully sorted patients into their correct biological routing tracks.

### B. The Classifier Head Separates the Targets
Once the gated features are routed, the classifier head projects them into a 1D decision space using the weighted sum:
$$\text{Sum} = P_{tab} s_{tab} + P_{soft} s_{soft} + P_{hard} s_{hard}$$
$$y_{logit} = scale \cdot \text{Sum} + bias_{classifier}$$

Because the visual lie ($s_{soft} \approx 0$ for diseased vapers) is successfully suppressed by the gate ($P_{soft} \to 0$), the sum correctly reflects the patient's true disease status without visual contamination. 
When we evaluate the final output ($\hat{y} = \sigma(y_{logit})$), **the target groups (healthy vs. diseased) are separated much better in the CAF than in the Late Concatenation model** (resulting in $69.57\%$ accuracy vs. $60.87\%$).

---

## Talking Points ("Lines to Say") to Your Professor

### 1. Explaining the PCA Separation
> *"Professor, in Plot B, the gated features $[P_m s_m]$ cluster by vaping status rather than disease status. This is mathematically expected. Because the gate suppresses the visual channel for vapers ($P_{soft} \to 0$) and activates it for non-vapers ($P_{soft} \approx 0.50$), the presence or absence of the visual channel is the dominant vector of variance. PCA captures this routing split on the first principal component."*

### 2. Defending the Classification Flow
> *"The gating network is responsible for **routing and conflict resolution**, not the final classification. Once the gated features are summed and passed to the classifier head ($y_{logit} = scale \cdot \sum P_m s_m + b$), the target groups (healthy vs. diseased) are successfully separated, which is why the CAF achieves a significant accuracy improvement over late concatenation ($69.57\%$ vs. $60.87\%$, McNemar $p = 0.0455$)."*
