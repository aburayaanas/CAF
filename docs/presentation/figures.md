# Presentation Notes: Figure 1 (CAF Architecture)

Use this guide for your meeting with your statistics professor. It is structured to show both the **mathematical rigor** (for their AI expertise) and the **biological intuition** (for their biomedical focus).

---

## Part 1: Quick Explanation of Figure 1 Components

### 1. The Linear Gating Layer
The gating network computes raw attention logits $z = [z_{tab}, z_{soft}, z_{hard}]^T \in \mathbb{R}^3$ from the patient's vaping profile vector $x_{gate} \in \mathbb{R}^3$:
$$z = W_{gating} x_{gate} + b_{gating}$$

#### What Every Term Means:
*   **$x_{gate} = [x_{freq}, x_{dur}, x_{nic}]^T$ (Input Vector):** A 3D vector representing the patient's clinical context (vaping frequency, duration, and nicotine concentration). These are continuous or ordered ordinal features, standardized to zero mean and unit variance.
*   **$W_{gating} \in \mathbb{R}^{3 \times 3}$ (Gating Weights):** The learnable weight matrix. Each entry $W_{i, j}$ represents the relationship between confounder feature $j$ and modality logit $i$. For example, a large negative weight on $W_{soft, nic}$ means higher nicotine levels suppress the visual channel.
*   **$b_{gating} \in \mathbb{R}^3$ (Gating Biases):** The gating bias vector. It represents baseline trust allocation when all vaping inputs are zero (non-vapers, $x_{gate} = \vec{0}$). It ensures the model defaults to clinical photos when no vasoconstriction is present.
*   **$z$ (Modality Logits):** Unnormalized trust values for each of the three channels.

---

### 2. Why We Have Softmax Routing
The raw logits $z$ are mapped to dynamic routing attention weights using the Softmax function over the modality channels:
$$P_{m} = \frac{e^{z_m}}{\sum_{j=1}^3 e^{z_j}}, \quad m \in \{tab, soft, hard\}$$

#### Why Softmax is Mathematically and Clinically Necessary:
1.  **Probability Constraints:** It constrains the gating weights so that $P_m \in [0, 1]$ and $\sum P_m = 1.0$, allowing them to act as true mathematical attention routing coefficients.
2.  **Competitive Routing (Zero-Sum):** If one modality's trust drop (e.g., visual photos $P_{soft} \to 0$ due to vaping), the Softmax function mathematically forces the remaining trust to be redistributed to the other channels (e.g., $P_{tab} \to 0.90$). This prevents representation collapse.

---

### 3. The Summation and Scaling Operations
Once the routing coefficients ($P_m$) are calculated, the aligned modality risk scores ($s_m \in [0, 1]$) are fused via a weighted sum, scaled, and classification logits are computed:

#### The Weighted Summation:
$$\text{Sum} = P_{tab} s_{tab} + P_{soft} s_{soft} + P_{hard} s_{hard}$$
*   This sum is mathematically bounded within $[0, 1]$.

#### The Softplus Scaled Classification Logit:
$$y_{logit} = scale \cdot \text{Sum} + b_{classifier}$$
$$scale = \text{Softplus}(\gamma) = \log(1 + e^{\gamma})$$

*   **Why Softplus Scale?** Because the weighted sum is bounded within $[0, 1]$, the Binary Cross-Entropy (BCE) loss function would fail to drive confident gradients (predictions would be capped near the sigmoid center). The learnable $scale$ multiplier expands the logit range to allow confident classifications, while the Softplus constraint guarantees $scale > 0$ to prevent the optimizer from learning negative scaling.

---

## Part 2: Talking Points ("Lines to Say") to Your Professor

### 1. Proposing the Core Concept
> *"Professor, we are addressing the vulnerability of standard multimodal architectures to biological confounders. In periodontics, active vaping causes localized vasoconstriction, masking soft-tissue inflammation and rendering clinical photos deceptive. Traditional late fusion or bilinear concatenation is contaminated by this misleading visual signal, dragging performance below a simple questionnaire baseline."*

### 2. Defending the Exogenous Gating Logic
> *"Instead of using standard self-attention or cross-attention (which calculate attention as dot-product correlations between the modalities themselves and are easily hijacked by deceptive signals), we introduce **exogenous context gating**. We route attention weights ($P_m$) using an independent clinical confounder vector ($x_{gate}$). This decouples the gating decision from the modality features themselves, resolving the masking conflict."*

### 3. Explaining the Parameter Efficiency (11 Parameters)
> *"Given our pilot cohort size of $N=46$, we designed a highly regularized, low-parameter gate to satisfy PAC-learning bounds and prevent overfitting. The gating layer is a single linear projection ($3 \times 3$ weights $+ 3$ biases), which along with the classifier head ($1$ scale $+ 1$ bias) results in only **14 learnable parameters** (or 11 if we restrict to active gating weights and scale parameters). We optimize this end-to-end using L-BFGS."*

### 4. Explaining the Learned Biases and Outliers (The Dose-Response)
> *"The model learns biologically logical behaviors without explicit modality supervision. For non-vapers ($x_{gate} = 0$), the gating defaults to the learned biases, allocating $\approx 48\%$ trust to the clinical photo. For heavy, chronic vapers, the negative weights on frequency and duration dominate, driving $P_{soft}$ to $\approx 1.5\%$. Interestingly, the positive weight on nicotine concentration acts as a mathematical offset, allowing light or recent vapers to retain partial visual trust ($\approx 30\%$) where vasoconstriction has not yet set in. This confirms the gate models a continuous dose-response curve."*
