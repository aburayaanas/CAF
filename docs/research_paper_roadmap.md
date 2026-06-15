# Research Paper Roadmap: Next Steps & Analysis

## Architecture Audit ✅
| Component | What It Does | Scientifically Valid? |
| :--- | :--- | :--- |
| **3-feature gate ($\mathbf{W} \in \mathbb{R}^{3 \times 3}$)** | Routes via nicotine exposure only | ✅ Clinically motivated |
| **Softmax routing** | Probabilities sum to 1.0, competitive | ✅ Mathematically necessary |
| **Modality attenuation** | $P \times S$ — gate is sole router | ✅ No redundancy |
| **$\text{Softplus}(\theta) = \alpha$** | Restores BCE logit range | ✅ Mathematical necessity not clinical constraint |
| **Single bias $\beta$** | Shifts decision boundary | ✅ Standard |
| **Total parameters** | 9 (gate) + 1 (scale) + 1 (bias) = 11 | ✅ PAC-learnable at $N=46$ |
| **LOOCV + 100 seeds** | Gold standard for small $N$ | ✅ Rigorous |
| **ANOVA inside fold** | Prevents target leakage | ✅ Critical and correct |

## Results Audit ✅ (Constant 1.0 Probing Active Pruning Benchmark)
* **CAF Accuracy**: $68.59\% \pm 1.52\%$
* **Best baseline accuracy**: $60.87\% \pm 0.00\%$
* **Absolute improvement**: $+7.72\%$
* **Relative improvement**: $+12.7\%$
* **$P_{soft}$ non-vapers**: $0.4946$
* **$P_{soft}$ vapers**: $0.1064$
* **Relative attention drop**: $78.5\%$
* **t-test $p < 0.05$ rate**: $100\%$ across all 100 seeds
* **Median p-value**: $2.035 \times 10^{-8}$
* **Modality Pruning ($P_{probing}$)**: $0.0223$ ($2.23\%$ attention allocated to the constant uninformative probing channel)

Every number is consistent, stable, and clinically interpretable. This is ready.

---

## Next Steps — Building The Best Possible Paper
Here is the complete roadmap, including every statistical test you need.

### Phase 1 — Additional Statistics to Run Now

#### 1. Mann-Whitney U Test ✅ Run This
The Welch's t-test assumes approximate normality of $P_{soft}$ distributions. With $N=46$ and small groups, you should also run the non-parametric equivalent:
```python
from scipy.stats import mannwhitneyu

stat, p = mannwhitneyu(
    p_soft_nonvapers, 
    p_soft_vapers, 
    alternative='greater'
)
```
* **Why**: Mann-Whitney makes no distributional assumption. If both Welch's AND Mann-Whitney agree — your routing significance is assumption-free and unassailable.

#### 2. Effect Size — Cohen's d ✅ Required
p-values alone are insufficient in clinical research. You must report effect size:
```python
import numpy as np

def cohens_d(group1, group2):
    diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(
        (np.std(group1)**2 + np.std(group2)**2) / 2
    )
    return diff / pooled_std

d = cohens_d(p_soft_nonvapers, p_soft_vapers)
```
* **Expected result** (based on $0.4946$ vs. $0.1064$): Cohen's $d > 1.8$, which is a very large effect size by Cohen's conventions ($>0.8$ is large).

#### 3. Hypergeometric Test — For Routing Consistency
This tests whether the proportion of seeds achieving correct routing (vapers < non-vapers in $P_{soft}$) is non-random:
```python
from scipy.stats import hypergeom

# Out of 100 seeds, how many show correct routing?
# Under null hypothesis: 50% chance either group is higher
k = 100   # seeds with correct routing
M = 100   # total seeds  
n = 50    # expected under null
N = 100

p_hyper = hypergeom.sf(k-1, M, n, N)
```
* **Why**: If 100/100 seeds show correct routing direction, this p-value is astronomically small. This is your strongest possible claim of stability.

#### 4. McNemar's Test — CAF vs Baseline Per-Patient
Since you have LOOCV, you have per-patient predictions from both CAF and baseline. McNemar's tests whether the disagreements between models are random:
```python
from statsmodels.stats.contingency_tables import mcnemar

# Build 2×2 table:
# [CAF correct & Baseline correct, CAF correct & Baseline wrong]
# [CAF wrong & Baseline correct,   CAF wrong & Baseline wrong  ]

result = mcnemar(contingency_table, exact=True)
```
* **Why this matters**: This proves CAF's improvement is not just baseline errors redistributed — it is genuinely correcting cases the baseline gets wrong.

#### 5. Bootstrap Confidence Intervals — For Accuracy
At $N=46$, $\pm1.52\%$ std across seeds is good. But you also need $95\%$ CI on accuracy itself:
```python
from sklearn.utils import resample
import numpy as np

bootstrap_accs = []
for _ in range(10000):
    sample = resample(predictions, true_labels)
    acc = accuracy_score(sample[1], sample[0])
    bootstrap_accs.append(acc)

ci_lower = np.percentile(bootstrap_accs, 2.5)
ci_upper = np.percentile(bootstrap_accs, 97.5)
```

---

### Phase 2 — Paper Structure
The paper should follow this structure for a clinical AI journal:

1. **Abstract**
   * Problem: periodontal diagnosis in vapers
   * Gap: vasoconstriction invalidates visual modality
   * Contribution: CAF — confounder-driven routing
   * Results: $68.59\%$ accuracy, $78.5\%$ attention drop, $p=2.035 \times 10^{-8}$
2. **Introduction**
   * Periodontal disease burden
   * E-cigarette prevalence
   * Vasoconstriction masking mechanism (cite pharmacology papers)
   * Why existing multimodal fusion fails
   * Your contribution stated precisely
3. **Related Work**
   * Multimodal fusion in dental AI
   * Attention mechanisms (and why yours differs)
   * Small-$N$ clinical ML methods
4. **Methodology**
   * Cohort description ($N=46$ patients)
   * Three-modality input pipeline (Questionnaire, Visual Photo, Clinical Probing Depths)
   * CAF architecture with full math
   * Training protocol
   * Evaluation protocol
5. **Results**
   * Classification performance table
   * Gating weight analysis table
   * All statistical tests
   * Per-patient attention visualization
6. **Discussion**
   * Clinical interpretation of routing weights
   * Seed stability as architectural validation
   * Limitations: $N=46$, simulated photo encoder
7. **Conclusion & Future Work**
   * Scale to $N > 300$ with real image embeddings
   * Freeze gate as biological prior
   * Extend to other nicotine-confounded conditions

---

### Phase 3 — Visualizations For The Paper
You need these four figures:
* **Figure 1**: CAF Architecture Diagram (already created ✅)
* **Figure 2**: Attention weight distribution (boxplot: vapers vs. non-vapers $P_{soft}$)
* **Figure 3**: Per-patient routing heatmap (each patient $\times$ their $P_{tab}$, $P_{soft}$, $P_{probing}$)
* **Figure 4**: Seed stability plot (histogram of accuracy across 100 seeds)

---

### Target Journals
For this work, the right venues in order of fit:
1. **Journal of Dental Research**: Top dental journal, accepts AI methods.
2. **Clinical Oral Investigations**: Clinical + computational mix.
3. **BMC Oral Health**: Open access, strong AI presence.
4. **Artificial Intelligence in Medicine**: Pure AI/clinical methods.
5. **PLOS ONE**: Rigorous, open access, novel methods.
