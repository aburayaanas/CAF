# Project Summary: The Clinical Attention Framework (CAF) for Periodontal Diagnosis

**Prepared for:** University Scientific & Clinical Review Board
**Project Title:** Resolving the "Vaping Paradox" in Multimodal Periodontal Screening via Confounder-Driven Gating

---

## 1. Executive Summary

This project introduces and validates the **Clinical Attention Framework (CAF)**, a novel, gated multimodal deep learning architecture designed to handle physiological confounders in medical diagnostics. As a clinical proof-of-concept, we address the **"Vaping Paradox"** in periodontics—a phenomenon where nicotine-induced peripheral vasoconstriction suppresses the cardinal visual signs of gingival inflammation (erythema and bleeding), presenting a deceptive "healthy" visual appearance in patients with underlying, active periodontitis.

By utilizing patient-specific clinical context (vaping habits) as an exogenous gating signal, the CAF dynamically routes diagnostic trust. It down-weights the confounded photographic channel in active vapers and relies instead on objective risk factors and radiographic inputs. Tested on a clinical cohort ($N=46$), the CAF achieved an ensembled accuracy of **$69.57\%$** (mean 100-seed LOOCV: $68.59\% \pm 1.52\%$), breaking the performance ceiling of unimodal tabular classifiers and significantly outperforming traditional static fusion models which suffer from boundary contamination.

---

## 2. Clinical Cohort and Data Collection Protocol

The study cohort was recruited at the European University Tbilisi (Tbilisi, Georgia). After data quality screening and removing entries with incomplete questionnaires, the final pilot cohort consisted of **$N=46$ subjects** divided into two groups:

1. **Non-Vapers ($n=24$):** 16 periodontally healthy, 8 diagnosed with active periodontitis.
2. **Active Vapers ($n=22$):** 7 periodontally healthy, 15 diagnosed with active periodontitis.

For each patient, three primary streams of data (modalities) were collected:

* **Subjective Tabular Questionnaire ($X_{tab}$):** Detailed demographic data (age, gender) alongside clinical risk factors, oral hygiene habits, self-reported symptoms (gum bleeding, tenderness), and specific vaping profiles. The vaping profile recorded:
  * *Vaping Frequency:* Categorized from non-vaper ($0$) to rare (less than once a week) to heavy ($>5$ times/day).
  * *Nicotine Concentration:* E-liquid strengths mapped from $0\text{ mg/ml}$ to $50\text{ mg/ml}$ ($5\%$).
  * *Vaping Duration:* Length of usage in years.
* **Intraoral Clinical Photographs ($s_{soft}$):** High-resolution photographs of the anterior gingival tissues. These photos were evaluated by three independent dental clinicians who rated the visible tissue inflammation on a standardized $1\text{--}10$ consensus scale.
* **Radiographs / Pocket Depth ($s_{hard}$):** Panoramic radiographs evaluating alveolar bone loss and physical pocket depth scores (PPD) across all teeth on a $0\text{--}9$ scale.
  * *Note on Experimental Setup:* To stress-test the gating network's noise rejection capability, the radiographic channel was simulated with a constant uninformative score of $1.0$ across all patients. A correctly functioning gating network was expected to recognize the zero predictive variance in this channel and suppress its weight.

---

## 3. The Clinical Challenge: The "Vaping Paradox"

In a standard, non-vaping clinical subject, tissue inflammation manifests visibly as redness (erythema) and bleeding on probing (BOP) due to microvascular engorgement. Thus, visual photographs are highly diagnostic.

However, nicotine is a potent sympathomimetic agonist that stimulates catecholamine release, binding to $\alpha_1$-adrenergic receptors in the gingival microvasculature to trigger acute vasoconstriction. This vasoconstriction suppresses bleeding and redness, creating a clinical mask. An active vaper with advanced periodontitis can present pink, firm, and seemingly healthy gingiva in a photograph, misleading clinicians and standard AI computer vision models alike.

Traditional multimodal fusion models (such as Late Concatenation or Bilinear Tensor Fusion) apply a static weight to the photographic channel across all patients. Because visual photos are highly predictive for the $24$ non-vapers, the model assigns a positive weight to this channel. When applied to active vapers with periodontitis, the deceptive "healthy-looking" photograph drags down the prediction, resulting in severe false negatives. This is the **boundary contamination mechanism** that causes static fusion to perform worse ($60.9\%$ accuracy) than a simple questionnaire baseline ($65.2\%$).

---

## 4. The Clinical Attention Framework (CAF) Architecture

To resolve this conflict, the CAF introduces **Dynamic Reliability Routing**. Instead of statically merging modalities, it dynamically adjusts the trust allocated to each channel based on the patient's individual clinical context:

1. **Exogenous Gating:** The gating network ingests the patient's vaping habits context vector ($x_{gate} \in \mathbb{R}^3$, representing frequency, nicotine concentration, and duration). It projects this context through a linear gating layer to generate raw attention logits:

   $$
   z = W_{gating} x_{gate} + b_{gating}
   $$
2. **Softmax Attention Allocation:** A Softmax activation is applied over the logits to distribute attention weights competitively:

   $$
   P_m = \text{Softmax}(z)_m, \quad \sum_{m} P_m = 1.0
   $$

   This competitive allocation ensures a zero-sum redistribution of trust: if the gate suppresses the visual channel ($P_{soft} \to 0$), it mathematically amplifies the remaining channels.
3. **Calibrated Fusion & Positivity Constraint:** Aligned risk scores ($s_{tab}$, $s_{soft}$, $s_{hard} \in [0,1]$) are fused and scaled to generate the final diagnostic logit:

   $$
   y_{logit} = scale \cdot \left( P_{tab}s_{tab} + P_{soft}s_{soft} + P_{hard}s_{hard} \right) + bias_{classifier}
   $$

   To prevent optimization collapse and ensure numerical stability, the scale parameter is bound strictly positive using the Softplus function:
   $$
   scale = \text{Softplus}(\gamma) = \log(1 + e^{\gamma})
   $$
4. **Parameter Efficiency:** The entire meta-reasoning fusion layer utilizes only **14 learnable parameters** ($W_{gating} \in \mathbb{R}^{3 \times 3}$, $b_{gating} \in \mathbb{R}^3$, $\gamma$, and the classifier bias). This extremely small parameter footprint satisfies PAC-learning bounds, preventing overfitting on our small clinical cohort ($N=46$).

---

## 5. Experimental Validation Protocol

To ensure clinical and statistical rigor, we implemented the following protocols:

* **Nested Leave-One-Out Cross-Validation (LOOCV):** In each fold, a single patient was held out. Preprocessing, continuous tabular Z-scoring, and ANOVA-based `SelectKBest` feature selection were fitted strictly on the $45$ training patients to guarantee zero target leakage.
* **10-Seed Ensemble:** Within each fold, model outputs were averaged across 10 randomized initializations to eliminate local minima noise.
* **100-Seed Loop:** The entire LOOCV loop was executed across 100 randomized seeds to generate robust means and standard deviations.
* **Baseline Benchmarks:** CAF was compared against a Questionnaire-Only baseline, a Visual-Only baseline, Late Concatenation, a Gated Multimodal Unit (GMU), and Bilinear Tensor Fusion.
* **Pairwise Significance Testing:** Formulated in R, including Welch's t-test, Wilcoxon rank-sum test, Fisher's Exact Test, and McNemar's test.

---

## 6. Key Scientific Results

* **Diagnostic Accuracy:** The consensus ensembled CAF model achieved **$69.57\%$ accuracy** and **$0.7127$ ROC-AUC**, breaking the $65.2\%$ questionnaire ceiling, whereas static late concatenation collapsed to **$60.87\%$** accuracy.
* **Modality Routing & Vasoconstriction Drop:**
  * The mean visual attention weight ($P_{soft}$) was **$0.4946$** for non-vapers, but dropped to **$0.1064$** for active vapers—representing a **$78.5\%$ relative drop** in visual trust.
  * Welch's t-test confirmed the significance of this attention drop ($t = 16.441, df = 29.6, p < 2.2 \times 10^{-16}$) with an astronomical Cohen's $d$ effect size of **$4.99$**.
* **Zero-Variance Noise Pruning:** The simulated radiographic channel ($s_{hard} = 1.0$) was pruned down to a cohort-wide mean attention weight of just **$1.77\%$** ($P_{hard} = 0.0177$), proving the network's capacity to suppress uninformative noise.
* **Fisher's Exact Test (Modality Dismissal):** Setting a threshold of $P_{soft} < 0.25$, the model dismissed the visual channel in $18$ out of $22$ active vapers ($81.8\%$) and $0$ out of $24$ non-vapers ($0\%$). This perfect separation was highly significant ($p = 2.64 \times 10^{-8}$).
* **Continuous Dose-Response Behavior:** The 4 active vaper outliers who retained partial visual attention ($P_{soft} > 0.25$) were clinically verified to have lighter exposure profiles (Patient ID 2 and 19 vaped "rarely, <1 time/week"; Patient ID 27 and 58 vaped daily but had short exposure durations of <1 year). The model autonomously routed trust proportionally to their exposure level, demonstrating continuous biological modeling rather than a binary switch.
* **McNemar's Test (Clinical Superiority):** Pairwise evaluation showed that CAF corrected 4 patient classifications where the baseline failed ($p = 0.0455$), including correct visual suppression for diseased vapers and preventing visual false positives from localized red tissue.
* **Autonomous Discovery:** Crucially, the model was never explicitly trained on nicotine pharmacology or visual vasoconstriction; it independently discovered this routing strategy using classification error minimization alone.

---

## 7. Future Scalability Blueprint

To scale this architecture from a pilot proof-of-concept to a production clinical system, the next phases will replace manual score predictors with automated deep learning encoders:

1. **Automated Modality Encoders:** Train a vision-language model (e.g., BiomedCLIP) or vision transformer (e.g., DINOv2) to extract 512-dimensional embeddings directly from clinical photographs and panoramic X-rays.
2. **High-Dimensional Scaling:** The current design mathematically guarantees scalability without architectural modification: the scalar gate probability ($P_m$) can be directly multiplied by the entire 512-D visual vector.
3. **Biological Prior Transfer:** Freeze the trained gating network as a "biological prior" representing vasoconstrictive masking, transferring it to guide diagnostic training on larger multi-center cohorts.
4. **Generalization:** Generalize the confounder-driven gate to other nicotine-confounded clinical applications, such as post-surgical wound healing monitoring and peripheral cardiovascular perfusion assessments.
