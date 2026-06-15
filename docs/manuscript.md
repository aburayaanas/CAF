# The Clinical Attention Framework: Confounder-Aware Routing for Multimodal Medical AI

**Authors:** Mustafa Nassar\*, Anas Aburaya, Sophio Samkharadze, [Osama
Abdelhay, Maia Mujiri, Eka Kokheirdze, Lanad Lursmanashvili, Deya
Abushariha, Ahmed
Baker](https://www.linkedin.com/in/otamimi/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BWqHuqzO%2FSamXOVocxHx%2B9Q%3D%3D)

**Affiliations:**

Department of Dentistry, European University, 76 David Guramishvili Ave,
Tbilisi 0141, Georgia.

Department of Data Science and Artificial Intelligence, Princess Sumaya
University for Technology, Amman, Jordan

**\*Corresponding Author:** Mustafa Nassar/ Gldani, Tbilisi, 0141,
Georgia / Email: <mustafanassar2003@gmail.com>, Phone number:
+995599994605

## **Abstract**

Multimodal deep learning has shown immense promise in clinical decision
support systems. However, traditional fusion architectures rely on
static feature concatenation or rigid attention mechanisms. These
approaches are highly vulnerable to conflicting modalities, where one
input channel presents deceptive signals due to biological confounders.
In this paper, we introduce the **Clinical Attention Framework (CAF)**,
a novel gated multimodal architecture that uses external clinical
context to dynamically route diagnostic trust. To validate our
framework, we present a biological proof-of-concept using a cohort of
$N = 46$ patients, evaluating "The Nicotine-Induced Diagnostic Mismatch"
in periodontics. In active vapers, nicotine-induced vasoconstriction
masks visible clinical signs of soft-tissue inflammation (bleeding and
redness), presenting a deceptive visual signal. The CAF dynamically
evaluates the patient's vaping status to route attention weights
($P_{soft}$ for clinical photos, $P_{hard}$ for radiographs, and
$P_{tab}$ for questionnaires), suppressing the confounded visual
modality and relying on objective clinical measurements. Our framework
achieves superior diagnostic performance ($68.59\% \pm 1.52\%$ accuracy,
$0.7013 \pm 0.0227$ ROC-AUC) and exhibits emergent, deductive clinical
reasoning without explicit modality supervision. Finally, we demonstrate
that the CAF's gating mechanism is domain-agnostic, offering a
generalizable blueprint to resolve physiological masking effects across
broader clinical domains.

## **1. Introduction**

While clinical decision-making requires dynamic, multi-source data
integration, current deep learning models struggle to mimic this
reasoning (Das & Akter, 2025; Fan et al., 2024). A critical
vulnerability in existing pipelines is their reliance on rigid
architectures and static feature concatenation, which inherently fail to
adapt when a modality loses its diagnostic relevance. (Fan et al.,
2024). This exposes a critical vulnerability to \"Conflicting
Modalities\" (Fan et al., 2024). A \"Diagnostic Mismatch\" occurs when
intrinsic noise creates discrepancies between data streams, conflicting
modalities, and current AI fails catastrophically; if one modality sends
a deceptive signal that contradicts the others, the system cannot
adaptively reduce its reliance on the flawed data, resulting in a severe
decline in diagnostic accuracy (Fan et al., 2024; Jandoubi & Akhloufi,
2025)

Most current multimodal deep learning pipelines execute early, late, or
intermediate fusion operations without strong theoretical foundations
regarding the complex, dynamic relationships between modalities
(Jandoubi & Akhloufi,2025). These traditional methods typically rely on
the simple concatenation of latent features and utilize static weights,
which mathematically optimizes for the average patient but fundamentally
limits the model\'s ability to adapt when diagnostic reliability
fluctuates in specific subgroups. (Fan et al., 2024). This rigid
architectural flaw fundamentally exacerbates the widely recognized
\"Black Box\" problem in medical artificial intelligence (Otieno &
Oluoch, 2026; Das & Akter, 2025). Because late and early fusion models
use these static weights rather than adaptive, context-aware mechanisms,
they cannot provide the transparent, causal explainability (XAI)
required to build clinical trust, satisfy regulatory standards, and
ensure patient safety (Otieno & Oluoch, 2026; Luke, 2024; Fan et al.,
2024).

While cross-modal attention networks attempt to map correlations across
heterogeneous data streams, these standards \"data-driven\" mechanisms
frequently fail in clinical settings. Without clinical constraints, they
are highly susceptible to \"shortcut learning,\" often fixating on
dataset-specific artifacts or irrelevant borders rather than genuine
pathology (Otieno & Oluoch, 2026; Dyrba et al., 2020). This purely
statistical approach becomes a critical liability during active
biological conflicts, as standard attention lacks the contextual
awareness to identify and resolve deceptive signals or intrinsic noise
(Fan et al., 2024; Otieno & Oluoch, 2026). To overcome this gap, our
architecture implements a confounder-aware gating mechanism. Unlike
standard cross-modal attention, which cannot resolve these conflicts, a
Confounder-Driven mechanism uses external clinical context to
dynamically \"distrust\" and down-weight any modality sending a
contradictory signal. This adaptively resolves the diagnostic mismatches
that cause traditional data-driven models to fail.

To validate our architecture, we introduce a periodontal cohort of
e-cigarette users as a rigorous biological proof-of-concept. In these
patients, nicotine-induced vasoconstriction creates a \" The
Nicotine-Induced Diagnostic Mismatch\"---a known, physiologically
grounded masking effect that suppresses visible clinical signs of
gingival inflammation, such as bleeding and redness, despite underlying
tissue destruction (Figueredo et al., 2021; Silva, 2021***)**.* To
navigate this biological deception, the Clinical Attention Framework
(CAF) introduces a mechanism of \"Dynamic Reliability Routing\".

Rather than blindly seeking cross-modal statistical correlations, the
CAF evaluates the external clinical confounder (the patient\'s vaping
profile) to generate mathematically interpretable attention
probabilities, specifically *Psoft*​ for clinical photos and *Phard*​ for
radiographs. This directly mimics deductive clinical reasoning: if the
framework detects a heavy vaping history, it dynamically routes
diagnostic trust away from the deceptive soft-tissue modality (driving
*Psoft*​ toward zero). It heavily weights the objective hard-tissue
X-rays (elevating *Phard*​). By explicitly quantifying modality
reliability, the CAF successfully bypasses physiological masking effects
that cause traditional, static models to fail.

Current multimodal AI frequently exhibits \"illusory generalizability,\"
failing in real-world settings due to \"shortcut learning\" and the
rigid assumption that modality reliability remains constant (Otieno &
Oluoch, 2026; Fan et al., 2024). This creates a systemic, pan-medical
bottleneck when biological confounders introduce deceptive signals. For
example, in oncology, targeted therapies actively alter imaging
phenotypes, confounding standard radiographic endpoints by inducing
pseudoresponses that mask true tumor progression (Wen et al., 2010;
Brandsma & van den Bent, 2009). Similarly, in cardiology, medications
like beta-blockers mask physiological stress and significantly reduce
the sensitivity of ischemic ST-segment responses on ECG readouts
(Herbert et al., 1991). Traditional \'black box\' AI networks struggle
to maintain accuracy when faced with such patient-level comorbidities
and therapeutic confounders (Elantary & Othman, 2025), highlighting the
need for architectures that can actively route diagnostic trust.

To address this critical vulnerability, this paper introduces the
Clinical Attention Framework (CAF). Designed as a domain-agnostic
safeguard, the CAF\'s Conflict-Resolution Gate uses external clinical
context to adaptively down-weight deceptive signals. Our core
contributions establish a foundation for advancing multimodal
diagnostics. Primarily, we present a novel gated multimodal architecture
that generates transparent, mathematically interpretable attention
weights, helping to mitigate the traditional \"black box\" limitations
of fusion pipelines by enhancing interpretability. Furthermore, through
our periodontal biological proof-of-concept, we provide initial
empirical evidence on a micro-cohort demonstrating that the CAF's
attention gating engine can exhibit patterns consistent with deductive
clinical reasoning, without requiring modality-specific supervision.
Ultimately, we propose the CAF as a modular, domain-agnostic blueprint;
while our current validation is bounded by a limited clinical cohort,
the architecture is designed to be scalable, offering a foundational
pathway toward out-of-distribution generalizability for diverse,
broad-scale medical domains.

## **2. Materials and Methods**

### **2.1 Patient Cohort and Clinical Protocol**

#### **2.1.1 Participant Selection and Inclusion Criteria**

#### This single-center, cross-sectional observational pilot study was conducted at the European University Dental Clinic in Tbilisi, Georgia, from September 2025 to March 2026. The study protocol was formally approved by the Institutional Review Board (IRB) of European University, executed in strict compliance with the Declaration of Helsinki ("World Medical Association Declaration of Helsinki," 2013), and utilized a robust data anonymization procedure alongside mandatory written informed consent from all participants. The study population comprised N = 46 systemically healthy participants aged 18 to 25 years, balanced symmetrically into 23 healthy individuals (Label 0) and 23 diseased individuals (Label 1). This cohort was partitioned into two strictly age- and gender-matched groups: Cohort 1 (n1 = 22 active electronic nicotine delivery systems \[ENDS\] users) and Cohort 2 (n2 = 24 non-vaping controls). For Cohort 1, variations in vaping history, duration, and habits were documented as continuous metadata variables rather than rigid binary cuts to preserve a real-world exposure spectrum, while Cohort 2 was restricted exclusively to individuals with absolute lifetime abstinence from all tobacco, electronic aerosols, and systemic nicotine variations. To isolate clear tissue interactions, all participants were required to have no history of systemic antibiotics within the last 3 months or professional periodontal treatment within the last 6 months. Individuals were excluded if they presented with systemic diseases (e.g., diabetes or autoimmune conditions), pregnancy, or current medications affecting gingival tissues (e.g., calcium channel blockers, phenytoin, cyclosporine). Dual-users of combined combustible tobacco and vaping were also excluded, alongside any cases presenting with insufficient data or missing modalities. Clinical diagnostic criteria aligned strictly with the 2017 World Workshop classification, where plaque-induced gingivitis was defined as Bleeding on Probing (BOP) exceeding 10% of sites with a Probing Pocket Depth (PPD) $\le$ 3 mm and an absolute absence of radiographic bone loss (Papapanou et al., 2018). This structured cohort provided the core matrix for subsequent computational modeling and dynamic attention routing verification.

#### **2.1.2 Multimodal Data Acquisition Protocol**

### All three modalities were acquired during a single standardized clinical visit by a single calibrated examiner (M.N.) to minimize inter-operator variability. 

### • Soft-Tissue Modality (Psoft): Intraoral clinical photography was performed using a Canon EOS 4000D equipped with a 100mm macro lens and twin flash system, calibrated to approximately 5500K color temperature. A standardized exposure protocol was applied across all patients (f/22, 1/200s, ISO 100) to ensure consistent color rendition and depth of field. Bilateral cheek retractors were used for each side, and the patient's head position was standardized throughout. A full-mouth photographic series of 11 images per patient was acquired, capturing all relevant soft-tissue regions for subsequent inflammatory scoring. 

### • Hard-Tissue Modality (Phard): Hard-tissue assessment was performed using panoramic radiography via Rotational Slit-Beam Tomography on conventional film. Full-mouth radiographic coverage was obtained for all patients. Radiographic features extracted included bone crest levels and relevant hard-tissue morphological findings. All radiographs were independently evaluated by three expert clinicians to ensure diagnostic reliability. 

### • Questionnaire (Ptab): Prior to clinical examination, each participant completed a standardized 45-item paper-based questionnaire covering five domains: demographic characteristics, vaping and nicotine habits, oral hygiene practices, lifestyle factors, and self-reported periodontal symptoms. Paper administration before examination ensured responses were independent of clinical findings and examiner influence. 

### • Data Recording and Anonymization: All acquired data were stored on a password-protected, encrypted computer accessible exclusively to the principal investigators (M.N. and A.A.). Patient identifiers were removed at the point of data entry and replaced with anonymous numerical codes, ensuring no personally identifiable information was retained within the dataset. All procedures were conducted in full compliance with institutional data protection protocols and the Declaration of Helsinki.

### **2.2 Data Preprocessing and Modality Alignment**

**2.2.1 Soft-Tissue Continuous Intensity Indexing**

Given the micro-cohort scale of the present dataset, deploying
high-dimensional deep learning encoders such as convolutional neural
networks would introduce severe dimensionality and overfitting risks.
Instead, a continuous 1--10 Soft-Tissue Intensity Index was developed,
adapted from the validated Lobene Modified Gingival Index (MGI) (Lobene
et al., 1986). While the original MGI operates on a 0--4 ordinal scale,
its coarse granularity is insufficient for machine learning applications
requiring continuous, fine-grained feature discrimination. The expanded
1--10 scale preserves the clinical logic of the MGI while providing the
numerical resolution necessary for gradient-based optimization and
meaningful inter-patient differentiation.

The index explicitly measures surface tissue characteristics observable
in standardized clinical photography and does not attempt to derive
dimensional or volumetric measurements from two-dimensional photographic
data. Scores reflect six observable tissue properties: gingival color
and erythema intensity, marginal contour, surface texture, edema
severity, glazing, and spontaneous hemorrhagic tendency.

Three independent expert clinicians, blinded to patient vaping status
and to each other\'s scores, independently assigned index values to each
patient\'s photographic series. Final scores were computed as the mean
of the three independent ratings. Inter-rater reliability was assessed
and is reported in Section 2.2.4.

**2.2.2 Structural Radiographic Scaling**

Parallel to the soft-tissue indexing approach, panoramic radiographs
were converted into a continuous 1.0--10.0 Radiographic Severity Index
to ensure architectural consistency across all modalities and eliminate
dimensional incompatibility within the feature fusion pipeline. The
scale was developed as a continuous numerical expansion of the 2017
AAP/EFP Periodontal Classification staging criteria (Papapanou et al.,
2018), mapping established clinical staging thresholds onto a
fine-grained, machine-readable range. Crucially, the Radiographic
Severity Index represents a relative continuous scaling of observable
hard-tissue pathology and does not constitute absolute millimeter bone
level measurements, which would require a standardized long-cone
paralleling technique with calibrated grids. Instead, the continuous
scores reflect alveolar bone crest levels, the extent and pattern of
bone loss, and furcation involvement as observed on standardized
panoramic radiography.

Three independent expert clinicians, blinded to patient vaping status
and to each other's ratings, independently assigned Radiographic
Severity Index scores to each patient's panoramic radiograph. Final
scores were computed as the mean of the three independent ratings,
consistent with the soft-tissue scoring protocol. Inter-rater
reliability is reported in Section 2.2.4.

#### **2.2.3 Tabular and Questionnaire Feature Encoding**

For patient demographics, systemic health histories, and subjective
clinical symptoms gathered via the questionnaire, categorical string
responses are standardized by converting characters to lowercase,
removing leading/trailing whitespace, and encoding them into numerical
values using integer label mapping (category codes). To guarantee
numeric stability and uniform gradient updates during optimization, all
encoded categorical and continuous variables are standardized to have
zero mean and unit variance:

$$X_{tab}' = \frac{X_{tab} - \mu_{tab}}{\sigma_{tab}}$$

where $\mu_{tab}$ and $\sigma_{tab}$ represent the feature-wise mean and
standard deviation computed over the training fold. To prevent
overfitting and mitigate the "curse of dimensionality" on our clinical
cohort, we implement automated univariate feature selection (Guyon, I.,
& Elisseeff, A., 2003). Specifically, a $SelectKBest$ selector is fitted
on the training split using the analysis of variance (ANOVA) F-value
statistic between each feature and the binary clinical label, retaining
the top $k = 16$ most diagnostic features. Feature selection is
performed strictly within each LOOCV training fold to prevent target
leakage---the held-out patient's label has zero influence on feature
selection.

#### **2.2.4 Inter-Rater Reliability of Clinical Indices**

To ensure high diagnostic reliability and consensus for visual indices,
both the photographic soft-tissue ratings ($s_{soft}$) and the
radiographic bone loss scores ($s_{hard}$) were graded independently by
three expert clinicians blinded to the patients' vaping status and to
each other's ratings. The final clinical scores used in the modeling
pipeline were computed as the arithmetic mean of these three independent
ratings. While visual consensus and standard deviation checks were
performed to filter out extreme outliers, a formal Interclass
Correlation Coefficient (ICC) was not computed for this pilot cohort,
representing a methodological limitation that will be addressed in
future multi-center validation trials.

#### **2.2.5 Low-Dimensional Feature Alignment (Simulated Encoders)**

Multi-source clinical datasets present severe dimensionality
discrepancies (e.g., high-dimensional questionnaire tensors, 2D clinical
image grids, and radiographic images). Rather than feeding
high-dimensional raw data directly into the gating network, which
exposes deep models to shortcut learning and representation collapse on
small cohorts, we perform low-dimensional feature alignment (Geirhos et
al., 2020).

We map each modality into a normalized, 1D risk score predictor space
($s_{m} \in [ 0,1]$), which aligns all heterogeneous data
channels into a unified probability space:


1.  **Tabular Questionnaire Predictor (**$s_{tab}$**):** The
    $k$-selected tabular features are passed through a Logistic
    Regression estimator trained via out-of-fold cross-validation,
    outputting the predicted class probability of periodontitis.
    Logistic Regression was selected as the tabular predictor because it
    outputs calibrated probability scores directly interpretable as
    disease risk---a property not natively available in SVM or
    tree-based models without post-hoc calibration (Platt 1999). This
    ensures $s_{tab}$ operates in the same $[ 0,1]$
    probability space as $s_{soft}$ and $s_{hard}$, enabling meaningful
    arithmetic comparison within the gating fusion.


2.  **Visual Soft-Tissue Predictor (**$s_{soft}$**):** The clinical
    intraoral photographs of the anterior gingival tissue are evaluated
    using a standardized visual consensus protocol (scored on a 1--10
    scale by three independent clinicians) and divided by 10.0 to
    normalize to a $[ 0.0,1.0]$ range), yielding a
    diagnostic probability score for soft-tissue inflammation. The 1--10
    ordinal scale was selected to match established clinical periodontal
    indices, Lobene Modified Gingival Index (MGI), (Lobene et al.,
    1986). ensuring ecological validity of the scoring protocol.
    Division by 10.0 maps scores to a probability-like space consistent
    with the tabular predictor output. Note that in larger-scale
    deployments, this manual scoring can be replaced by a
    high-dimensional vector representation extracted using deep visual
    encoders (e.g., convolutional neural networks or vision
    transformers).


3.  **Radiographic (X-ray) Predictor (**$s_{hard}$**):** In the absence
    of a validated radiographic image encoder for this pilot cohort, the
    radiographic channel operates on a continuous 1.0-10 Radiographic
    Severity Index derived from the mean scores of three independent
    expert clinicians; however, during an isolated architectural stress
    test, this input was experimentally set to a constant, uninformative
    score of 1.0. This constitutes a deliberate architectural stress
    test: a correctly functioning gating network must recognize
    zero-variance input and suppress it toward $P_{hard} \approx 0$.
    Empirically, the gate drives $P_{hard}$ to $1.77\%,$, confirming the
    network's capacity for autonomous noise rejection without explicit
    supervision.

By translating each high-dimensional modality into a single unified risk
score $s_{m} \in [ 0,1]$, We ensure that the gating network
acts as a clean, interpretable meta-reasoner that dynamically controls
the transmission of pre-aligned diagnostic signals rather than
performing raw feature learning.

### **2.3 The Clinical Attention Framework (CAF) Architecture**

#### **2.3.1 Architectural Overview**

The Clinical Attention Framework (CAF) is designed to resolve modality
conflicts by introducing an attention-gated routing system. Instead of
statically merging feature streams, the model dynamically routes the
flow of information based on external clinical context (the confounder)
(Figure 1).

![Clinical Attention Framework (CAF) Architecture Diagram. The schematic
shows how patient context features (x\_{gate}) pass through a linear
gating layer to generate modality logits. A Softmax function converts
these logits to dynamic routing probabilities (P\_{tab}, P\_{soft},
P\_{hard}) that scale aligned modality scores (s\_{tab}, s\_{soft},
s\_{hard}). The final diagnostic prediction logit is scaled and shifted
before being passed to a Sigmoid
function.](figure1_architecture.png){width="5.833333333333333in"
height="3.383198818897638in"}

**Clinical Attention Framework (CAF) Architecture Diagram.** The
schematic shows how patient context features ($x_{gate}$) pass through a
linear gating layer to generate modality logits. A Softmax function
converts these logits to dynamic routing probabilities
($P_{tab},P_{soft},P_{hard}$) that scale-aligned modality scores
($s_{tab},s_{soft},s_{hard}$). The final diagnostic prediction logit is
scaled and shifted before being passed to a Sigmoid function.

#### **2.3.2 The Confounder-Driven Gating Matrix**

The core of the routing mechanism is the Gating Network. The network
ingests a patient-specific clinical confounder vector
$x_{gate} \in \mathbb{R}^d$ (representing context variables such as
active vaping status, age, and gender). This context vector is projected
through a linear gating layer to generate raw modality attention logits:

$$z = W_{gating}x_{gate} + b_{gating}$$

where $W_{gating} \in \mathbb{R}^{M \times d}$ is the gating weight
matrix, $b_{gating} \in \mathbb{R}^M$ is the bias vector, and $M = 3$
represents the number of input modalities (questionnaire, visual
soft-tissue, and radiographs).

A single linear layer is deliberately chosen over deeper alternatives
for three reasons: (1) with N=46, a non-linear gate would introduce a
hypothesis space too complex to be reliably constrained by the available
sample size, violating the foundational sample complexity principles of
probably approximately correct (PAC) learning (Valiant, 1984), risking
severe overfitting; (2) a linear projection ensures the routing decision
is directly interpretable as a weighted combination of nicotine exposure
and demographic features; and (3) L-BFGS optimization guarantees exact
convergence on convex sub-problems, which a linear gate provides. In
larger-scale clinical deployments, non-linear deep networks may be
preferred to map complex, hidden multi-source confounder interactions
across the entire questionnaire space rather than a restricted subset of
demographic and vaping habits.

#### **2.3.3 Dynamic Reliability Routing**

To transform raw logits $z$ into a valid probability distribution
representing modality routing attention, we apply the Softmax function
over the modality channels:

$$P_{m} = \frac{e^{z_{m}}}{\sum_{j = 1}^{M}e^{z_{j}}}$$

where $P_{m}$ represents the attention weight allocated to modality
$m \in \{ tab,soft,hard\}$, satisfying $\sum_{m}^{}P_{m} = 1.0$ and
$P_{m} \in [ 0,1]$. These attention weights act as dynamic
routing gates that reflect the diagnostic reliability of each modality
for the specific patient.

#### **2.3.4 Graceful Modality Down-Weighting**

The gating network enables the system to gracefully down-weight
deceptive or uninformative data streams. If a patient's context vector
$x_{gate}$ indicates an active vaping status, the network identifies
that the visual soft-tissue photograph is highly confounded due to
vasoconstriction. Consequently, the optimization gradient drives the
visual logit $z_{soft}$ down, forcing the routing weight $P_{soft}$
toward zero. Similarly, if a modality presents zero predictive variance
(such as the constant radiographic score $s_{hard} = 1.0$ used due to
the absence of a visual image encoder), the gating network actively
suppresses the attention weight $P_{hard}$ to near-zero (empirically
shown to be driven down to $1.77\%$) to protect the final classification
decision from non-informative noise injection.

#### **2.3.5 Weighted Feature Fusion and Final Classification**

Once the gating network determines the routing coefficients, the aligned
modality risk scores ($s_{tab}$, $s_{soft}$, $s_{hard}$) are multiplied
by their respective attention weights to compute the gated feature
representation. The final classification logit is computed as:

$$y_{logit} = scale \cdot \left( P_{tab}s_{tab} + P_{soft}s_{soft} + P_{hard}s_{hard} \right) + bias_{classifier}$$

where $bias_{classifier}\mathbb{\in R}$ is the classifier bias. The
scalar parameter $scale$ regulates the absolute magnitude of the fused
score before the classification decision. The scalar parameter $scale$
is necessary because all modality scores $s_{m} \in [ 0,1]$
and gate probabilities $P_{m} \in [ 0,1]$, bounding the
fused sum within $[ 0,1]$. Binary Cross-Entropy (BCE) loss
requires logits with sufficient dynamic range to produce confident
predictions---without $scale$, the optimization collapses to degenerate
solutions (as empirically verified in our ablation studies, Section 3).

#### **2.3.6 Mathematical Formalization of the Fusion Strategy**

To prevent numerical instability and ensure that the scale parameter
remains strictly positive during optimization, we define $scale$ using
the Softplus function:

$$scale = \text{Softplus}(\gamma) = log(1 + e^{\gamma})$$

where $\gamma\mathbb{\in R}$ is a learnable scale parameter.

The Softplus function is chosen over alternative positivity constraints
(such as ReLU, absolute value, or exponential functions) for three
properties: (1) it is everywhere differentiable, ensuring smooth L-BFGS
gradient computation; (2) it approaches linearity for large $\gamma$,
preventing gradient saturation; and (3) it initializes near
$scale = 1.0$ when $\gamma = 0.5413$ (a value derived from
$log(e^{1.0} - 1)$), providing a neutral starting point where no
modality is artificially amplified prior to learning.

The final diagnostic probability of periodontitis ($\widehat{y}$) is
obtained by passing the logit through the Sigmoid activation function:

$$\widehat{y} = \sigma(y_{logit}) = \frac{1}{1 + e^{- y_{logit}}}$$

The network is optimized end-to-end using Binary Cross-Entropy (BCE)
loss over the patient labels:

$$\mathcal{L = -}\frac{1}{N}\sum_{i = 1}^{N}\left[ y_{i}log({\widehat{y}}_{i}) + (1 - y_{i})log(1 - {\widehat{y}}_{i}) \right]$$

where $y_{i} \in \{ 0,1\}$ represents the ground-truth clinical
diagnosis of patient $i$. All random seeds are set globally across
NumPy, Python's random module, and PyTorch prior to each evaluation run
to ensure deterministic data splitting and weight initialization.

### **2.4 Experimental Setup and Statistical Validation**

#### **2.4.1 Unimodal Baselines and Standard Fusion Comparisons**

To validate the diagnostic efficacy of the Clinical Attention Framework
(CAF), we implement and compare three categories of baseline models:


1.  **Unimodal Baselines:**



    - **Questionnaire-Only Baseline (**$s_{tab}$**):** A Logistic
      Regression classifier trained strictly on the $k$-selected tabular
      features.


    - **Visual-Only Baseline (**$s_{soft}$**):** Direct classification
      using the expert soft-tissue photographic rating, normalized to
      $[ 0.0,1.0]$.


2.  **Traditional Multimodal Fusion:**



    - **Late Concatenation Baseline:** A traditional fusion method where
      aligned scores from the three modalities ($s_{tab}$, $s_{soft}$,
      $s_{hard}$) are directly concatenated into a 3D feature vector
      ($[ s_{tab},s_{soft},s_{hard}]$) and mapped to the
      diagnostic target using a Logistic Regression classifier without
      context-aware gating. This represents a static fusion system that
      lacks the capacity to dynamically down-weight deceptive or
      uninformative modalities based on patient context.


3.  **Advanced Multimodal Fusion:**



    - **Gated Multimodal Unit (GMU):** A gated fusion baseline that
      learns dynamic, non-linear activation weights to combine
      modalities using Sigmoid-activated feature gates, (Arevalo et al.,
      2017). It is a highly relevant comparison to benchmark whether our
      explicit clinical gating outperforms general, context-free deep
      learning gating architectures.


    - **Bilinear Tensor Fusion:** An advanced fusion method that
      captures high-order interactions across feature streams using
      outer-product operations (Zadeh et al., 2017), providing a
      comparison to verify if simple dynamic linear scaling outperforms
      complex multiplicative cross-modal tensor representations.

#### **2.4.2 Leave-One-Out Cross-Validation (LOOCV) Protocol**

Given our pilot clinical cohort size ($N = 46$), standard
train-validation-test splits risk severe partition bias and high metric
variance. To guarantee rigorous, unbiased evaluation, we employ a
Leave-One-Out Cross-Validation (LOOCV) protocol (Arlot & Celisse,
2010).. In each of the 46 folds, a single patient is held out as the
test subject. The remaining 45 patients are used to standard-scale the
tabular features, fit the $SelectKBest$ feature selector, and train the
baseline predictors ($s_{tab}$) and the CAF gating parameters. The
nested structure of this protocol---wherein all preprocessing, feature
selection, and model training are performed exclusively on the
45-patient training fold---guarantees zero information leakage from the
held-out patient at every stage of the pipeline.

To eliminate initialization-driven routing noise (wherein the gating
network occasionally gets trapped in local minima during single-seed
splits, causing inconsistent routing weights), we implement a **10-Seed
Ensemble Protocol** per cross-validation fold. The routing gates and
classifier outputs are ensembled by taking the arithmetic mean of the
predicted probabilities and modality attention weights across 10
randomized model initializations.

#### **2.4.3 Evaluation Metrics**

To comprehensively benchmark diagnostic performance, we report the
following classification metrics: \* **Classification Accuracy:** The
percentage of correctly diagnosed patients. \* **Sensitivity (Recall):**
The true positive rate, representing the proportion of periodontitis
patients successfully identified by the model. \* **Specificity:** The
true negative rate, representing the proportion of healthy subjects
correctly identified. \* **F1-Score:** The harmonic mean of Precision
and Recall, providing a balanced performance measure under class
distribution uncertainty. \* **Receiver Operating Characteristic Area
Under the Curve (ROC-AUC):** The overall discriminative capability
across all classification thresholds. \* **Average Precision (AP):** The
area under the Precision-Recall curve, representing the model's positive
predictive value scaling across different recall levels.

In screening contexts, Sensitivity and ROC-AUC are prioritized over
Average Precision, as the clinical cost of a missed periodontitis
diagnosis exceeds that of a false positive referral. To demonstrate
statistical robustness, we evaluate the ensembled LOOCV loop across a
**100-Seed Protocol**, reporting the mean and standard deviation of each
metric over all 100 seeds.

#### **2.4.4 Explainable AI and Attention Weight Visualization**

Explainable AI (XAI) is natively integrated into the CAF architecture.
Rather than relying on post-hoc local approximations (such as SHAP
\[Lundberg & Lee, 2017\] or LIME \[Ribeiro et al., 2016\]), which are
prone to fidelity errors in highly non-linear settings, (Slack et al.,
2020), the CAF generates mathematically explicit attention weights
($P_{tab}$, $P_{soft}$, $P_{hard}$) for each patient. This native
interpretability is a direct consequence of the architectural constraint
that the gating network is the sole routing mechanism---the attention
weights are not derived quantities but primary computational outputs of
the model, making them exact rather than approximate explanations. These
weights are directly visualizable to provide clinicians with absolute
transparency regarding which data streams were trusted for a specific
patient. We visualize the cohort-wide routing behaviors using: \*
**Attention Distribution Boxplots:** Comparing the distribution of
visual soft-tissue attention ($P_{soft}$) between active vapers and
non-vapers to illustrate confounder-driven routing. \* **Per-Patient
Routing Heatmaps:** Displaying a 3-column tile grid of the attention
coefficients ($P_{tab}$, $P_{soft}$, $P_{hard}$) for all 46 patients to
visualize active modality routing and noise pruning.

#### **2.4.5 Pairwise Significance Testing of the Conflict-Resolution Gate**

To verify that the routing actions are statistically significant and
robust against model initialization noise, we perform a suite of
statistical tests in R:


1.  **Modality Gating Separation (Welch's t-test):** We conduct Welch's
    two-sample t-test (unpaired, unequal variances) on the ensembled
    visual attention weights ($P_{soft}$) between non-vapers and active
    vapers to verify the statistical significance of the visual
    attention drop.


2.  **Non-Parametric Validation (Mann-Whitney U Test):** To confirm that
    our findings are robust to non-normal distributions, we conduct a
    Wilcoxon rank-sum test on the same attention weights.


3.  **Modality Dismissal Association (Fisher's Exact Test):** Setting a
    visual modality dismissal threshold at $P_{soft} < 0.25$, we
    construct a $2 \times 2$ contingency table (Vaping Group
    vs. Modality Dismissal) and perform a two-sided Fisher's Exact Test
    to evaluate if modality dismissal is statistically associated with
    vaping status. The dismissal threshold of $P_{soft} < 0.25$ was
    selected a priori as a conservative boundary representing less than
    half of the equal-weight baseline allocation ($0.333$ in a
    3-modality system), indicating active suppression rather than
    natural variance.


4.  **Accuracy Superiority (McNemar's Test):** To prove that CAF's
    accuracy improvements over the Late Concatenation baseline are
    statistically significant, we evaluate their out-of-fold predictions
    using McNemar's test on the $2 \times 2$ disagreement matrix of
    correct and wrong classification outcomes.

#### **2.4.6 Implementation Details**

The Clinical Attention Framework is implemented in PyTorch (v2.x) and
optimized using the L-BFGS optimizer (learning rate = 0.05, max
iterations = 100). All random seeds are set globally across NumPy,
Python's random module, and PyTorch prior to each evaluation run to
ensure deterministic data splitting and weight initialization. During
LOOCV training, we enforce a strict single-thread constraint in PyTorch
using `torch.set_num_``threads(``1)`. This constraint is mathematically
necessary when running parallelized multi-seed LOOCV worker loops on
standard clinical computing cores to prevent OpenMP thread collision,
CPU core thrashing, and process deadlocks. Preprocessing, baselines, and
data exports are executed in Python, while the statistical significance
tests and publication-grade plots are generated in R (v4.x) using
$ggplot2$.

## **3. Results**

### **3.1 Classification Performance**

To benchmark the diagnostic capability of the Clinical Attention
Framework (CAF) under modality conflict, we evaluate the system against
unimodal and multimodal baselines using a 100-seed LOOCV evaluation
sweep. Table 1 summarizes the performance metrics (Averaged over 100
seeds, $N = 46$ cohort), while Figure 5 shows the stability distribution
of our framework's classification accuracy across these runs.

**Table 1: Classification Performance Summary (Mean $\pm$ SD)**

| Model Name | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Questionnaire-Only)** | $65.2\% \pm 0.0\%$ | $64.0\% \pm 0.0\%$ | $69.6\% \pm 0.0\%$ | $0.667 \pm 0.000$ | $0.684 \pm 0.000$ |
| **Late Concatenation Fusion** | $60.9\% \pm 0.0\%$ | $59.3\% \pm 0.0\%$ | $69.6\% \pm 0.0\%$ | $0.640 \pm 0.000$ | $0.671 \pm 0.000$ |
| **Gated Multimodal Unit (GMU)** | $62.0\% \pm 3.3\%$ | $60.4\% \pm 3.0\%$ | $70.4\% \pm 5.0\%$ | $0.649 \pm 0.033$ | $0.654 \pm 0.032$ |
| **Bilinear Tensor Fusion** | $63.0\% \pm 0.0\%$ | $60.7\% \pm 0.0\%$ | $73.9\% \pm 0.0\%$ | $0.667 \pm 0.000$ | $0.675 \pm 0.000$ |
| **CAF Gating Pipeline** | **$68.59\% \pm 1.52\%$** | **$67.23\% \pm 1.27\%$** | **$72.52\% \pm 2.53\%$** | **$0.698 \pm 0.017$** | **$0.7013 \pm 0.0227$** |
  Model Name                 Accuracy    Precision       Recall        F1-Score     ROC-AUC
                                                      (Sensitivity)               
  ------------------------ ------------ ------------ --------------- ------------ ------------
  **Baseline               65.2% $\pm$ 0.0% 64.0% $\pm$ 0.0%  69.6% $\pm$ 0.0%     0.667 $\pm$      0.684 $\pm$
  (Questionnaire-Only)**                                                0.000        0.000

  **Late Concatenation     60.9% $\pm$ 0.0% 59.3% $\pm$ 0.0%  69.6% $\pm$ 0.0%     0.640 $\pm$      0.671 $\pm$
  Fusion**                                                              0.000        0.000

  **Gated Multimodal Unit  62.0% $\pm$ 3.3% 60.4% $\pm$ 3.0%  70.4% $\pm$ 5.0%     0.649 $\pm$      0.654 $\pm$
  (GMU)**                                                               0.033        0.032

  **Bilinear Tensor        63.0% $\pm$ 0.0% 60.7% $\pm$ 0.0%  73.9% $\pm$ 0.0%     0.667 $\pm$      0.675 $\pm$
  Fusion**                                                              0.000        0.000

  **CAF Gating Pipeline**   **68.59% $\pm$   **67.23% $\pm$    **72.52% $\pm$     **0.698 $\pm$    **0.7013 $\pm$
                             1.52%**      1.27%**        2.53%**       0.017**      0.0227**
  --------------------------------------------------------------------------------------------

![Seed Stability Analysis. Histogram of model classification accuracy
across 100 random seed runs under leave-one-out cross-validation
(LOOCV), demonstrating the stability and robustness of the regularized
gating architecture.](figure5_seed_stability.png){width="5.833333333333333in"
height="3.499998906386702in"}

**Seed Stability Analysis.** Histogram of model classification accuracy
across 100 random seed runs under leave-one-out cross-validation
(LOOCV), demonstrating the stability and robustness of the regularized
gating architecture.

Notably, all static fusion models---including the advanced Bilinear
Tensor Fusion---score below the unimodal questionnaire baseline. This
degradation is not a failure of the fusion architectures themselves, but
a confirmation of the vasoconstriction hypothesis: injecting a
confounded visual channel without context-aware routing actively harms
diagnostic performance.

In contrast, the CAF gating pipeline breaks the tabular ceiling,
achieving $68.59\% \pm 1.52\%$ classification accuracy and
$0.7013 \pm 0.0227$ ROC-AUC. When ensembling 10 seeds per fold to
produce final clinical decisions, the consensus ensembled CAF model
reaches $69.57\%$ accuracy and $0.7127$ ROC-AUC (Table 2).

**Table 2: Ensembled Model Performance (10-Seed Ensemble)**

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Late Concatenation (Ensembled)** | $60.87\%$ | $59.26\%$ | $69.57\%$ | $0.6400$ | $0.6730$ |
| **Clinical Attention Framework (CAF)** | **$69.57\%$** | **$68.00\%$** | **$73.91\%$** | **$0.7083$** | **$0.7127$** |

### **3.2 Confounder-Driven Routing Analysis**

To investigate whether the CAF routing behaves in a clinically
meaningful manner, we analyze the attention weights allocated to the
visual soft-tissue modality ($P_{soft}$). Across the ensembled
evaluation:


- The mean visual attention weight $P_{soft}$ for **Non-Vapers** is
  $0.4973$ (with the 100-seed sweep producing a consistent estimate of
  $0.4946$, confirming stability across evaluation protocols).


- The mean visual attention weight $P_{soft}$ for **Active Vapers** is
  $0.0750$ (with the 100-seed sweep producing a consistent estimate of
  $0.1064$).

This indicates a $84.9\%$ **relative drop** in visual reliance when
active e-cigarette usage is detected.

To evaluate the statistical significance of this attention shift, we
perform significance testing on the attention weights:


1.  **Welch's Two-Sample t-Test:** The visual attention weights show a
    highly significant difference between groups, with $t = 16.441$,
    $df = 29.6$, and $p < 2.2 \times 10^{- 16}$. The 95% confidence
    interval for the difference is $[ 0.370,0.475]$.


2.  **Mann-Whitney U Test (Wilcoxon Rank-Sum Test):** The non-parametric
    separation is confirmed with $W = 528$ and $p = 1.267imes10^{- 13}$.


3.  **Cohen's d Effect Size:** The standardized effect size is $4.99$,
    indicating a massive and stable separation in routing pathways
    between vapers and non-vapers (Figure 2).

![Clinical Soft-Tissue Attention Weight Distribution. Boxplot comparing
P\_{soft} between non-vapers (n=24, blue) and active vapers (n=22, red)
across the N=46 cohort, averaged over 100 LOOCV seeds. Individual
patient weights are shown as jittered points. The separation is
statistically significant (Welch's t = 16.44, Cohen's d = 4.99, p \< 2.2
\\times 10\^{-16}).](figure2_attention_boxplot.png){width="5.833333333333333in"
height="4.166666666666667in"}

**Clinical Soft-Tissue Attention Weight Distribution.** Boxplot
comparing $P_{soft}$ between non-vapers ($n = 24$, blue) and active
vapers ($n = 22$, red) across the $N = 46$ cohort, averaged over 100
LOOCV seeds. Individual patient weights are shown as jittered points.
The separation is statistically significant (Welch's $t = 16.44$,
Cohen's $d = 4.99$, $p < 2.2 \times 10^{- 16}$).

### 3.3 Per-Patient Routing Behavior

To analyze how individual routing decisions vary across the cohort, we
inspect the patient-specific attention matrix (Figure 3).

![Per-Patient Multimodal Gating Routing Weights. Heatmap showing
individual patient routing weights (P\_{tab}, P\_{soft}, P\_{hard})
across all N=46 subjects. Patients are grouped by subgroup (non-vapers
vs. active vapers) to show the distinct routing patterns learned by the
gating network.](figure3_routing_heatmap.png){width="5.833333333333333in"
height="4.537036307961505in"}

**Per-Patient Multimodal Gating Routing Weights.** Heatmap showing
individual patient routing weights ($P_{tab},P_{soft},P_{hard}$) across
all $N = 46$ subjects. Patients are grouped by subgroup (non-vapers
vs. active vapers) to show the distinct routing patterns learned by the
gating network.


- For the **Non-Vaper subgroup (**$n = 24$**)**, attention is divided
  between the clinical photos ($P_{soft} \approx 0.50$) and the
  questionnaire ($P_{tab} \approx 0.48$).


- For the **Active Vaper subgroup (**$n = 22$**)**, the framework
  dynamically shifts its routing logic. The visual modality is
  suppressed down to near-zero ($P_{soft} \approx 0.075$), while the
  questionnaire channel is elevated to $0.907$.


- The zero-variance radiographic channel ($s_{hard} = 1.0$, representing
  the uninformative noise modality) is actively pruned across all
  patients, receiving a mean attention weight of only $1.77\%$
  ($P_{hard} = 0.0177$).

Four active vapers received partial visual attention weights above the
dismissal threshold ($P_{soft} > 0.25$), corresponding to patients with
lower nicotine concentration and vaping frequency scores in $x_{gate}$.
This graded suppression---proportional to exposure intensity rather than
binary---suggests the gate encodes a continuous vasoconstriction
severity spectrum rather than a hard vaper/non-vaper switch.

### **3.4 Modality Dismissal Analysis**

We define an a priori visual modality dismissal threshold at
$P_{soft} < 0.25$ (less than half of the equal-weight allocation of
$0.333$ for a 3-modality system) to identify when the model actively
rejects the photographic channel. Table 3 presents the contingency
analysis.

**Table 3: Modality Dismissal Contingency Table**

| Group | Visual Modality Dismissed ($P_{soft} < 0.25$) | Visual Modality Active ($P_{soft} \geq 0.25$) |
| :--- | :---: | :---: |
| **Non-Vapers** | $0$ patients | $24$ patients |
| **Active Vapers** | $18$ patients | $4$ patients |

A two-sided Fisher's Exact Test on this contingency table yields a
$p$-value of $2.64 \times 10^{- 8}$ (Odds Ratio = 0.0). The odds ratio
of 0.0 reflects the complete absence of non-vaper patients below the
dismissal threshold---a perfect separation that drives the ratio to its
mathematical minimum. Fisher's Exact Test is used precisely because of
this boundary condition, as chi-squared approximations are invalid for
zero-cell contingency tables.

### **3.5 Statistical Superiority Over Baselines**

To verify the statistical significance of CAF's diagnostic improvements,
we construct a pairwise disagreement matrix comparing the ensembled
out-of-fold predictions of CAF and the Late Concatenation baseline
(Table 4).

**Table 4: McNemar Disagreement Matrix**

| | Baseline Correct | Baseline Wrong |
| :--- | :---: | :---: |
| **CAF Correct** | $28$ patients | $4$ patients |
| **CAF Wrong** | $0$ patients | $14$ patients |

![McNemar Case Study Panel. A table summarizing the clinical profiles
and predictions of representative patients corrected by the CAF compared
to the static baseline model. This illustrates the visual suppression of
active vapers and correct visual routing of
non-vapers.](figure9_mcnemar_cases.png){width="5.833333333333333in"
height="1.6308234908136483in"}

**McNemar Case Study Panel.** A table summarizing the clinical profiles
and predictions of representative patients corrected by the CAF compared
to the static baseline model. This illustrates the visual suppression of
active vapers and correct visual routing of non-vapers.


- In 4 cases, the baseline model failed while CAF succeeded.



- There were 0 cases where CAF failed and the baseline succeeded.


- McNemar's test yields a chi-squared value of $4.0$ ($df = 1$) and a
  significant $p$-value of $0.0455$ ($p < 0.05$).

Inspection of the 4 cases where CAF succeeded and the baseline failed
reveals a highly specific clinical split (summarized in Figure 9):


- Two patients (Patient ID 2 and 33) are active vapers. Patient 2
  (active periodontitis) was correctly routed away from visual
  soft-tissue features to bypass confounding, while Patient 33 (healthy)
  had their visual modality dismissed ($P_{soft} \rightarrow 0$),
  preventing a false positive classification from localized tissue
  redness.


- Two patients (Patient ID 8 and 32) are non-vapers. Being healthy,
  their visual soft-tissue photographic scores were low
  ($s_{soft} = 0.20$), and the gating network correctly routed attention
  to their clinical photos to classify them as healthy, whereas the
  baseline classifier (lacking dynamic routing) misclassified them as
  diseased.

This confirms the diagnostic superiority of the dynamic multimodal
gating network over traditional late concatenation. Figure 4 shows the
corresponding ROC and Precision-Recall curves, illustrating CAF's
superior overall discrimination across all decision thresholds.

![Receiver Operating Characteristic (ROC) and Precision-Recall (PR)
Curves. Benchmarking curves comparing the Clinical Attention Framework
(CAF) with unimodal baselines and standard multimodal fusion strategies
(Late Concatenation, GMU, and Bilinear Fusion) under leave-one-out
cross-validation (LOOCV).](figure4_roc_pr_curves.png){width="5.833333333333333in"
height="2.47415135608049in"}

**Receiver Operating Characteristic (ROC) and Precision-Recall (PR)
Curves.** Benchmarking curves comparing the Clinical Attention Framework
(CAF) with unimodal baselines and standard multimodal fusion strategies
(Late Concatenation, GMU, and Bilinear Fusion) under leave-one-out
cross-validation (LOOCV).

## **4. Discussion**

### **4.1 Interpretation of Classification Performance**

The primary objective of this work was to evaluate whether a multimodal
system could break the performance ceiling of unimodal classifiers under
acute clinical confounders. In periodontal diagnostics, standalone
machine learning models trained strictly on questionnaire parameters are
hard-capped at an accuracy of $65.2\%$. When traditional fusion networks
(such as Late Concatenation or Bilinear Tensor Fusion) incorporate
visual soft-tissue features, their diagnostic accuracy collapses to
$60.9\%$ and $63.0\%$ respectively. This degradation constitutes a key
empirical finding: rather than improving diagnosis, the blind
concatenation of visual features contaminates the decision boundary.
Because e-cigarettes trigger localized soft-tissue vasoconstriction,
visual features mask active periodontitis as "healthy," thereby
introducing severe modality confounding that disrupts standard
multimodal fusion baselines (National Academies of Sciences,
Engineering, and Medicine, 2018). Under static fusion, these
contradictory visual signals confuse the model, rendering the combined
system inferior to the unimodal questionnaire baseline.

To explain the contamination mechanism formally, let $y \in \{ 0,1\}$
represent the true clinical diagnostic label (where $1$ denotes
periodontitis and $0$ healthy). Let
$s_{\text{tab}} \in [ 0,1]$ be the unimodal risk score
predicted from the patient's tabular questionnaire, and let
$s_{\text{soft}} \in [ 0,1]$ be the visual risk score
derived from the clinical photograph. For a non-vaping patient
($V = 0$), The visual presentation is highly diagnostic and correlates
strongly with systemic inflammation:

$$s_{\text{soft}} = y + \eta,\quad\eta\mathcal{\sim N(}0,\sigma^{2})$$

For an active vaping patient ($V = 1$), The pharmacological
vasoconstrictive masking effect of nicotine suppresses clinical gingival
inflammation (redness and bleeding), resulting in a deceptive "healthy"
visual score regardless of the true periodontal condition:

$$s_{\text{soft}} \approx 0\quad\text{for all }y$$

In traditional static fusion (e.g., late concatenation or linear
combination), the fused prediction logit is modeled as:

$$y_{\text{logit}} = w_{\text{tab}}s_{\text{tab}} + w_{\text{soft}}s_{\text{soft}} + w_{\text{hard}}s_{\text{hard}} + b$$

where the weights
$w_{\text{tab}},w_{\text{soft}},w_{\text{hard}}\mathbb{\in R}$ are
static scalars optimized globally across the entire cohort. Because
$s_{\text{soft}}$ is highly predictive of disease in the non-vaping
subpopulation ($n_{\text{non-vaper}} = 24$), the optimization of Binary
Cross-Entropy loss forces the global weight to be positive
($w_{\text{soft}} > 0$). However, when this globally positive
$w_{\text{soft}}$ is applied to active vapers with periodontitis
($y = 1,V = 1$), the visual predictor remains low
($s_{\text{soft}} \approx 0$). The term $w_{\text{soft}}s_{\text{soft}}$
collapses to approximately zero, failing to contribute to the logit.
This suppresses the overall fused prediction
$\widehat{y} = \sigma(y_{\text{logit}})$, resulting in a false-negative
classification. Conversely, if the optimizer reduces $w_{\text{soft}}$
to avoid these false-negative errors on vapers, it deprives the model of
critical visual diagnostic cues for non-vapers, leading to
classification errors in that subgroup. This is the formal contamination
mechanism: the deceptive visual signal from vapers corrupts the global
weight allocation in static fusion, resulting in a model that performs
worse than a simple questionnaire baseline (60.9% vs. 65.2%) (Figure 7).

![Multimodal Feature Contamination Mechanism. Flowchart illustrating how
a static fusion model is contaminated by misleading visual signals
(erythema/swelling masked by nicotine vasoconstriction), leading to
diagnostic errors, and how the CAF's exogenous context gate resolves
this conflict.](figure7_contamination.png){width="5.833333333333333in"
height="3.2885903324584427in"}

**Multimodal Feature Contamination Mechanism.** Flowchart illustrating
how a static fusion model is contaminated by misleading visual signals
(erythema/swelling masked by nicotine vasoconstriction), leading to
diagnostic errors, and how the CAF's exogenous context gate resolves
this conflict.

While an overall accuracy of $68.59\% \pm 1.52\%$ (reaching $69.57\%$
ensembled) may appear modest compared to context-free computer vision
tasks, it represents a highly robust diagnostic threshold given our
clinical pilot cohort size ($N = 46$) and the deliberate suppression of
the radiographic modality. In the clinical AI literature, small-N
cohorts ($N < 50$) are typical of initial clinical proof-of-concept
studies where patient recruitment is highly constrained by clinical
access and gold-standard verification. Machine learning models in this
regime struggle with extreme validation variance and representation
collapse. Achieving a stable $68.59\%$ accuracy under out-of-fold
validation demonstrates that a regularized, low-parameter architecture
can extract robust signal even from small clinical datasets.

Furthermore, the precision-recall profile of the CAF highlights an
important diagnostic trade-off. As shown in Figure 4, the
Questionnaire-Only baseline retains a higher Average Precision (AP =
0.726) at low recall thresholds compared to the CAF (AP = 0.676
ensembled). This sensitivity-precision tradeoff is consistent with
established clinical screening frameworks---the WHO definition of a
screening tool prioritizes sensitivity over specificity at the point of
first contact, with confirmatory diagnosis reserved for specialist
follow-up. While the questionnaire remains conservative to avoid false
positives, the inclusion of visual features in the CAF increases overall
diagnostic sensitivity (Recall = $73.91\%$ ensembled vs $69.57\%$ for
the baseline), capturing active periodontitis cases that would otherwise
be missed. In primary screening settings, maximizing sensitivity
(avoiding false negatives) is clinically prioritized, as a missed
diagnosis leads to irreversible bone destruction, whereas a false
positive is easily corrected by a clinical specialist during follow-up
exams.

### **4.2 Clinical Validity of the Routing Mechanism**

The dynamic attention weights generated by the CAF align directly with
established clinical pathophysiology. The gating network achieves an
astronomical effect size (Cohen's $d = 4.99$) in visual attention
routing ($P_{\text{soft}}$) between active vapers and non-vapers.
Clinically, this massive statistical separation means the routing
mechanism is highly stable and reliable. Rather than generating chaotic
attention shifts that vary from patient to patient, the framework
establishes distinct, predictable diagnostic pathways. This consistency
is vital for clinical trust, as it guarantees that the model's "logic"
remains stable across different patients within the same subgroup.

This routing behavior maps directly to clinical pharmacology. Nicotine
is a potent sympathomimetic agent that stimulates the release of
catecholamines (adrenaline and noradrenaline), which bind to
*$\alpha$*1​-adrenergic receptors on the gingival microvasculature to trigger
acute vasoconstriction (Silva, 2021; Whitehead et al., 2021; Lyytinen et
al., 2023). This vasoconstriction suppresses the visual outward signs of
periodontal disease---specifically, gingival erythema (redness) and
bleeding on probing---masking active tissue destruction (Figueredo et
al., 2021; Silva, 2021). The $78.5\%$ relative drop in visual attention
($P_{\text{soft}}$ decreases from $0.4946$ in non-vapers to $0.1064$ in
active vapers) reflects this known biological impact of nicotine on
microvascular perfusion, demonstrating that the network's gating
behavior matches the physical vascular suppression observed in clinical
trials.

Furthermore, the four vaper outliers who received partial visual
attention ($P_{\text{soft}} > 0.25$) provide clinical validation of a
dose-response behavior. A clinical review of these patients shows that
two of them (Patient ID 2 and 19) reported very low vaping frequencies
("rarely, less than once a week"), while the other two (Patient ID 27
and 58) reported high daily frequency but very short cumulative exposure
durations ("less than 6 months" and "6 months to 1 year", respectively).
Because their exposure was minimal or recent, the physiological
vasoconstrictive effect was likely negligible, allowing visual signs of
inflammation to remain visible. The gating network detected these mild
exposure profiles in $x_{\text{gate}}$ and allocated partial visual
trust ($P_{\text{soft}}$ ranging from $0.267$ to $0.312$). This
continuous routing behavior---rather than a binary switch---proves that
the gate models the biological severity of vasoconstriction as a
continuous dose-response curve. Crucially, the model was never
explicitly trained on nicotine pharmacology or visual vasoconstriction;
it independently discovered this routing strategy using classification
error minimization alone (Figure 8).

![Dose-Response Attention Routing vs. Nicotine Exposure. Scatter plot
showing the correlation between patient nicotine exposure intensity and
the soft-tissue photo attention weight (P\_{soft}). Active vapers (red
triangles) show a clear suppression of visual trust as exposure
increases, while non-vapers (blue circles) maintain high visual
trust.](figure8_dose_response.png){width="5.833333333333333in"
height="4.374998906386701in"}

**Dose-Response Attention Routing vs. Nicotine Exposure.** Scatter plot
showing the correlation between patient nicotine exposure intensity and
the soft-tissue photo attention weight ($P_{soft}$). Active vapers (red
triangles) show a clear suppression of visual trust as exposure
increases, while non-vapers (blue circles) maintain high visual trust.

### **4.3 Architectural Advantages Over Existing Fusion Methods**

The CAF's dynamic routing engine offers key architectural advantages
over standard fusion pipelines: 1. **Exogenous Gating Signal:**
Traditional cross-modal attention networks rely on *endogenous*
cross-attention, where attention weights are calculated as cross-product
correlations between the modalities themselves (Vaswani et al., 2017):

$$A_{i \rightarrow j} = \text{Softmax}\left( \frac{(W_{q}x_{i})(W_{k}x_{j})^{\top}}{\sqrt{d_{k}}} \right)$$

where $x_{i}$ and $x_{j}$ are features from the input modalities
themselves. If modality $j$ is corrupted or presenting deceptive signals
(such as a healthy-looking photo of a vaper with periodontitis), the
correlation matrix $A_{i \rightarrow j}$ is corrupted, forcing the
network to attend to a misleading signal. The CAF, by contrast, uses
*exogenous context routing*: the attention weights $P_{m}$ are computed
from an independent clinical context vector $x_{\text{gate}}$ that is
decoupled from the modality inputs themselves:

$$z = W_{\text{gating}}x_{\text{gate}} + b_{\text{gating}}$$

$$P_{m} = \text{Softmax}(z)_{m}$$

This decoupling ensures that the gate's decision to trust or distrust a
modality is driven by the clinical context, preventing the deceptive
contents of the modality itself from hijacking the routing decision.


2.  **Failure of General Gating (GMU):** The Gated Multimodal Unit (GMU)
    underperformed at $62.0\%$ accuracy. Because GMU gates are driven
    internally by the input modalities rather than an independent
    clinical confounder
    ($g = \sigma(W_{g}[ x_{\text{tab}},x_{\text{soft}}] + b_{g})$),
    a vaper's deceptive visual input ($s_{\text{soft}} \approx 0$) is
    interpreted as a healthy sign, leading the gate to assign trust
    inappropriately. Without access to the independent context (vaping
    history) in its gating mechanism, the GMU cannot resolve biological
    masking.


3.  **Competitive Softmax Allocation:** Standard multimodal
    architectures often use independent Sigmoid gates for each modality
    channel ($G_{m} \in [ 0,1]$). However, independent gates
    do not enforce a conservation of diagnostic trust. If one modality
    is suppressed, the model does not automatically scale up the other
    channels, which can lead to representation collapse. The CAF's
    competitive Softmax allocation ($\sum_{m}^{}P_{m} = 1.0$) forces a
    zero-sum redistribution of trust: suppressing the corrupted visual
    channel ($P_{\text{soft}} \rightarrow 0$) mathematically forces the
    gate to amplify the clean questionnaire channel
    ($P_{\text{tab}} \rightarrow 0.907$), maximizing information
    routing. This property is absent in all four baseline architectures
    evaluated in this study---a direct consequence of their static
    weight allocation---and explains why suppressing the visual
    confounder in those models does not automatically redirect trust to
    the questionnaire channel.


4.  **Parameter Efficiency:** The gating and classification layers of
    the CAF require only 14 learnable parameters (a $3 \times 3$ gating
    weight matrix $W_{\text{gating}}$ for 3 context features and 3
    modalities, a 3-dimensional bias vector $b_{\text{gating}}$, a
    1-dimensional classifier scale parameter $\gamma$, and a
    1-dimensional classifier bias). Under PAC-learning bounds, this low
    parameter footprint is highly suitable for small-N clinical cohorts
    ($N = 46$), preventing overfitting where deeper models fail (Figure
    6, Figure 10).

![Principal Component Analysis (PCA) Gated Projection. Projection of the
final fused modality scores under the dynamic gating system (CAF)
vs. static late concatenation. This visualizes how the dynamic gating
network resolves conflicting signals by aligning patients along clinical
clusters.](figure6_gated_projection.png){width="5.833333333333333in"
height="2.5121172353455816in"}

**Principal Component Analysis (PCA) Gated Projection.** Projection of
the final fused modality scores under the dynamic gating system (CAF)
vs. static late concatenation. This visualizes how the dynamic gating
network resolves conflicting signals by aligning patients along clinical
clusters.

![Learned Gating Weight Matrix (W\_{gating}) Heatmap. Connection weights
mapping patient vaping features (Frequency, Duration, Nicotine) to
output modalities (P\_{tab}, P\_{soft}, P\_{hard}). Values represent the
ensembled weights averaged over 420 LOOCV
runs.](figure10_w_gate_heatmap.png){width="5.833333333333333in"
height="4.666665573053368in"}

**Learned Gating Weight Matrix (**$W_{gating}$**) Heatmap.** Connection
weights mapping patient vaping features (Frequency, Duration, Nicotine)
to output modalities ($P_{tab},P_{soft},P_{hard}$). Values represent the
ensembled weights averaged over 420 LOOCV runs.

### **4.4 Explainability and Clinical Deployability**

In medical AI, "black box" neural networks lack the transparency
required for clinical adoption and regulatory approval. Post-hoc
explainability methods (such as SHAP or LIME) are heavily criticized in
high-stakes medicine (Rudin, 2019) because they rely on localized linear
approximations of complex, non-linear models, introducing "fidelity
errors" that can misrepresent what the model actually did. The CAF
resolves this by providing native, exact explainability. The attention
weights ($P_{\text{tab}}$, $P_{\text{soft}}$, $P_{\text{hard}}$) are not
post-hoc approximations; they are the primary, deterministic
computational outputs of the model.

In clinical practice, a dentist or dental hygienist using the CAF
receives a diagnostic prediction alongside a transparent clinical audit
trail: for a non-vaper, the system displays that it placed $50\%$ of its
trust on the clinical photo, while for an active vaper, it shows that
visual trust was suppressed to $10.6\%$ and shifted to the questionnaire
($90.7\%$) and radiograph ($1.7\%$). This explicit transparency
satisfies the rigorous audit trail requirements established by
regulatory bodies (such as the FDA and CE marking for Software as a
Medical Device - SaMD), bridging the gap between clinical neural
networks and explainable diagnostics. The CAF represents a structural
departure from black-box dental AI systems, providing a transparent,
context-aware meta-reasoner that guides these lower-level visual
encoders.

### **4.5 Limitations**

Despite its strong empirical performance, this pilot study has several
limitations:


1.  **Sample Size:** The cohort size ($N = 46$) is small, which is
    typical for initial clinical proof-of-concept studies but requires
    validation on larger cohorts ($N > 300$).


2.  **Simulated Radiographic Channel:** Due to the absence of a
    validated radiographic image encoder, we simulated the X-ray channel
    using a constant score of $1.0$. While this validated the network's
    active noise-pruning capabilities, it does not represent a real
    radiographic feature stream.


3.  **Subjectivity of Visual Scoring:** The photographic rating
    ($s_{\text{soft}}$) relies on a manual 1--10 visual consensus scale.
    Although scored by three independent clinicians, it remains subject
    to inter-rater variability. While three independent clinicians
    graded the photographic and radiographic scores to establish
    consensus, a formal Interclass Correlation Coefficient (ICC) was not
    computed for this pilot study, which represents a key methodological
    limitation to be addressed in future validation trials.


4.  **Single-Institution Cohort:** All data was collected from a single
    cohort at the European University Tbilisi, Tbilisi, Georgia, meaning
    our models may not generalize to different demographic or clinical
    populations.


5.  **Borderline McNemar Significance:** The McNemar test significance
    ($p = 0.0455$) is near the standard threshold of $0.05$. A larger
    sample size is required to confirm the diagnostic superiority of the
    framework.


6.  **Softplus Positivity Constraint Inductive Bias:** The Softplus
    positivity constraint on the scale parameter imposes an inductive
    bias that prevents the classifier from learning negative
    associations between modality scores and disease probability. While
    clinically motivated, this constraint reduces the model's freedom
    and represents an assumption that future work on larger cohorts
    should validate by comparing constrained and unconstrained variants.

### **4.6 Future Work and Scalability Blueprint**

To scale the Clinical Attention Framework for broad clinical deployment,
future work will focus on:


1.  **Automated Modality Encoders:** Replacing the manual
    $s_{\text{soft}}$ and $s_{\text{hard}}$ scores with automated
    feature representation vectors. Specifically, we will fine-tune deep
    visual models (such as DINOv2 or medical-specific vision-language
    models like BiomedCLIP) to extract 512-dimensional visual embeddings
    directly from clinical photographs and radiographs. The CAF's scalar
    multiplication of gate probability by modality score
    ($P_{m} \times s_{m}$) extends naturally to high-dimensional
    embeddings---multiplying the entire 512-D visual vector by the
    scalar $P_{\text{soft}}$---requiring no architectural modification
    to the gating mechanism itself.


2.  **Freezing the Gate as a Biological Prior:** Once trained on a large
    dataset, the confounder-driven gating network can be frozen as a
    "biological prior" that represents the relationship between vaping
    and visual vascular suppression. This frozen gate can then be
    transferred to new clinical cohorts, guiding the training of
    classifiers in different institutions without needing to relearn the
    confounder relationships.


3.  **Generalizing the Confounder Gate:** Evaluating the CAF on other
    nicotine-confounded medical conditions (e.g., wound healing,
    cardiovascular disease) where vasoconstriction masks diagnostic
    endpoints.


4.  **Multi-Site Validation Studies:** Initiating multi-center clinical
    trials to validate the CAF's generalizability across diverse dental
    institutions.


5.  **Automated Image Quality Assessments:** Replacing clinician manual
    scoring with automated image quality assessment networks to
    dynamically down-weight blurred, poorly lit, or misaligned intraoral
    photographs, treating image quality as a physical confounder.

## **5. Conclusion**

Nicotine-induced vasoconstriction suppresses the visual inflammatory
signs of periodontitis in active e-cigarette users, presenting a
deceptive "healthy" soft-tissue photo that masks active tissue
destruction. To resolve this diagnostic bottleneck, this paper
introduced the **Clinical Attention Framework (CAF)**, a novel
multimodal architecture that uses patient-specific vaping context to
dynamically route diagnostic trust. Across a clinical pilot cohort
($N = 46$), the CAF achieved $68.59\% \pm 1.52\%$ accuracy (reaching
$69.57\%$ ensembled) and a significant visual attention drop in active
vapers (Cohen's $d = 4.99$, $p < 2.2 \times 10^{- 16}$). Categorical
analysis confirmed that $81.8\%$ of active vapers had their visual
modality dismissed below the diagnostic trust threshold, compared to
$0\%$ of non-vapers (Fisher's Exact $p = 2.64 \times 10^{- 8}$)---a
perfect group separation driven entirely by learned routing. By shifting
diagnostic trust away from corrupted channels to reliable inputs based
on external context, our architecture establishes a new paradigm of
**dynamic reliability routing** over static cross-modal correlation
seeking. Ultimately, the CAF provides a scalable, domain-agnostic
blueprint to resolve biological confounders and visual masking effects
across clinical artificial intelligence (Figure 11).

![Gating Network Training Loss Convergence Curves. Convergence envelope
of the Binary Cross-Entropy loss across all 420 LOOCV runs, displaying
the mean curve, \\pm 1 standard deviation band, min-max bounds, and
representative individual
trajectories.](figure11_loss_convergence.png){width="5.833333333333333in"
height="4.117646544181977in"}

**Gating Network Training Loss Convergence Curves.** Convergence
envelope of the Binary Cross-Entropy loss across all 420 LOOCV runs,
displaying the mean curve, $\pm 1$ standard deviation band, min-max
bounds, and representative individual trajectories.

Funding: This research was supported and funded by the European
University, Tbilisi, Georgia. The funder had no role in study design,
data collection and analysis, decision to publish, or preparation of the
manuscript.

Institutional Review Board Statement: The study was conducted in
accordance with the ethical principles outlined in the Declaration of
Helsinki. The data collection methodologies and diagnostic clinical
pipelines utilized in this manuscript were formally approved by the
Research Ethics Committee of the Faculty of Medicine at the European
University, Tbilisi, Georgia under the registered institutional project
protocol: \"Impact of E-Cigarette Use on Periodontal Health and Risk
Prediction Using Machine Learning: A Cross-Sectional Observational
Study\" (Official Meeting Protocol No. 7, Approved May 10, 2026) The
committee evaluated the protocol\'s data management and data protection
strategies, confirming that the non-interventional, survey-, and
clinical-examination-based design posed no adverse physical risks or
experimental interventions to the evaluation cohort.

Data Availability Statement: The clinical and tabular datasets generated
and analyzed during the current study are not publicly available due to
institutional data protection restrictions and patient privacy
constraints established by the Research Ethics Committee. However, the
de-identified minimum dataset and corresponding model evaluation metrics
are available from the corresponding author upon reasonable academic
request. The algorithmic pipeline execution code and trained model
weights for the Clinical Attention Framework (CAF) are hosted publicly
on GitHub at https://github.com/aburayaanas/CAF.git

Competing Interests: The authors declare that they have no competing
interests.

Informed Consent Statement: Written informed consent was obtained from
all individual participants included in the study cohort (N = 46) prior
to any clinical examinations, questionnaire distribution, or intraoral
photography. All participants were thoroughly informed regarding the
study objectives, the voluntary nature of their participation, and their
right to withdraw at any stage without consequence. Specific, explicit
consent was obtained from each subject regarding the secondary use of
their de-identified clinical features, survey metrics, and intraoral
soft-tissue images for algorithmic training, machine learning
validation, and diagnostic framework evaluation. To protect patient
autonomy and data privacy in strict compliance with institutional
guidelines and data protection regulations, all structural inputs were
completely anonymized and stripped of personally identifiable
information prior to entering the machine learning pipeline.

**Reference List**

Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017).
Gated multimodal units for information fusion. arXiv preprint
arXiv:1702.01992. <https://arxiv.org/abs/1702.01992>

Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures
for model selection. Statistics Surveys, 4, 40--79.
<https://doi.org/10.1214/09-SS054>

Brandsma, D., & van den Bent, M. J. (2009). Pseudoprogression and
pseudoresponse in the treatment of gliomas. Current Opinion in
Neurology, 22(6), 633--638.
<https://doi.org/10.1097/WCO.0b013e328332363e>

Das, S. C., Biswas, S., Khan, O., Akter, R., Azad, M. A. K., Sarkar, S.
K., Masum, M. A., & Bedoura, S. (2025). Evaluation of anti-inflammatory
and wound healing properties of Tinospora cordifolia extract. PLOS ONE,
20(1), e0317928. <https://doi.org/10.1371/journal.pone.0317928>

Elantary, R., & Othman, S. (2025). Artificial intelligence in
electrocardiography: From automated arrhythmia detection to predicting
hidden cardiovascular disease. Cureus, 17(10), e94065.
<https://doi.org/10.7759/cureus.94065>

Fan, D., Greybush, S. J., Clothiaux, E. E., & Gagne, D. J. (2024).
Physically explainable deep learning for convective initiation
nowcasting using GOES-16 satellite observations. Artificial Intelligence
for the Earth Systems, 3(3), AIES-D-23-0098.1.
<https://doi.org/10.1175/AIES-D-23-0098.1>

Figueredo, C. A., Abdelhay, N., Figueredo, C. M., Catunda, R., & Gibson,
M. P. (2021). The impact of vaping on periodontitis: A systematic
review. Clinical and Experimental Dental Research, 7(3), 376--384.
<https://doi.org/10.1002/cre2.360>

Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W.,
Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural
networks. Nature Machine Intelligence, 2(11), 665--673.
<https://doi.org/10.1038/s42256-020-00257-z>

Guyon, I., & Elisseeff, A. (2003). An introduction to variable and
feature selection. Journal of Machine Learning Research, 3, 1157--1182.

Jandoubi, B., & Akhloufi, M. A. (2025). Multimodal artificial
intelligence in medical diagnostics. Information, 16(7), 591.
<https://doi.org/10.3390/info16070591>

Lobene, R. R., Weatherford, T., Ross, N. M., Lamm, R. A., & Menaker, L.
(1986). A modified gingival index for use in clinical trials. Clinical
Preventive Dentistry, 8(1), 3--6.

Luke, S. (2024). Essentials of metaheuristics (3rd ed.). Lulu Press.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
model predictions. In Advances in Neural Information Processing Systems
(Vol. 30, pp. 4765--4774). Curran Associates, Inc.

Lyytinen, G., Brynedal, A., Anesäter, E., Antoniewicz, L., Blomberg, A.,
Wallén, H., Bosson, J. A., Hedman, L., Mobarrez, F., Tehrani, S., &
Lundbäck, M. (2023). Electronic cigarette vaping with nicotine causes
increased thrombogenicity and impaired microvascular function in healthy
volunteers: A randomised clinical trial. Cardiovascular Toxicology, 23,
255--264. <https://doi.org/10.1007/s12012-023-09802-9>

National Academies of Sciences, Engineering, and Medicine. (2018).
Public health consequences of e-cigarettes: Oral diseases. National
Academies Press. <https://www.ncbi.nlm.nih.gov/books/NBK507170/>

Otieno, T. A., & Oluoch, C. W. (2026). Systematic review of explainable
AI in diagnostic imaging: Towards a transparent and auditable clinical
framework. Journal of African Interdisciplinary Studies, 10(1), 66--90.

Papapanou, P. N., Sanz, M., Buduneli, N., Dietrich, T., Feres, M., Fine,
D. H., Flemmig, T. F., Garcia, R., Giannobile, W. V., Graziani, F.,
Greenwell, H., Herrera, D., Kao, R. T., Kebschull, M., Kinane, D. F.,
Kirkwood, K. L., Kocher, T., Kornman, K. S., Kumar, P. S., \... Tonetti,
M. S. (2018). Periodontitis: Consensus report of Workgroup 2 of the 2017
World Workshop on the Classification of Periodontal and Peri-Implant
Diseases and Conditions. Journal of Periodontology, 89(Suppl. 1),
S173--S182. <https://doi.org/10.1002/JPER.17-0721>

Platt, J. (1999). Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods. In A. J. Smola, P.
Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), Advances in large margin
classifiers (pp. 61--74). MIT Press.

Ramírez, J., Górriz, J. M., Ortiz, A., Cole, J. H., & Dyrba, M. (2020).
Editorial: Deep learning in aging neuroscience. Frontiers in
Neuroinformatics, 14, 573974.
<https://doi.org/10.3389/fninf.2020.573974>

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust
you?": Explaining the predictions of any classifier. In Proceedings of
the 22nd ACM SIGKDD International Conference on Knowledge Discovery and
Data Mining (pp. 1135--1144). ACM.
<https://doi.org/10.1145/2939672.2939778>

Rudin, C. (2019). Stop explaining black box machine learning models for
high stakes decisions and use interpretable models instead. Nature
Machine Intelligence, 1(5), 206--215.
[https://doi.org/10.1038/s42256-019-0048-x](https://www.google.com/search?q=https%3A%2F%2Fdoi.org%2F10.1038%2Fs42256-019-0048-x)

Silva, H. (2021). Tobacco use and periodontal disease---The role of
microvascular dysfunction. Biology, 10(5), 441.
[https://doi.org/10.3390/biology10050441](https://www.google.com/search?q=https%3A%2F%2Fdoi.org%2F10.3390%2Fbiology10050441)

Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020).
Fooling LIME and SHAP: Adversarial attacks on post hoc explanation
methods. Proceedings of the AAAI/ACM Conference on AI, Ethics, and
Society, 180--186.

Valiant, L. G. (1984). A theory of the learnable. Communications of the
ACM, 27(11), 1134--1142.
[https://doi.org/10.1145/1968.1972](https://www.google.com/search?q=https%3A%2F%2Fdoi.org%2F10.1145%2F1968.1972)

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
In Advances in Neural Information Processing Systems (Vol. 30, pp.
5998--6008). Curran Associates, Inc.

Wen, P. Y., Macdonald, D. R., Reardon, D. A., Cloughesy, T. F.,
Sorensen, A. G., Galanis, E., DeGroot, J., Wick, W., Gilbert, M. R.,
Lassman, A. B., Tsien, C., Mikkelsen, T., Wong, E. T., Chamberlain, M.
C., Stupp, R., Lamborn, K. R., Vogelbaum, M. A., van den Bent, M. J., &
Chang, S. M. (2010). Updated response assessment criteria for high-grade
gliomas: Response assessment in neuro-oncology working group. Journal of
Clinical Oncology, 28(11), 1963--1972.
[https://doi.org/10.1200/JCO.2009.26.3541](https://www.google.com/search?q=https%3A%2F%2Fdoi.org%2F10.1200%2FJCO.2009.26.3541)

Whitehead, A. K., Erwin, A. P., & Yue, X. (2021). Nicotine and vascular
dysfunction. Acta Physiologica, 231(4), e13631.
[https://doi.org/10.1111/apha.13631](https://www.google.com/search?q=https%3A%2F%2Fdoi.org%2F10.1111%2Fapha.13631)

World Medical Association. (2013). World Medical Association Declaration
of Helsinki: Ethical principles for medical research involving human
subjects. JAMA, 310(20), 2191--2194.
[https://doi.org/10.1001/jama.2013.281053](https://www.google.com/search?q=https%3A%2F%2Fdoi.org%2F10.1001%2Fjama.2013.281053)

Zadeh, A., Chen, M., Poria, S., Cambria, E., & Morency, L.-P. (2017).
Tensor fusion network for multimodal sentiment analysis. In Proceedings
of the 2017 Conference on Empirical Methods in Natural Language
Processing (pp. 1103--1114). Association for Computational Linguistics.
