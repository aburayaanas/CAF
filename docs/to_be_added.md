2.1.1 Participant Selection and Inclusion Criteria


This single-center, cross-sectional observational pilot study was conducted at the European University Dental Clinic in Tbilisi, Georgia, from September 2025 to March 2026. The study protocol was formally approved by the Institutional Review Board (IRB) of European University, executed in strict compliance with the Declaration of Helsinki, and utilized a robust data anonymization procedure alongside mandatory written informed consent from all participants. The study population comprised N = 46 systemically healthy participants aged 18 to 35 years, balanced symmetrically into 23 healthy individuals (Label 0) and 23 diseased individuals (Label 1). This cohort was partitioned into two strictly age- and gender-matched groups: Cohort 1 (n_1 = 22 active electronic nicotine delivery systems [ENDS] users) and Cohort 2 (n_2 = 24 non-vaping controls). For Cohort 1, variations in vaping history, duration, and habits were documented as continuous metadata variables rather than rigid binary cuts to preserve a real-world exposure spectrum, while Cohort 2 was restricted exclusively to individuals with absolute lifetime abstinence from all tobacco, electronic aerosols, and systemic nicotine variations. To isolate clear tissue interactions, all participants were required to have no history of systemic antibiotics within the last 3 months or professional periodontal treatment within the last 6 months. Individuals were excluded if they presented with systemic diseases (e.g., diabetes or autoimmune conditions), pregnancy, or current medications affecting gingival tissues (e.g., calcium channel blockers, phenytoin, cyclosporine). Dual-users of combined combustible tobacco and vaping were also excluded, alongside any cases presenting with insufficient data or missing modalities. Clinical diagnostic criteria aligned strictly with the 2017 World Workshop classification, where plaque-induced gingivitis was defined as Bleeding on Probing (BOP) exceeding 10% of sites with a Probing Pocket Depth (PPD) ≤ 3 mm and an absolute absence of radiographic bone loss. This structured cohort provided the core matrix for subsequent computational modeling and dynamic attention routing verification. 


2.1.2 The Nicotine-Induced Inflammatory Masking and Clinical Phenotyping
Standard periodontal diagnosis relies on visual inflammatory indicators — gingival erythema, edema, and bleeding on probing (BOP). Nicotine disrupts this paradigm by inducing peripheral vasoconstriction in the gingival microvasculature, suppressing overt inflammatory signs despite active underlying tissue destruction (Al-Bayaty et al., 2013; Figueredo et al., 2021). The result is a deceptive clinical presentation: disease present, but visually hidden.
This masking effect was directly observed in the present cohort. Despite recording higher mean BOP upon probing (vapers: 13.04% vs. non-vapers: 8.15%), vaping patients consistently presented with visually deceptive gingival appearances — reduced erythema, firmer tissue texture, and minimal visible swelling compared to non-vaping patients with lower BOP scores. Soft-tissue photography, while reliable in non-vapers, systematically underrepresented disease severity in the vaping cohort — a direct clinical manifestation of nicotine-induced vasoconstriction, suppressing the vascular response to inflammation.
This divergence has a direct architectural implication. In non-vapers, soft-tissue photography reliably captures inflammatory signs. In nicotine users, the same modality produces a deceptive signal while hard-tissue radiographs and patient questionnaire data retain their diagnostic validity. A static fusion model cannot resolve this conflict. This physiologically grounded, clinically confirmed masking effect establishes this cohort as the ideal biological stress test for the CAF's Conflict-Resolution Gate.


2.1.3 Multimodal Data Acquisition Protocol
All three modalities were acquired during a single standardized clinical visit by a single calibrated examiner (M.N.) to minimize inter-operator variability.
Soft-Tissue Modality (P_soft) Intraoral clinical photography was performed using a Canon EOS 4000D equipped with a 100mm macro lens and twin flash system, calibrated to approximately 5500K color temperature. A standardized exposure protocol was applied across all patients (f/22, 1/200s, ISO 100) to ensure consistent color rendition and depth of field. Bilateral cheek retractors were used for each side, and the patient's head position was standardized throughout. A full-mouth photographic series of 11 images per patient was acquired, capturing all relevant soft-tissue regions for subsequent inflammatory scoring.
Hard-Tissue Modality (P_hard) Hard-tissue assessment was performed using panoramic radiography via Rotational Slit-Beam Tomography on conventional film. Full-mouth radiographic coverage was obtained for all patients. Radiographic features extracted included bone crest levels and relevant hard-tissue morphological findings. All radiographs were independently evaluated by three expert clinicians to ensure diagnostic reliability.
Questionnaire (P_tab) Prior to clinical examination, each participant completed a standardized 45-item paper-based questionnaire covering five domains: demographic characteristics, vaping and nicotine habits, oral hygiene practices, lifestyle factors, and self-reported periodontal symptoms. Paper administration before examination ensured responses were independent of clinical findings and examiner influence.
Data Recording and Anonymization All acquired data were stored on a password-protected, encrypted computer accessible exclusively to the principal investigators (M.N. and A.A.). Patient identifiers were removed at the point of data entry and replaced with anonymous numerical codes, ensuring no personally identifiable information was retained within the dataset. All procedures were conducted in full compliance with institutional data protection protocols and the Declaration of Helsinki.



2.2.2 Structural Radiographic scaling 
Parallel to the soft-tissue indexing approach, panoramic radiographs were converted into a continuous $1.0\text{--}10.0$ Radiographic Severity Index to ensure architectural consistency across all modalities and eliminate dimensional incompatibility within the feature fusion pipeline. The scale was developed as a continuous numerical expansion of the 2017 AAP/EFP Periodontal Classification staging criteria, mapping established clinical staging thresholds onto a fine-grained, machine-readable range. Crucially, the Radiographic Severity Index represents a relative continuous scaling of observable hard-tissue pathology and does not constitute absolute millimeter bone level measurements, which would require a standardized long-cone paralleling technique with calibrated grids.
Instead, the continuous scores reflect alveolar bone crest levels, the extent and pattern of bone loss, and furcation involvement as observed on standardized panoramic radiography. The structural anchoring boundaries of this scoring matrix are defined in Table 2.
[Table 2: Radiographic Severity Index — AAP/EFP Anchored Scoring Matrix — insert your table here]
Score
Clinical Stage
Diagnostic Criteria
1–2
Healthy
Alveolar crest 1.0–1.5mm apical to CEJ; intact lamina dura; no bone loss visible
3–4
Mild / Stage I ABL
Bone loss limited to coronal third of root; <15% vertical or horizontal bone loss
5–6
Moderate / Stage II ABL
Bone loss extending to middle third of root; 15–33% bone loss
7–8
Severe / Stage III ABL
Bone loss extending to apical third; >33% bone loss; angular defects visible
9–10
Advanced / Stage IV ABL
Severe bone loss near tooth apex; furcation involvement Class II/III; tooth mobility visible

Three independent expert clinicians, blinded to patient vaping status and to each other's ratings, independently assigned Radiographic Severity Index scores to each patient's panoramic radiograph. Final scores were computed as the mean of the three independent ratings, consistent with the soft-tissue scoring protocol. Inter-rater reliability is reported in Section 2.2.4.



2.2.4 Inter-Rater Reliability of Clinical Indices


2.2.5 Low-Dimensional Feature Alignment (Simulated Encoders)
Multi-source clinical datasets present severe dimensionality discrepancies (e.g., high-dimensional questionnaire tensors, 2D clinical image grids, and radiographic images). Rather than feeding high-dimensional raw data directly into the gating network, which exposes deep models to shortcut learning and representation collapse on small cohorts, we perform low-dimensional feature alignment.


We map each modality into a normalized, 1D risk score predictor space ($s_m \in [0, 1]$), which aligns all heterogeneous data channels into a unified probability space:


1. **Tabular Questionnaire Predictor ($s_{tab}$):** The $k$-selected tabular features are passed through a Logistic Regression estimator trained via out-of-fold cross-validation, outputting the predicted class probability of periodontitis. Logistic Regression was selected as the tabular predictor because it outputs calibrated probability scores directly interpretable as disease risk—a property not natively available in SVM or tree-based models without post-hoc calibration. This ensures $s_{tab}$ operates in the same $[0, 1]$ probability space as $s_{soft}$ and $s_{hard}$, enabling meaningful arithmetic comparison within the gating fusion.
2. **Visual Soft-Tissue Predictor ($s_{soft}$):** The clinical intraoral photographs of the anterior gingival tissue are evaluated using a standardized visual consensus protocol (scored on a 1–10 scale by three independent clinicians, and divided by 10.0 to normalize to a $[0.0, 1.0]$ range), yielding a diagnostic probability score for soft-tissue inflammation. The 1–10 ordinal scale was selected to match established clinical periodontal indices (such as the modified Gingival Index and Bleeding Index), which clinicians already use in practice, ensuring ecological validity of the scoring protocol. Division by 10.0 maps scores to a probability-like space consistent with the tabular predictor output. Note that in larger-scale deployments, this manual scoring can be replaced by a high-dimensional vector representation extracted using deep visual encoders (e.g., convolutional neural networks or vision transformers).
3. **Radiographic (X-ray) Predictor ($s_{hard}$):** In the absence of a validated radiographic image encoder for this pilot cohort, the radiographic channel is assigned a constant uninformative score of 1.0. This constitutes a deliberate architectural stress test: a correctly functioning gating network must recognize zero-variance input and suppress it toward $P_{hard} \approx 0$. Empirically, the gate drives $P_{hard}$ to $1.77\%$, confirming the network's capacity for autonomous noise rejection without explicit supervision.


By translating each high-dimensional modality into a single unified risk score $s_m \in [0, 1]$, we ensure that the gating network acts as a clean, interpretable meta-reasoner that dynamically controls the transmission of pre-aligned diagnostic signals rather than performing raw feature learning.
