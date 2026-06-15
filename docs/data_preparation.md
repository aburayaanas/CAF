# Data Preparation & Cohort Filtering Protocol

This document details the exact, reproducible data cleaning, standardization, feature engineering, and cohort filtering steps performed to prepare the Clinical Attention Framework (CAF) research dataset.

---

## 1. Raw Data Sources

The dataset is compiled from three primary files in the `data/` directory:
1. **`questionnaire.xlsx`**: Excel file containing patient questionnaire responses across five sheets:
   - *Demographic Info*
   - *Vaping Habits*
   - *Oral Hygiene Practices*
   - *Lifestyle Factors*
   - *Self-Reported Periodontal Sympt*
2. **`periodontogram_dataset.xlsx`**: Detailed tooth-level clinical pocket depth measurements, Bleeding on Probing (BOP), and Full Mouth Plaque Index (FMPI) measurements.
3. **`PARTICIPANT ID.xlsx`**: Clinician photo scores ($S_{soft}$) and gold-standard clinical diagnosis grades (`Gingivitis grading `).

---

## 2. Cohort Filtering & Exclusions

A strict clinical filtering process was applied to enforce dataset validity and completeness:
* **Manual Exclusions**: IDs `{18, 20, 23, 40, 65, 68}` were excluded from the demographic sheet due to incomplete surveys or clinical disqualifications.
* **Complete-Case Outcome Requirement**: To ensure scientific integrity, we applied strict complete-case filtering. Patients without a clinician visual photo rating (`1-10 pic`) or a gold-standard diagnosis grade (`Gingivitis grading `) were dropped. **No imputation was performed on clinical target variables or rating inputs.**
* **Final Cohort Intersection**:
  - Raw Demographic Sheet: 71 patients.
  - Demographic Filtered Sheet: 65 patients.
  - Non-null Clinician Photo Ratings: 55 patients.
  - **Final Cohort (Clean Intersection): 50 patients.**

---

## 3. Cleaning & Standardization Protocol

### Demographics
* String values normalized to lowercase with trailing whitespaces stripped.
* `Age` normalized to numerical values and empty values imputed using the cohort mean age (`22.4`).
* Categorical columns (`Highest Education Level Completed`, `Occupation`) mapped to standard labels (e.g. fixing typos like `some collage` -> `some college`).

### Vaping Habits
* Normalised strings and standard categorical responses.
* Enforced strict survey logic: if a patient is a non-vaper, all vape-related columns (duration, device, frequency, nicotine concentration) are mapped to `'0'`.
* Standardised traditional cigarette smoking duration (`If so, how long did you smoke traditional cigarettes?`): missing entries (`NaN`) and `'never'` variants were mapped to a single uniform `'never'` state to represent 0 smoking years.

### Oral Hygiene
* Standardised brushing frequency and toothbrush changing intervals.
* **Multi-hot Feature Extraction**: Extracted binary multi-hot features from the unstructured text of interdental cleaning aids:
  - `uses_miswak`
  - `uses_tongue_scraper`
  - `uses_interdental_brushes`
  - `uses_floss_pick`
  - `uses_water_irrigator`
  - `uses_toothpick`

### Lifestyle Factors
* Normalized diet types and exercise patterns.
* **Multi-hot Feature Extraction**: Extracted binary indicators from supplements and chronic diseases:
  - Supplements: `takes_creatine`, `takes_omega3`, `takes_vitamin_d`, `takes_zinc`, `takes_multivitamins`, `takes_magnesium`, `takes_protein`.
  - Chronic Diseases: `has_asthma`, `has_lupus`, `has_psoriasis`, `has_hypothyroidism`.

### Periodontogram Wide-Pivoting
* **Clinical Indices**: BOP and FMPI percentages were parsed, stripped of `%` characters, and scaled to floats between `0.0` and `1.0`.
* **Pocket Depth Parsing**: Pocket depth scores (stored as 3-character text sequences like `'102'`) were unpacked into three discrete distal, mid, and mesial scores (`score_s1`, `score_s2`, `score_s3`).
* **Wide-Pivot**: Re-organized tooth-level measurements into a wide table where each row represents a single patient, producing 280+ features spanning 28 teeth (excluding wisdom teeth) and 2 surfaces (front/back). Missing tooth sites default to healthy (`0`).

---

## 4. Final Merged Output

All processed sheets were merged on patient `id` using an `inner` join. The final dataset is saved to:
`data/questionnaire_cleaned.xlsx`

* **Dimensions**: 50 rows, 349 columns.
* **Leaky Columns**: Separated from baseline questionnaire inputs (e.g. `label`, `bop_scaled`, `fmpi_scaled`, clinician scores).
* **Missing values**: Zero `NaN` values remain in the final cleaned file.
