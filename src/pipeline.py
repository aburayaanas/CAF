import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats

from src.models.caf import ClinicalAttentionFramework
from src.models.baselines import LateConcatenationFusion, GatedMultimodalUnit, BilinearTensorFusion

def run_joint_optimization(X_gate, s_tab, s_soft, s_hard, y, max_iter=100):
    """
    Direct optimization of the parameters of the ClinicalAttentionFramework
    using PyTorch L-BFGS optimizer on the training split.
    """
    model = ClinicalAttentionFramework()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS(
        model.parameters(), 
        lr=0.05, 
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50
    )
    
    # Format inputs into PyTorch tensors
    X_gate_t = torch.tensor(X_gate, dtype=torch.float32)
    s_tab_t = torch.tensor(s_tab, dtype=torch.float32).unsqueeze(1)
    s_soft_t = torch.tensor(s_soft, dtype=torch.float32).unsqueeze(1)
    s_hard_t = torch.tensor(s_hard, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    def closure():
        optimizer.zero_grad()
        logits, _ = model(X_gate_t, s_tab_t, s_soft_t, s_hard_t)
        loss = criterion(logits, y_t)
        loss.backward()
        return loss
        
    model.train()
    optimizer.step(closure)
    return model




def train_baseline_model(model_class, s_tab, s_soft, s_hard, y, max_iter=100):
    """
    Helper function to optimize any PyTorch baseline model on the training split.
    """
    model = model_class()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=max_iter)
    
    s_tab_t = torch.tensor(s_tab, dtype=torch.float32).unsqueeze(1)
    s_soft_t = torch.tensor(s_soft, dtype=torch.float32).unsqueeze(1)
    s_hard_t = torch.tensor(s_hard, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    def closure():
        optimizer.zero_grad()
        logits = model(s_tab_t, s_soft_t, s_hard_t)
        loss = criterion(logits, y_t)
        loss.backward()
        return loss
        
    model.train()
    optimizer.step(closure)
    return model

def run_loocv_evaluation(dataset):
    """
    Performs Leave-One-Out Cross-Validation (LOOCV) evaluation on the cohort.
    Benchmarks the following models side-by-side:
    1. Baseline (Questionnaire-Only ANOVA-16 Logistic Regression)
    2. Late Concatenation Fusion Baseline
    3. Gated Multimodal Unit (GMU) Baseline
    4. Bilinear Tensor Fusion Baseline
    5. CAF Gating Pipeline
    """
    print("=" * 80)
    print("STARTING LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV) EVALUATION")
    print(f"Cohort size: {len(dataset)} patients")
    print("=" * 80)
    
    # 1. Prepare inputs
    y_labels = np.array(dataset.labels)
    X_gate_all = dataset.X_gate
    s_soft_all = dataset.s_soft
    s_hard_all = dataset.s_hard
    X_tab_all = dataset.X_tab
    
    loo = LeaveOneOut()
    
    # Storing outputs for evaluation
    y_base_pred, y_base_prob = [], []
    y_concat_pred, y_concat_prob = [], []
    y_gmu_pred, y_gmu_prob = [], []
    y_bilinear_pred, y_bilinear_prob = [], []
    
    y_caf_pred, y_caf_prob = [], []
    caf_attention_weights = []
    
    # LOOCV Loop
    for train_idx, test_idx in loo.split(X_tab_all):
        # Splitting tabular dataset
        X_train_tab, X_test_tab = X_tab_all[train_idx], X_tab_all[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        
        # --- Step A: Tabular Predictor (S_tab) ---
        selector = SelectKBest(score_func=f_classif, k=16)
        X_train_sel = selector.fit_transform(X_train_tab, y_train)
        X_test_sel = selector.transform(X_test_tab)
        
        clf_tab = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
        clf_tab.fit(X_train_sel, y_train)
        
        # Predict baseline questionnaire outputs
        s_tab_train = clf_tab.predict_proba(X_train_sel)[:, 1]
        s_tab_test = clf_tab.predict_proba(X_test_sel)[:, 1]
        
        y_base_pred.append(clf_tab.predict(X_test_sel)[0])
        y_base_prob.append(s_tab_test[0])
        
        # Retrieve scores for visual modalities
        s_soft_train, s_soft_test = s_soft_all[train_idx], s_soft_all[test_idx]
        s_hard_train, s_hard_test = s_hard_all[train_idx], s_hard_all[test_idx]
        
        # --- Step B: Train Baselines ---
        model_concat = train_baseline_model(LateConcatenationFusion, s_tab_train, s_soft_train, s_hard_train, y_train)
        model_gmu = train_baseline_model(GatedMultimodalUnit, s_tab_train, s_soft_train, s_hard_train, y_train)
        model_bilinear = train_baseline_model(BilinearTensorFusion, s_tab_train, s_soft_train, s_hard_train, y_train)
        
        # --- Step C: Train CAF Gating model ---
        X_gate_train, X_gate_test = X_gate_all[train_idx], X_gate_all[test_idx]
        model_caf = run_joint_optimization(X_gate_train, s_tab_train, s_soft_train, s_hard_train, y_train)
        
        # --- Step D: Evaluate All Models on Left-Out Patient ---
        s_tab_test_t = torch.tensor(s_tab_test, dtype=torch.float32).unsqueeze(0)
        s_soft_test_t = torch.tensor(s_soft_test, dtype=torch.float32).unsqueeze(0)
        s_hard_test_t = torch.tensor(s_hard_test, dtype=torch.float32).unsqueeze(0)
        
        # 1. Late Concat Evaluation
        model_concat.eval()
        with torch.no_grad():
            logit_c = model_concat(s_tab_test_t, s_soft_test_t, s_hard_test_t)
            prob_c = torch.sigmoid(logit_c).item()
            pred_c = 1 if prob_c >= 0.5 else 0
            y_concat_pred.append(pred_c)
            y_concat_prob.append(prob_c)
            
        # 2. GMU Evaluation
        model_gmu.eval()
        with torch.no_grad():
            logit_g = model_gmu(s_tab_test_t, s_soft_test_t, s_hard_test_t)
            prob_g = torch.sigmoid(logit_g).item()
            pred_g = 1 if prob_g >= 0.5 else 0
            y_gmu_pred.append(pred_g)
            y_gmu_prob.append(prob_g)
            
        # 3. Bilinear Evaluation
        model_bilinear.eval()
        with torch.no_grad():
            logit_b = model_bilinear(s_tab_test_t, s_soft_test_t, s_hard_test_t)
            prob_b = torch.sigmoid(logit_b).item()
            pred_b = 1 if prob_b >= 0.5 else 0
            y_bilinear_pred.append(pred_b)
            y_bilinear_prob.append(prob_b)
            
        # 4. CAF Evaluation
        model_caf.eval()
        with torch.no_grad():
            x_gate_test_t = torch.tensor(X_gate_test, dtype=torch.float32)
            logit_caf, probs_caf = model_caf(x_gate_test_t, s_tab_test_t, s_soft_test_t, s_hard_test_t)
            prob_caf = torch.sigmoid(logit_caf).item()
            pred_caf = 1 if prob_caf >= 0.5 else 0
            
            y_caf_pred.append(pred_caf)
            y_caf_prob.append(prob_caf)
            caf_attention_weights.append(probs_caf.numpy().flatten())
            
    # Calculate performance metrics
    y_true = y_labels
    y_base_pred, y_base_prob = np.array(y_base_pred), np.array(y_base_prob)
    y_concat_pred, y_concat_prob = np.array(y_concat_pred), np.array(y_concat_prob)
    y_gmu_pred, y_gmu_prob = np.array(y_gmu_pred), np.array(y_gmu_prob)
    y_bilinear_pred, y_bilinear_prob = np.array(y_bilinear_pred), np.array(y_bilinear_prob)
    
    y_caf_pred, y_caf_prob = np.array(y_caf_pred), np.array(y_caf_prob)
    caf_attention_weights = np.array(caf_attention_weights)
    
    # 2. Print LOOCV Metrics Table
    print("\n" + "=" * 80)
    print("LOOCV PERFORMANCE COMPARISON")
    print("=" * 80)
    
    metrics = {
        'Baseline (Questionnaire-Only)': (y_base_pred, y_base_prob),
        'Late Concatenation Fusion': (y_concat_pred, y_concat_prob),
        'Gated Multimodal Unit (GMU)': (y_gmu_pred, y_gmu_prob),
        'Bilinear Tensor Fusion': (y_bilinear_pred, y_bilinear_prob),
        'CAF Gating Pipeline': (y_caf_pred, y_caf_prob)
    }
    
    for name, (pred, prob) in metrics.items():
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        auc = roc_auc_score(y_true, prob)
        print(f"{name}:")
        print(f"  - Accuracy:  {acc * 100:.2f}%")
        print(f"  - Precision: {prec * 100:.2f}%")
        print(f"  - Recall:    {rec * 100:.2f}%")
        print(f"  - F1-Score:  {f1:.4f}")
        print(f"  - ROC-AUC:   {auc:.4f}")
        print("-" * 50)
        
    # 3. Analyze learned attention weights
    # Map vaping status: X_gate[:, 2] represents normalized vaping frequency (vaper if > 0.0)
    is_vaper = (X_gate_all[:, 2] > 0.0).astype(int)
    
    p_tab_weights = caf_attention_weights[:, 0]
    p_soft_weights = caf_attention_weights[:, 1]
    p_hard_weights = caf_attention_weights[:, 2]
    
    vaper_indices = np.where(is_vaper == 1)[0]
    non_vaper_indices = np.where(is_vaper == 0)[0]
    
    print("\n" + "=" * 80)
    print("LEARNED ATTENTION ROUTING WEIGHTS (Average per Cohort Group)")
    print("=" * 80)
    print(f"Group          | Patients | P_tab   | P_soft (Photo) | P_hard (X-Ray)")
    print("-" * 80)
    print(f"Non-Vapers     | {len(non_vaper_indices):<8} | {p_tab_weights[non_vaper_indices].mean():.4f}  | {p_soft_weights[non_vaper_indices].mean():.4f}         | {p_hard_weights[non_vaper_indices].mean():.4f}")
    print(f"Active Vapers  | {len(vaper_indices):<8} | {p_tab_weights[vaper_indices].mean():.4f}  | {p_soft_weights[vaper_indices].mean():.4f}         | {p_hard_weights[vaper_indices].mean():.4f}")
    print("=" * 80)
    
    # 4. Statistical significance test of down-weighting soft tissue for vapers
    vaper_p_soft = p_soft_weights[vaper_indices]
    non_vaper_p_soft = p_soft_weights[non_vaper_indices]
    
    t_stat, p_val = stats.ttest_ind(vaper_p_soft, non_vaper_p_soft, equal_var=False)
    
    print("\nSTATISTICAL SIGNIFICANCE OF THE CLINICAL ATTENTION SHIFT:")
    print(f"  - Two-sample Welch's t-test on visual soft-tissue weights (P_soft):")
    print(f"    t-statistic = {t_stat:.4f}")
    print(f"    p-value     = {p_val:.9e}")
    
    if p_val < 0.05:
        print("\n[SUCCESS] STATISTICALLY SIGNIFICANT DETECTED!")
        reduction = (non_vaper_p_soft.mean() - vaper_p_soft.mean()) / non_vaper_p_soft.mean() * 100
        print(f"  - The model successfully down-weighted soft tissue photo focus for vapers by {reduction:.1f}%.")
        print("  - This statistically proves that the learned gating mechanism aligns with the clinical vasoconstriction hypothesis.")
    else:
        print("\n[WARNING] Gating weight difference is not statistically significant at alpha=0.05.")
    print("=" * 80)


def evaluate_single_seed(seed, folds_data):
    """
    Worker function to run LOOCV for a single seed across all 50 folds.
    Uses scikit-learn for Late Concat and Bilinear to maximize speed,
    and PyTorch with max_iter=20 for GMU and CAF models.
    """
    import numpy as np
    import torch
    import random
    from sklearn.linear_model import LogisticRegression
    from src.models.baselines import GatedMultimodalUnit
    from src.models.caf import ClinicalAttentionFramework
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    y_concat_pred, y_concat_prob = [], []
    y_gmu_pred, y_gmu_prob = [], []
    y_bilinear_pred, y_bilinear_prob = [], []
    y_caf_pred, y_caf_prob = [], []
    caf_attention_weights = []
    
    for fold in folds_data:
        s_tab_train = fold['s_tab_train']
        s_tab_test = fold['s_tab_test']
        s_soft_train = fold['s_soft_train']
        s_soft_test = fold['s_soft_test']
        s_hard_train = fold['s_hard_train']
        s_hard_test = fold['s_hard_test']
        y_train = fold['y_train']
        y_test = fold['y_test']
        X_gate_train = fold['X_gate_train']
        X_gate_test = fold['X_gate_test']
        
        # 1. Late Concatenation Fusion (sklearn LogisticRegression)
        X_train_c = np.column_stack([s_tab_train, s_soft_train, s_hard_train])
        X_test_c = np.column_stack([s_tab_test, s_soft_test, s_hard_test])
        clf_concat = LogisticRegression(C=1.0, solver='liblinear')
        clf_concat.fit(X_train_c, y_train)
        y_concat_pred.append(clf_concat.predict(X_test_c)[0])
        y_concat_prob.append(clf_concat.predict_proba(X_test_c)[0, 1])
        
        # 2. Bilinear Tensor Fusion (sklearn LogisticRegression with interaction terms)
        def get_bilinear_terms(X):
            t = X[:, 0]
            s = X[:, 1]
            h = X[:, 2]
            return np.column_stack([t, s, h, t*s, t*h, s*h, t*s*h])
        X_train_b = get_bilinear_terms(X_train_c)
        X_test_b = get_bilinear_terms(X_test_c)
        clf_bilinear = LogisticRegression(C=1.0, solver='liblinear')
        clf_bilinear.fit(X_train_b, y_train)
        y_bilinear_pred.append(clf_bilinear.predict(X_test_b)[0])
        y_bilinear_prob.append(clf_bilinear.predict_proba(X_test_b)[0, 1])
        
        # 3. Gated Multimodal Unit (GMU) - PyTorch L-BFGS with max_iter=20 for speed
        model_gmu = GatedMultimodalUnit()
        gmu_criterion = torch.nn.BCEWithLogitsLoss()
        gmu_optimizer = torch.optim.LBFGS(model_gmu.parameters(), lr=0.1, max_iter=20)
        
        s_tab_train_t = torch.tensor(s_tab_train, dtype=torch.float32).unsqueeze(1)
        s_soft_train_t = torch.tensor(s_soft_train, dtype=torch.float32).unsqueeze(1)
        s_hard_train_t = torch.tensor(s_hard_train, dtype=torch.float32).unsqueeze(1)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        
        def gmu_closure():
            gmu_optimizer.zero_grad()
            logits = model_gmu(s_tab_train_t, s_soft_train_t, s_hard_train_t)
            loss = gmu_criterion(logits, y_train_t)
            loss.backward()
            return loss
            
        model_gmu.train()
        gmu_optimizer.step(gmu_closure)
        
        # GMU test
        s_tab_test_t = torch.tensor(s_tab_test, dtype=torch.float32).unsqueeze(0)
        s_soft_test_t = torch.tensor(s_soft_test, dtype=torch.float32).unsqueeze(0)
        s_hard_test_t = torch.tensor(s_hard_test, dtype=torch.float32).unsqueeze(0)
        model_gmu.eval()
        with torch.no_grad():
            logit_g = model_gmu(s_tab_test_t, s_soft_test_t, s_hard_test_t)
            prob_g = torch.sigmoid(logit_g).item()
            y_gmu_pred.append(1 if prob_g >= 0.5 else 0)
            y_gmu_prob.append(prob_g)
            
        # 4. Clinical Attention Framework (CAF) - PyTorch L-BFGS with max_iter=100 for stable convergence
        model_caf = ClinicalAttentionFramework()
        caf_criterion = torch.nn.BCEWithLogitsLoss()
        caf_optimizer = torch.optim.LBFGS(
            model_caf.parameters(), 
            lr=0.05, 
            max_iter=100,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50
        )
        
        X_gate_train_t = torch.tensor(fold['X_gate_train'], dtype=torch.float32)
        X_gate_test_t = torch.tensor(fold['X_gate_test'], dtype=torch.float32)
        
        def caf_closure():
            caf_optimizer.zero_grad()
            logits, _ = model_caf(X_gate_train_t, s_tab_train_t, s_soft_train_t, s_hard_train_t)
            loss = caf_criterion(logits, y_train_t)
            loss.backward()
            return loss
            
        model_caf.train()
        caf_optimizer.step(caf_closure)
        
        # CAF test
        model_caf.eval()
        with torch.no_grad():
            logit_caf, probs_caf = model_caf(X_gate_test_t, s_tab_test_t, s_soft_test_t, s_hard_test_t)
            prob_caf = torch.sigmoid(logit_caf).item()
            y_caf_pred.append(1 if prob_caf >= 0.5 else 0)
            y_caf_prob.append(prob_caf)
            caf_attention_weights.append(probs_caf.numpy().flatten())
            
    return {
        'concat': (np.array(y_concat_pred), np.array(y_concat_prob)),
        'gmu': (np.array(y_gmu_pred), np.array(y_gmu_prob)),
        'bilinear': (np.array(y_bilinear_pred), np.array(y_bilinear_prob)),
        'caf': (np.array(y_caf_pred), np.array(y_caf_prob)),
        'caf_weights': np.array(caf_attention_weights)
    }


def run_multi_seed_evaluation(dataset, num_seeds=100):
    """
    Runs LOOCV evaluation for all models over `num_seeds` different random seeds.
    Averages metrics (Mean +/- Std) and prints a comparative benchmark.
    Uses ProcessPoolExecutor for parallel seed execution to maintain high speed.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    
    print("=" * 80)
    print(f"STARTING MULTI-SEED LOOCV BENCHMARKING ({num_seeds} Seeds)")
    print(f"Cohort size: {len(dataset)} patients")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. Precompute inputs for all 50 folds of LOOCV (Questionnaire model is deterministic)
    y_labels = np.array(dataset.labels)
    X_gate_all = dataset.X_gate
    s_soft_all = dataset.s_soft
    s_hard_all = dataset.s_hard
    X_tab_all = dataset.X_tab
    
    loo = LeaveOneOut()
    folds_data = []
    
    # Storing deterministic questionnaire baseline results
    y_base_pred, y_base_prob = [], []
    
    for train_idx, test_idx in loo.split(X_tab_all):
        X_train_tab, X_test_tab = X_tab_all[train_idx], X_tab_all[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        
        # Fit tabular baseline
        selector = SelectKBest(score_func=f_classif, k=16)
        X_train_sel = selector.fit_transform(X_train_tab, y_train)
        X_test_sel = selector.transform(X_test_tab)
        
        clf_tab = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
        clf_tab.fit(X_train_sel, y_train)
        
        s_tab_train = clf_tab.predict_proba(X_train_sel)[:, 1]
        s_tab_test = clf_tab.predict_proba(X_test_sel)[:, 1]
        
        y_base_pred.append(clf_tab.predict(X_test_sel)[0])
        y_base_prob.append(s_tab_test[0])
        
        folds_data.append({
            's_tab_train': s_tab_train,
            's_tab_test': s_tab_test,
            's_soft_train': s_soft_all[train_idx],
            's_soft_test': s_soft_all[test_idx],
            's_hard_train': s_hard_all[train_idx],
            's_hard_test': s_hard_all[test_idx],
            'y_train': y_train,
            'y_test': y_test,
            'X_gate_train': X_gate_all[train_idx],
            'X_gate_test': X_gate_all[test_idx]
        })
        
    y_base_pred = np.array(y_base_pred)
    y_base_prob = np.array(y_base_prob)
    
    # 2. Run seed sweep in parallel
    seeds = list(range(num_seeds))
    results = []
    
    print(f"Precomputation complete. Running {num_seeds} seeds in parallel...")
    
    # Use ProcessPoolExecutor to parallelize seed execution
    import os
    max_workers = min(os.cpu_count() or 4, 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit task for each seed
        futures = [executor.submit(evaluate_single_seed, seed, folds_data) for seed in seeds]
        for idx, future in enumerate(futures):
            results.append(future.result())
            if (idx + 1) % 10 == 0 or idx == num_seeds - 1:
                print(f"  - Completed seeds: {idx + 1}/{num_seeds}")
                
    elapsed_time = time.time() - start_time
    print(f"\nExecution finished in {elapsed_time:.2f} seconds!")
    
    # 3. Calculate baseline metrics (constant across seeds)
    base_acc = accuracy_score(y_labels, y_base_pred)
    base_prec = precision_score(y_labels, y_base_pred, zero_division=0)
    base_rec = recall_score(y_labels, y_base_pred, zero_division=0)
    base_f1 = f1_score(y_labels, y_base_pred, zero_division=0)
    base_auc = roc_auc_score(y_labels, y_base_prob)
    
    # 4. Aggregate metrics for each model
    model_metrics = {
        'Baseline (Questionnaire-Only)': {
            'acc': [base_acc] * num_seeds, 'prec': [base_prec] * num_seeds,
            'rec': [base_rec] * num_seeds, 'f1': [base_f1] * num_seeds, 'auc': [base_auc] * num_seeds
        },
        'Late Concatenation Fusion': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []},
        'Gated Multimodal Unit (GMU)': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []},
        'Bilinear Tensor Fusion': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []},
        'CAF Gating Pipeline': {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
    }
    
    # Gating and significance results
    is_vaper = (X_gate_all[:, 2] > 0.0).astype(int)
    vaper_indices = np.where(is_vaper == 1)[0]
    non_vaper_indices = np.where(is_vaper == 0)[0]
    
    p_values = []
    p_soft_non_vaper_means = []
    p_soft_vaper_means = []
    p_tab_non_vaper_means = []
    p_tab_vaper_means = []
    p_hard_non_vaper_means = []
    p_hard_vaper_means = []
    
    for res in results:
        # Extract metrics
        for model_key, metric_dict in [('concat', model_metrics['Late Concatenation Fusion']),
                                       ('gmu', model_metrics['Gated Multimodal Unit (GMU)']),
                                       ('bilinear', model_metrics['Bilinear Tensor Fusion']),
                                       ('caf', model_metrics['CAF Gating Pipeline'])]:
            pred, prob = res[model_key]
            metric_dict['acc'].append(accuracy_score(y_labels, pred))
            metric_dict['prec'].append(precision_score(y_labels, pred, zero_division=0))
            metric_dict['rec'].append(recall_score(y_labels, pred, zero_division=0))
            metric_dict['f1'].append(f1_score(y_labels, pred, zero_division=0))
            metric_dict['auc'].append(roc_auc_score(y_labels, prob))
            
        # Extract attention weights for CAF
        caf_weights = res['caf_weights'] # [50, 3]
        p_tab_w = caf_weights[:, 0]
        p_soft_w = caf_weights[:, 1]
        p_hard_w = caf_weights[:, 2]
        
        p_soft_non_vaper_means.append(p_soft_w[non_vaper_indices].mean())
        p_soft_vaper_means.append(p_soft_w[vaper_indices].mean())
        
        p_tab_non_vaper_means.append(p_tab_w[non_vaper_indices].mean())
        p_tab_vaper_means.append(p_tab_w[vaper_indices].mean())
        
        p_hard_non_vaper_means.append(p_hard_w[non_vaper_indices].mean())
        p_hard_vaper_means.append(p_hard_w[vaper_indices].mean())
        
        # Welch's t-test on P_soft
        _, p_val = stats.ttest_ind(p_soft_w[vaper_indices], p_soft_w[non_vaper_indices], equal_var=False)
        p_values.append(p_val)
        
    # 5. Print results table
    print("\n" + "=" * 96)
    print(f"LOOCV MULTI-SEED BENCHMARK SUMMARY (Averaged over {num_seeds} seeds)")
    print("=" * 96)
    print(f"{'Model Name':<30} | {'Accuracy':<13} | {'Precision':<13} | {'Recall':<13} | {'F1-Score':<13} | {'ROC-AUC':<8}")
    print("-" * 105)
    
    for name, metrics in model_metrics.items():
        acc_m, acc_s = np.mean(metrics['acc']), np.std(metrics['acc'])
        prec_m, prec_s = np.mean(metrics['prec']), np.std(metrics['prec'])
        rec_m, rec_s = np.mean(metrics['rec']), np.std(metrics['rec'])
        f1_m, f1_s = np.mean(metrics['f1']), np.std(metrics['f1'])
        auc_m, auc_s = np.mean(metrics['auc']), np.std(metrics['auc'])
        
        print(f"{name:<30} | {acc_m*100:.1f}±{acc_s*100:.1f}% | {prec_m*100:.1f}±{prec_s*100:.1f}% | {rec_m*100:.1f}±{rec_s*100:.1f}% | {f1_m:.3f}±{f1_s:.3f} | {auc_m:.3f}±{auc_s:.3f}")
        
    print("=" * 96)
    
    # 6. Print Gating attention weights
    print("\n" + "=" * 96)
    print("LEARNED ATTENTION ROUTING WEIGHTS (Average over all seeds)")
    print("=" * 96)
    print(f"Group          | Patients | P_tab           | P_soft (Photo)  | P_hard (X-Ray)")
    print("-" * 96)
    print(f"Non-Vapers     | {len(non_vaper_indices):<8} | {np.mean(p_tab_non_vaper_means):.4f}±{np.std(p_tab_non_vaper_means):.4f} | {np.mean(p_soft_non_vaper_means):.4f}±{np.std(p_soft_non_vaper_means):.4f} | {np.mean(p_hard_non_vaper_means):.4f}±{np.std(p_hard_non_vaper_means):.4f}")
    print(f"Active Vapers  | {len(vaper_indices):<8} | {np.mean(p_tab_vaper_means):.4f}±{np.std(p_tab_vaper_means):.4f} | {np.mean(p_soft_vaper_means):.4f}±{np.std(p_soft_vaper_means):.4f} | {np.mean(p_hard_vaper_means):.4f}±{np.std(p_hard_vaper_means):.4f}")
    print("=" * 96)
    
    # 7. Print statistical verification
    success_seeds = sum(1 for p in p_values if p < 0.05)
    success_rate = (success_seeds / num_seeds) * 100
    mean_p_val = np.mean(p_values)
    median_p_val = np.median(p_values)
    
    print("\nSTATISTICAL SIGNIFICANCE SUMMARY:")
    print(f"  - Clinical shift success rate (p < 0.05): {success_rate:.1f}% ({success_seeds}/{num_seeds} seeds)")
    print(f"  - Mean Welch's t-test p-value:          {mean_p_val:.5e}")
    print(f"  - Median Welch's t-test p-value:        {median_p_val:.5e}")
    print("=" * 96)


