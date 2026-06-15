import os
import sys
import torch

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import ClinicalAttentionDataset

def test_clinical_attention_dataset():
    print("=" * 60)
    print("RUNNING PYTORCH DATASET INTEGRITY TEST")
    print("=" * 60)
    
    cleaned_path = os.path.join(project_root, "data", "questionnaire_cleaned.xlsx")
    
    # 1. Instantiate dataset
    try:
        dataset = ClinicalAttentionDataset(cleaned_path)
        print("[SUCCESS] Dataset successfully instantiated.")
    except Exception as e:
        print(f"[ERROR] Failed to instantiate dataset: {e}")
        sys.exit(1)
        
    # 2. Check length (must be exactly 50)
    cohort_len = len(dataset)
    print(f"[SUCCESS] Cohort size: {cohort_len} patients")
    assert cohort_len == 50, f"Expected 50 patients, but found {cohort_len}"
    
    # 3. Inspect a sample record
    sample = dataset[0]
    print("\nSample Record Tensors:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  - {key}: shape={list(val.shape)}, dtype={val.dtype}")
        else:
            print(f"  - {key}: value={val} (type={type(val)})")
            
    # 4. Run shape and range assertions
    assert sample['x_gate'].shape == (5,), f"Expected x_gate shape (5,), got {sample['x_gate'].shape}"
    assert sample['s_soft'].shape == (1,), f"Expected s_soft shape (1,), got {sample['s_soft'].shape}"
    assert sample['s_hard'].shape == (1,), f"Expected s_hard shape (1,), got {sample['s_hard'].shape}"
    assert sample['label'].shape == (), f"Expected label shape (), got {sample['label'].shape}"
    
    # Check that all features in self.X_tab, X_gate, s_soft, s_hard have zero NaNs
    nan_found = False
    for i in range(len(dataset)):
        record = dataset[i]
        for key in ['x_gate', 's_soft', 's_hard', 'x_tab', 'label']:
            if torch.isnan(record[key]).any():
                print(f"[ERROR] NaN found in patient {record['patient_id']} feature '{key}'!")
                nan_found = True
                
    assert not nan_found, "Dataset contains NaN values!"
    print("[SUCCESS] Zero NaN values present in the entire dataset.")
    
    # 5. Check ranges
    print("\nFeature Range Check (Min / Max / Mean):")
    # Retrieve all inputs for full cohort statistics
    all_x_gate = torch.stack([dataset[i]['x_gate'] for i in range(len(dataset))])
    all_s_soft = torch.stack([dataset[i]['s_soft'] for i in range(len(dataset))])
    all_s_hard = torch.stack([dataset[i]['s_hard'] for i in range(len(dataset))])
    
    print(f"  - x_gate: Min={all_x_gate.min(dim=0)[0].tolist()}, Max={all_x_gate.max(dim=0)[0].tolist()}")
    print(f"  - s_soft: Min={all_s_soft.min().item():.4f}, Max={all_s_soft.max().item():.4f}, Mean={all_s_soft.mean().item():.4f}")
    print(f"  - s_hard: Min={all_s_hard.min().item():.4f}, Max={all_s_hard.max().item():.4f}, Mean={all_s_hard.mean().item():.4f}")
    
    # Verify s_soft, s_hard and x_gate are bounded in [0.0, 1.0]
    assert (all_x_gate >= 0.0).all() and (all_x_gate <= 1.0).all(), "x_gate features out of bounds [0.0, 1.0]!"
    assert (all_s_soft >= 0.0).all() and (all_s_soft <= 1.0).all(), "s_soft features out of bounds [0.0, 1.0]!"
    assert (all_s_hard >= 0.0).all() and (all_s_hard <= 1.0).all(), "s_hard features out of bounds [0.0, 1.0]!"
    print("[SUCCESS] All gating inputs (x_gate, s_soft, s_hard) are successfully bounded in [0.0, 1.0] range.")
    
    print("\n[SUCCESS] ALL DATA INTEGRITY AND SHAPE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_clinical_attention_dataset()

