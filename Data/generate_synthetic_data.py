#!/usr/bin/env python
# Scripts/generate_full_bloodwork_pa_data.py

import numpy as np
import os
import pandas as pd

rng = np.random.default_rng(42)

def generate_full_bloodwork(n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2

    # -----------------------
    # Negative class (no PA)
    # -----------------------
    neg = pd.DataFrame({
        # CBC
        "Haemoglobin": rng.normal(14.0, 1.2, n_per_class),
        "Haematocrit": rng.normal(42, 3, n_per_class),
        "MCV": rng.normal(90, 5, n_per_class),
        "MCH": rng.normal(30, 2, n_per_class),
        "MCHC": rng.normal(34, 1, n_per_class),
        "RDW": rng.normal(13, 1, n_per_class),
        "WBC": rng.normal(6.5, 1.5, n_per_class),
        "Platelets": rng.normal(250, 50, n_per_class),

        # B12 / folate
        "Serum_B12": rng.normal(400, 120, n_per_class),
        "Folate": rng.normal(9, 3, n_per_class),
        "Methylmalonic_Acid": rng.normal(250, 60, n_per_class),
        "Homocysteine": rng.normal(12, 4, n_per_class),

        # Iron studies
        "Serum_Iron": rng.normal(90, 20, n_per_class),
        "Ferritin": rng.lognormal(mean=3.8, sigma=0.4, size=n_per_class),
        "TIBC": rng.normal(320, 40, n_per_class),
        "Transferrin_Saturation": rng.normal(28, 8, n_per_class),

        # Chemistry
        "LDH": rng.normal(180, 40, n_per_class),
        "Bilirubin": rng.normal(0.7, 0.2, n_per_class),
        "Creatinine": rng.normal(0.9, 0.2, n_per_class),
        "ALT": rng.normal(22, 8, n_per_class),
        "AST": rng.normal(24, 7, n_per_class),
    })
    neg["Diagnosed_PA"] = 0

    # -----------------------
    # Positive class (PA)
    # -----------------------
    pos = pd.DataFrame({
        # CBC
        "Haemoglobin": rng.normal(10.5, 1.4, n_per_class),
        "Haematocrit": rng.normal(32, 4, n_per_class),
        "MCV": rng.normal(104, 6, n_per_class),   # macrocytosis
        "MCH": rng.normal(34, 2, n_per_class),
        "MCHC": rng.normal(32, 1, n_per_class),
        "RDW": rng.normal(17, 2, n_per_class),
        "WBC": rng.normal(4.0, 1.2, n_per_class),
        "Platelets": rng.normal(180, 60, n_per_class),

        # B12 / folate
        "Serum_B12": rng.normal(160, 50, n_per_class),
        "Folate": rng.normal(8, 2.5, n_per_class),
        "Methylmalonic_Acid": rng.normal(700, 150, n_per_class),
        "Homocysteine": rng.normal(28, 8, n_per_class),

        # Iron studies
        "Serum_Iron": rng.normal(85, 25, n_per_class),
        "Ferritin": rng.lognormal(mean=3.6, sigma=0.5, size=n_per_class),
        "TIBC": rng.normal(310, 50, n_per_class),
        "Transferrin_Saturation": rng.normal(26, 9, n_per_class),

        # Chemistry
        "LDH": rng.normal(350, 80, n_per_class),
        "Bilirubin": rng.normal(1.2, 0.4, n_per_class),
        "Creatinine": rng.normal(1.0, 0.2, n_per_class),
        "ALT": rng.normal(24, 10, n_per_class),
        "AST": rng.normal(27, 9, n_per_class),
    })
    pos["Diagnosed_PA"] = 1

    # -----------------------
    # Combine & shuffle
    # -----------------------
    df = pd.concat([neg, pos], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = generate_full_bloodwork(n_samples=1000, seed=2025)
    out_path = os.path.join(os.path.dirname(__file__), "simulated_pa_full_bloodwork.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Synthetic full bloodwork dataset saved to {out_path}")
    print(df.head())
