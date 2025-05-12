import numpy as np
import pandas as pd
import argparse
import os

def generate_pa_dataset(n_samples=1000, seed=42):
    np.random.seed(seed)

    # Simulate features
    age = np.random.randint(30, 90, size=n_samples)
    sex = np.random.binomial(1, 0.5, size=n_samples)  
    mcv = np.random.normal(92, 10, size=n_samples)    
    hemoglobin = np.random.normal(13.5, 1.5, size=n_samples)
    b12 = np.random.normal(300, 100, size=n_samples)
    folate = np.random.normal(7, 2, size=n_samples)
    intrinsic_factor_ab = np.random.binomial(1, 0.1, size=n_samples)
    fatigue = np.random.binomial(1, 0.3, size=n_samples)
    autoimmune = np.random.binomial(1, 0.15, size=n_samples)
    on_ppi = np.random.binomial(1, 0.2, size=n_samples)

    # Label generation: Rule-based signal for PA likelihood
    score = (
        (b12 < 200).astype(int) +
        (mcv > 100).astype(int) +
        intrinsic_factor_ab +
        fatigue +
        autoimmune
    )
    diagnosed_pa = (score >= 3).astype(int)

    # Assemble dataset
    df = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'MCV': mcv,
        'Hemoglobin': hemoglobin,
        'B12_Level': b12,
        'Folate_Level': folate,
        'Intrinsic_Factor_Ab': intrinsic_factor_ab,
        'Fatigue_Symptom': fatigue,
        'Autoimmune_History': autoimmune,
        'On_PPI': on_ppi,
        'Diagnosed_PA': diagnosed_pa
    })

    return df


def save_dataset(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved to: {output_path}")
    

def main():
    parser = argparse.ArgumentParser(description="Generate simulated dataset for Pernicious Anaemia ML research.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic patient records to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="data/simulated_pa_data.csv", help="Path to save the output CSV.")

    args = parser.parse_args()

    print("📊 Generating simulated dataset...")
    df = generate_pa_dataset(n_samples=args.samples, seed=args.seed)
    save_dataset(df, args.output)


if __name__ == "__main__":
    main()
    
    
# Run from command line = python generate_pa_data.py --samples 1000 --seed 123 --output data/pa_dataset.csv
