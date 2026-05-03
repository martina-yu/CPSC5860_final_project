
import numpy as np
import pandas as pd
import io, gzip

cohort = pd.read_csv("data/cohort_labels.csv")
prs_df = pd.read_csv("data/prs_weights.csv") if False else None

r = "./PGS000004_hmPOS_GRCh38.txt.gz"

with gzip.open(io.BytesIO(r.content), 'rt') as f:
    lines = f.readlines()
data_lines = [l for l in lines if not l.startswith('#')]
prs_df = pd.read_csv(io.StringIO(''.join(data_lines)), sep='\t')
prs_df = prs_df.dropna(subset=['effect_weight', 'allelefrequency_effect'])
prs_df = prs_df[(prs_df['allelefrequency_effect'] > 0.01) & 
                (prs_df['allelefrequency_effect'] < 0.99)]
prs_df = prs_df.reset_index(drop=True)
betas = prs_df['effect_weight'].values
mafs  = prs_df['allelefrequency_effect'].values
M = len(prs_df)   # number of SNPs
N = len(cohort)   # number of patients
labels = cohort['label'].values  # 0=early, 1=late

#%%
print(f"SNPs after QC: {M}")
print(f"Patients: {N}  (early=0: {(labels==0).sum()}, late=1: {(labels==1).sum()})")
print(f"Beta range: [{betas.min():.4f}, {betas.max():.4f}]")
print(f"MAF range:  [{mafs.min():.4f}, {mafs.max():.4f}]")

np.random.seed(42)
SIGNAL_DELTA = 0.04

#%%
X = np.zeros((N, M), dtype=np.float32)

for i, lbl in enumerate(labels):
    if lbl == 1:   # late-onset → higher PRS → boost positive-beta SNPs
        adj_maf = np.clip(mafs + SIGNAL_DELTA * np.sign(betas), 0.01, 0.99)
    else:          # early-onset → lower PRS → boost negative-beta SNPs
        adj_maf = np.clip(mafs - SIGNAL_DELTA * np.sign(betas), 0.01, 0.99)
    X[i] = np.random.binomial(2, adj_maf).astype(np.float32)

print(f"\nGenotype matrix shape: {X.shape}")
print(f"Genotype value counts: 0→{(X==0).sum()}, 1→{(X==1).sum()}, 2→{(X==2).sum()}")

#%% Compute PRS for each patient
PRS = X @ betas   # shape (N,)
print(f"\nPRS stats:")
print(f"  Early-onset: mean={PRS[labels==0].mean():.4f}, std={PRS[labels==0].std():.4f}")
print(f"  Late-onset:  mean={PRS[labels==1].mean():.4f}, std={PRS[labels==1].std():.4f}")

# Save
np.save("data/X_genotype.npy", X)
np.save("data/PRS.npy", PRS)
np.save("data/labels.npy", labels)
prs_df.to_csv("data/prs_weights.csv", index=False)
cohort['PRS'] = PRS
cohort.to_csv("data/cohort_with_prs.csv", index=False)



### result

# SNPs after QC: 311
# Patients: 562  (early=0: 74, late=1: 488)
# Beta range: [-0.2609, 0.2017]
# MAF range:  [0.0115, 0.9846]

# Genotype matrix shape: (562, 311)
# Genotype value counts: 0→82895, 1→61130, 2→30757

# PRS stats:
#   Early-onset: mean=-1.9414, std=0.5226
#   Late-onset:  mean=0.9824, std=0.5619

# All data saved

#%% Calibrate signal delta

etas = prs_df['effect_weight'].values
mafs  = prs_df['allelefrequency_effect'].values
M = len(prs_df)
N = len(cohort)
labels = cohort['label'].values

for delta in [0.005, 0.008, 0.010, 0.012, 0.015]:
    np.random.seed(42)
    X_test = np.zeros((N, M), dtype=np.float32)
    for i, lbl in enumerate(labels):
        if lbl == 1:
            adj_maf = np.clip(mafs + delta * np.sign(betas), 0.01, 0.99)
        else:
            adj_maf = np.clip(mafs - delta * np.sign(betas), 0.01, 0.99)
        X_test[i] = np.random.binomial(2, adj_maf).astype(np.float32)
    prs_test = X_test @ betas
    mu_e = prs_test[labels==0].mean(); sd_e = prs_test[labels==0].std()
    mu_l = prs_test[labels==1].mean(); sd_l = prs_test[labels==1].std()
    pooled_sd = np.sqrt((sd_e**2 + sd_l**2) / 2)
    cohens_d = abs(mu_l - mu_e) / pooled_sd
    print(f"delta={delta:.3f} | early={mu_e:.3f}±{sd_e:.3f}, late={mu_l:.3f}±{sd_l:.3f} | Cohen's d={cohens_d:.3f}")
    
### result
# delta=0.005 | early=-0.704±0.473, late=-0.278±0.593 | Cohen's d=0.795
# delta=0.008 | early=-0.846±0.457, late=-0.169±0.590 | Cohen's d=1.285
# delta=0.010 | early=-0.923±0.467, late=-0.099±0.588 | Cohen's d=1.552
# delta=0.012 | early=-0.995±0.461, late=-0.029±0.585 | Cohen's d=1.836
# delta=0.015 | early=-1.088±0.469, late=0.084±0.585 | Cohen's d=2.208