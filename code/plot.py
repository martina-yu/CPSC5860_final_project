
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import pickle
from sklearn.metrics import roc_auc_score, f1_score
from graph_nn import GCN, GAT, build_and_train
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve

#%% PRS calculate
SIGNAL_DELTA = 0.04
cohort = pd.read_csv("data/cohort_labels.csv")
prs_df = pd.read_csv("data/prs_weights.csv")

betas = prs_df['effect_weight'].values
mafs  = prs_df['allelefrequency_effect'].values
M = len(prs_df)
N = len(cohort)
labels = cohort['label'].values

X = np.zeros((N, M), dtype=np.float32)
for i, lbl in enumerate(labels):
    if lbl == 1:   # late-onset: slightly higher PRS
        adj_maf = np.clip(mafs + SIGNAL_DELTA * np.sign(betas), 0.01, 0.99)
    else:           # early-onset: slightly lower PRS
        adj_maf = np.clip(mafs - SIGNAL_DELTA * np.sign(betas), 0.01, 0.99)
    X[i] = np.random.binomial(2, adj_maf).astype(np.float32)

PRS = X @ betas

#%% t-SNE

X = np.load("data/X_genotype.npy")
labels = np.load("data/labels.npy")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_2d = tsne.fit_transform(X_scaled)

#%% parameters
fig, ax = plt.subplots(figsize=(6, 5))
palette = {0: '#0279EE', 1: '#FF9400'}
label_names = {0: 'Early-onset (<40)', 1: 'Late-onset (>60)'}

for lbl, color in palette.items():
    mask = labels == lbl
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=color, label=label_names[lbl],
               alpha=0.65, s=18, linewidths=0)

ax.set_xlabel('t-SNE 1', fontsize=11)
ax.set_ylabel('t-SNE 2', fontsize=11)
ax.set_title('t-SNE of Simulated Patient SNP Vectors\n(colored by age-onset group)', fontsize=11)
ax.legend(frameon=True, fontsize=9)
sns.despine(ax=ax)
plt.tight_layout()

plt.savefig("results/02_eda/fig1_tsne.png", dpi=150, bbox_inches='tight')
plt.savefig("results/02_eda/fig1_tsne.svg", bbox_inches='tight')
plt.close()


#%% violin plot
df_prs = pd.DataFrame({'PRS': PRS,
                        'Group': ['Early-onset (<40)' if l==0 else 'Late-onset (>60)'
                                  for l in labels]})
fig, ax = plt.subplots(figsize=(5, 5))
sns.violinplot(data=df_prs, x='Group', y='PRS',
               palette={'Early-onset (<40)': '#0279EE', 'Late-onset (>60)': '#FF9400'},
               inner='box', cut=0, ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Polygenic Risk Score (PRS)', fontsize=11)
ax.set_title('PRS Distribution by Age-Onset Group\n(Simulated genotypes, PGS000004 weights)', fontsize=10)
mu_e, sd_e = PRS[labels==0].mean(), PRS[labels==0].std()
mu_l, sd_l = PRS[labels==1].mean(), PRS[labels==1].std()
pooled_sd = np.sqrt((sd_e**2 + sd_l**2) / 2)
d = abs(mu_l - mu_e) / pooled_sd
ax.text(0.97, 0.97, f"Cohen's d = {d:.2f}", transform=ax.transAxes,
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("results/02_eda/fig2_prs_violin.png", dpi=150, bbox_inches='tight')
plt.savefig("results/02_eda/fig2_prs_violin.svg", bbox_inches='tight')
plt.close()

#%% KNN degree
A = kneighbors_graph(X_scaled, n_neighbors=10, metric='cosine',
                     mode='connectivity', include_self=False)
A_sym = (A + A.T)  # symmetrize
A_sym.data[:] = 1.0
degrees = np.array(A_sym.sum(axis=1)).flatten().astype(int)

fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(degrees, bins=range(degrees.min(), degrees.max()+2),
        color='#75A025', edgecolor='white', linewidth=0.4, alpha=0.85)
ax.axvline(degrees.mean(), color='#E9ED4C', linewidth=2,
           linestyle='--', label=f'Mean degree = {degrees.mean():.1f}')
ax.set_xlabel('Node Degree', fontsize=11)
ax.set_ylabel('Number of Patients', fontsize=11)
ax.set_title('Degree Distribution of k-NN Patient Graph (k=10)', fontsize=11)
ax.legend(fontsize=9)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("results/02_eda/fig3_degree_hist.png", dpi=150, bbox_inches='tight')
plt.savefig("results/02_eda/fig3_degree_hist.svg", bbox_inches='tight')
plt.close()

#%% Model comparison bar chart
with open("data/baseline_results.pkl", "rb") as f:
    baseline = pickle.load(f)
with open("data/gnn_results.pkl", "rb") as f:
    gnn = pickle.load(f)
    
models_order = ['Majority', 'LR_PRS', 'MLP_SNP',
                'GCN\n(k=5)', 'GCN\n(k=10)', 'GCN\n(k=20)',
                'GAT\n(k=5)', 'GAT\n(k=10)', 'GAT\n(k=20)']
auroc_means, auroc_stds = [], []
f1_means, f1_stds = [], []

for m in ['Majority', 'LR_PRS', 'MLP_SNP']:
    auroc_means.append(np.mean(baseline[m]['auroc']))
    auroc_stds.append(np.std(baseline[m]['auroc']))
    f1_means.append(np.mean(baseline[m]['f1']))
    f1_stds.append(np.std(baseline[m]['f1']))

for model in ['GCN', 'GAT']:
    for k in [5, 10, 20]:
        auroc_means.append(np.mean(gnn[model][k]['auroc']))
        auroc_stds.append(np.std(gnn[model][k]['auroc']))
        f1_means.append(np.mean(gnn[model][k]['f1']))
        f1_stds.append(np.std(gnn[model][k]['f1']))

colors = ['#ECE9E2']*3 + ['#0279EE']*3 + ['#FF9400']*3

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(models_order))
w = 0.6

for ax, means, stds, title, ylabel in [
    (axes[0], auroc_means, auroc_stds, 'AUROC Comparison', 'AUROC'),
    (axes[1], f1_means,    f1_stds,    'Macro-F1 Comparison', 'Macro F1')
]:
    bars = ax.bar(x, means, width=w, color=colors,
                  yerr=stds, capsize=4, error_kw={'linewidth':1.2},
                  edgecolor='#333333', linewidth=0.6)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Random (0.5)')
    ax.set_xticks(x); ax.set_xticklabels(models_order, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=11); ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 0.95)
    # Annotate best GNN
    best_gnn_idx = np.argmax(means[3:]) + 3
    ax.annotate(f'{means[best_gnn_idx]:.3f}', xy=(x[best_gnn_idx], means[best_gnn_idx]+stds[best_gnn_idx]+0.02),
                ha='center', fontsize=7.5, color='#FF9400', fontweight='bold')
    ax.annotate(f'{means[1]:.3f}', xy=(x[1], means[1]+stds[1]+0.02),
                ha='center', fontsize=7.5, color='#333333', fontweight='bold')
    sns.despine(ax=ax)

legend_patches = [
    mpatches.Patch(color='#ECE9E2', edgecolor='#333', label='Baselines'),
    mpatches.Patch(color='#0279EE', label='GCN'),
    mpatches.Patch(color='#FF9400', label='GAT'),
]
axes[0].legend(handles=legend_patches, fontsize=8, loc='upper left')
plt.suptitle('Model Performance on Simulated TCGA-BRCA Early/Late-Onset Classification\n(5-seed mean ± std)', fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig("results/03_results/fig4_model_comparison.png", dpi=150, bbox_inches='tight')
plt.savefig("results/03_results/fig4_model_comparison.svg", bbox_inches='tight')
plt.close()

#%% Ablation — k vs AUROC for GCN and GAT
fig, ax = plt.subplots(figsize=(5.5, 4))
K_VALUES = [5, 10, 20]
for model, color, marker in [('GCN','#0279EE','o'), ('GAT','#FF9400','s')]:
    means = [np.mean(gnn[model][k]['auroc']) for k in K_VALUES]
    stds  = [np.std(gnn[model][k]['auroc'])  for k in K_VALUES]
    ax.errorbar(K_VALUES, means, yerr=stds, label=model, color=color,
                marker=marker, markersize=7, linewidth=2, capsize=5)

ax.axhline(np.mean(baseline['LR_PRS']['auroc']), color='#333333',
           linestyle='--', linewidth=1.5,
           label=f"LR_PRS baseline ({np.mean(baseline['LR_PRS']['auroc']):.3f})")
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6, label='Random (0.5)')
ax.set_xlabel('k (number of nearest neighbors)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('Ablation: Graph Density (k) vs AUROC\n(5-seed mean ± std, simulated data)', fontsize=10)
ax.set_xticks(K_VALUES); ax.set_ylim(0.35, 0.85)
ax.legend(fontsize=9); sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("results/03_results/fig5_ablation_k.png", dpi=150, bbox_inches='tight')
plt.savefig("results/03_results/fig5_ablation_k.svg", bbox_inches='tight')
plt.close()

#%% ROC curves

def roc_curves(pos_weight, from_scipy_sparse_matrix, tr_m, va_m, X_feat_gat, lr_probs,y_te):
    X_feat_gcn = PRS.reshape(-1,1).astype(np.float32)
    gcn_model = GCN(1, hidden=64, dropout=0.4)
    gcn_probs, _, _ = build_and_train(labels, pos_weight, from_scipy_sparse_matrix, X_scaled, gcn_model, X_feat_gcn, 10, tr_m, va_m, 42)

    gat_model = GAT(X_feat_gat.shape[1], hidden=32, heads=2, dropout=0.4)
    gat_probs, gat_data, gat_trained = build_and_train(gat_model, X_feat_gat, 10, tr_m, va_m, 42)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for probs, label, color in [
        (lr_probs,  f"LR_PRS (AUC={roc_auc_score(y_te,lr_probs):.3f})",  '#333333'),
        (gcn_probs, f"GCN k=10 (AUC={roc_auc_score(y_te,gcn_probs):.3f})", '#0279EE'),
        (gat_probs, f"GAT k=10 (AUC={roc_auc_score(y_te,gat_probs):.3f})", '#FF9400'),
    ]:
        fpr, tpr, _ = roc_curve(y_te, probs)
        ax.plot(fpr, tpr, label=label, linewidth=2)

    ax.plot([0,1],[0,1],'--', color='gray', linewidth=1, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves — Best Models (seed=42, k=10)\n[Simulated genotype data]', fontsize=10)
    ax.legend(fontsize=9, loc='lower right'); sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig("results/03_results/fig6_roc_curves.png", dpi=150, bbox_inches='tight')
    plt.savefig("results/03_results/fig6_roc_curves.svg", bbox_inches='tight')
    plt.close()
    return

#%% GAT attention weight distribution

gat_probs, gat_data, gat_trained = build_and_train(gat_model, X_feat_gat, 10, tr_m, va_m, 42)
gat_trained.eval()
with torch.no_grad():
    ei_att, alpha = gat_trained.get_attention(gat_data.x, gat_data.edge_index)
    alpha_np = alpha.mean(dim=1).numpy()  # average over heads

# Color edges by whether they connect same-class or cross-class nodes
src, dst = ei_att[0].numpy(), ei_att[1].numpy()
same_class = (labels[src] == labels[dst])

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(alpha_np[same_class],  bins=40, alpha=0.7, color='#75A025',
        label=f'Same-class edges (n={same_class.sum()})', density=True)
ax.hist(alpha_np[~same_class], bins=40, alpha=0.7, color='#FD9BED',
        label=f'Cross-class edges (n={(~same_class).sum()})', density=True)
ax.set_xlabel('GAT Attention Weight (Layer 1, mean over heads)', fontsize=10)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('GAT Attention Weight Distribution\nby Edge Type (k=10, seed=42)', fontsize=10)
ax.legend(fontsize=9); sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("results/03_results/fig7_gat_attention.png", dpi=150, bbox_inches='tight')
plt.savefig("results/03_results/fig7_gat_attention.svg", bbox_inches='tight')
plt.close()

#%% plot 2.1 Model comparison bar chart 
def model_comp_bar_chart(mpatches, all_results):
    cat_colors = {'Baseline': '#888888', 'Linear': '#0279EE', 'Non-linear': '#75A025', 'GNN': '#FF9400'}
    names  = list(all_results.keys())
    aucs   = [all_results[n]['auc']     for n in names]
    stds   = [all_results[n]['auc_std'] for n in names]
    cats   = [all_results[n]['cat']     for n in names]
    colors = [cat_colors[c] for c in cats]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, aucs, yerr=stds, capsize=4, color=colors, alpha=0.85,
                edgecolor='white', linewidth=0.8, error_kw={'elinewidth':1.2})
    ax.axhline(0.5,  color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Random (0.50)')
    ax.axhline(all_results['LR_PRS']['auc'], color='#0279EE', linestyle=':', linewidth=1.5,
            alpha=0.7, label=f"LR_PRS ({all_results['LR_PRS']['auc']:.3f})")
    ax.axhline(all_results['XGB_SNP100']['auc'], color='#75A025', linestyle=':', linewidth=1.5,
            alpha=0.7, label=f"XGB_SNP100 ({all_results['XGB_SNP100']['auc']:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('AUROC (mean ± SD, 5 seeds)', fontsize=11)
    ax.set_title('Model Comparison — Early vs Late-Onset Breast Cancer\n(Simulated TCGA-BRCA genotype data)', fontsize=12)
    ax.set_ylim(0.35, 0.80)
    ax.legend(fontsize=9, loc='upper left')

    # Category legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in cat_colors.items()]
    ax.legend(handles=patches + [
        plt.Line2D([0],[0], color='black', linestyle='--', label='Random (0.50)'),
        plt.Line2D([0],[0], color='#0279EE', linestyle=':', label=f"LR_PRS ref ({all_results['LR_PRS']['auc']:.3f})"),
        plt.Line2D([0],[0], color='#75A025', linestyle=':', label=f"XGB_SNP100 ref ({all_results['XGB_SNP100']['auc']:.3f})"),
    ], fontsize=8.5, loc='upper left', ncol=2)

    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('results/04_results_v2/fig1_model_comparison_v2.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/04_results_v2/fig1_model_comparison_v2.svg', bbox_inches='tight')
    plt.close()
    return

#%% pic 2.2 GNN k-ablation
def gnn_k_ablation(results_gnn):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    k_vals = [10, 20, 30]

    model_feat_best = {
        'GCN':       ('SNP+PRS', '#0279EE'),
        'GAT':       ('PRS',     '#FF9400'),
        'GraphSAGE': ('PRS',     '#75A025'),
    }
    for mname, (fname, color) in model_feat_best.items():
        aucs = [np.mean(results_gnn[f'{mname}|{fname}|k={k}']['aucs']) for k in k_vals]
        stds = [np.std( results_gnn[f'{mname}|{fname}|k={k}']['aucs']) for k in k_vals]
        ax.plot(k_vals, aucs, 'o-', color=color, label=f'{mname} ({fname})', linewidth=2, markersize=7)
        ax.fill_between(k_vals,
                        [a-s for a,s in zip(aucs,stds)],
                        [a+s for a,s in zip(aucs,stds)],
                        alpha=0.15, color=color)

    ax.axhline(0.648, color='#0279EE', linestyle='--', linewidth=1.2, alpha=0.6, label='LR_PRS (0.648)')
    ax.axhline(0.674, color='#75A025', linestyle='--', linewidth=1.2, alpha=0.6, label='XGB_SNP100 (0.674)')
    ax.axhline(0.500, color='gray',    linestyle=':',  linewidth=1.0, alpha=0.5, label='Random (0.500)')
    ax.set_xlabel('k (number of neighbors)', fontsize=11)
    ax.set_ylabel('AUROC (mean ± SD, 5 seeds)', fontsize=11)
    ax.set_title('GNN Performance vs Graph Connectivity k\n(Simulated data, best feature config per model)', fontsize=11)
    ax.set_xticks(k_vals)
    ax.set_ylim(0.35, 0.75)
    ax.legend(fontsize=9, loc='upper right')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('results/04_results_v2/fig2_k_ablation_v2.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/04_results_v2/fig2_k_ablation_v2.svg', bbox_inches='tight')
    plt.close()
    return

#%% pic 2.3, GAT, k10
def gat_k10(results_gnn):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    feat_labels = ['PRS\n(1-dim)', 'SNP100+PRS\n(101-dim)', 'Full\n(116-dim)']
    feat_keys   = ['PRS', 'SNP+PRS', 'Full']
    model_colors = {'GCN':'#0279EE', 'GAT':'#FF9400', 'GraphSAGE':'#75A025'}
    x = np.arange(len(feat_labels))
    width = 0.25

    for mi, (mname, color) in enumerate(model_colors.items()):
        aucs = [np.mean(results_gnn[f'{mname}|{fk}|k=10']['aucs']) for fk in feat_keys]
        stds = [np.std( results_gnn[f'{mname}|{fk}|k=10']['aucs']) for fk in feat_keys]
        ax.bar(x + mi*width, aucs, width, yerr=stds, capsize=4,
            color=color, alpha=0.82, label=mname, edgecolor='white',
            error_kw={'elinewidth':1.2})

    ax.axhline(0.648, color='#0279EE', linestyle='--', linewidth=1.2, alpha=0.6, label='LR_PRS (0.648)')
    ax.axhline(0.674, color='#75A025', linestyle='--', linewidth=1.2, alpha=0.6, label='XGB_SNP100 (0.674)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(feat_labels, fontsize=10)
    ax.set_ylabel('AUROC (mean ± SD, 5 seeds)', fontsize=11)
    ax.set_title('Feature Ablation Study (k=10, Hamming graph)\n(Simulated data)', fontsize=11)
    ax.set_ylim(0.35, 0.75)
    ax.legend(fontsize=9, loc='upper right')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('/mnt/results/04_results_v2/fig3_feature_ablation_v2.png', dpi=150, bbox_inches='tight')
    plt.savefig('/mnt/results/04_results_v2/fig3_feature_ablation_v2.svg', bbox_inches='tight')
    plt.close()
    return