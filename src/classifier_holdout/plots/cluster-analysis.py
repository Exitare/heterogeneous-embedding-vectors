#!/usr/bin/env python3
# cluster_analysis_full.py

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import h5py
from argparse import ArgumentParser
from itertools import combinations
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, silhouette_samples, pairwise_distances,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_completeness_v_measure, f1_score, classification_report
)
from sklearn.preprocessing import StandardScaler, normalize
from scipy.optimize import linear_sum_assignment
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------- utils -----------------------------
def fit_gmm_auto_k(X, k_min=2, k_max=20, cov_types=("full","diag","tied","spherical"), random_state=42, n_init=5):
    best = None
    best_bic = np.inf
    for cov in cov_types:
        for k in range(k_min, k_max+1):
            g = GaussianMixture(n_components=k, covariance_type=cov, random_state=random_state, n_init=n_init)
            g.fit(X)
            bic = g.bic(X)
            if bic < best_bic:
                best, best_bic = g, bic
    return best

def contingency(y_true, y_pred):
    return pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="cluster"))

def hungarian_map(y_true, y_pred, classes=None, clusters=None):
    ct = contingency(y_true, y_pred)
    # ensure deterministic order
    ct = ct.sort_index(axis=0).sort_index(axis=1)
    cost = -ct.values
    r, c = linear_sum_assignment(cost)
    mapping = {ct.columns[cj]: ct.index[ri] for ri, cj in zip(r, c)}
    mapped = pd.Series(y_pred).map(mapping).values
    acc = ct.values[r, c].sum() / ct.values.sum()
    macro_f1 = f1_score(y_true, mapped, average="macro")
    report = classification_report(y_true, mapped, digits=4)
    return mapping, mapped, acc, macro_f1, report, ct

def per_cancer_distances(X, y, metric="euclidean"):
    recs = []
    cancers = np.unique(y)
    for c in cancers:
        mask = (y == c)
        Xc = X[mask]
        # intra (upper triangle)
        if Xc.shape[0] > 1:
            d_intra = pairwise_distances(Xc, Xc, metric=metric)
            tri = np.triu_indices_from(d_intra, k=1)
            intra_mean = d_intra[tri].mean() if tri[0].size else np.nan
        else:
            intra_mean = np.nan
        # nearest inter
        inter_means = []
        for c2 in cancers:
            if c2 == c: continue
            Xo = X[y == c2]
            if Xc.size and Xo.size:
                d_inter = pairwise_distances(Xc, Xo, metric=metric)
                inter_means.append(d_inter.mean())
        nearest_inter = np.min(inter_means) if inter_means else np.nan
        if np.isfinite(intra_mean) and np.isfinite(nearest_inter) and max(intra_mean, nearest_inter) > 0:
            s_class = (nearest_inter - intra_mean) / max(intra_mean, nearest_inter)
        else:
            s_class = np.nan
        recs.append({
            "cancer": c,
            "intra_mean_dist": intra_mean,
            "nearest_inter_mean_dist": nearest_inter,
            "silhouette_class_from_means": s_class,
            "n_samples": int(mask.sum())
        })
    return pd.DataFrame(recs)

def lift_matrix(ct: pd.DataFrame):
    row_tot = ct.sum(axis=1)
    col_tot = ct.sum(axis=0)
    N = ct.values.sum()
    p_cluster = col_tot / N
    exp_same = float((p_cluster**2).sum())
    cancers = ct.index.tolist()
    L = pd.DataFrame(np.nan, index=cancers, columns=cancers, dtype=float)
    for a, b in combinations(cancers, 2):
        num_same = float((ct.loc[a] * ct.loc[b]).sum())
        obs_ij = num_same / (row_tot[a] * row_tot[b])
        val = obs_ij / exp_same if exp_same > 0 else np.nan
        L.loc[a, b] = L.loc[b, a] = val
    return L

def lift_perm_pvalues(y_true, y_pred, n_perm=200, rng=0):
    """One-sided p-values for LIFT >= observed by shuffling cluster labels (preserves cluster size)."""
    rng = np.random.default_rng(rng)
    ct_obs = contingency(y_true, y_pred)
    L_obs = lift_matrix(ct_obs)
    cancers = list(ct_obs.index)
    pairs = [(a, b) for a, b in combinations(cancers, 2)]
    counts = {pair: 0 for pair in pairs}
    y_pred = np.asarray(y_pred)
    for _ in range(n_perm):
        y_perm = rng.permutation(y_pred)  # preserve counts
        L_perm = lift_matrix(contingency(y_true, y_perm))
        for pair in pairs:
            a, b = pair
            if L_perm.loc[a, b] >= L_obs.loc[a, b]:
                counts[pair] += 1
    pvals = {pair: (counts[pair] + 1) / (n_perm + 1) for pair in pairs}
    # pack results into symmetric DataFrame
    P = pd.DataFrame(1.0, index=cancers, columns=cancers, dtype=float)
    for (a, b), p in pvals.items():
        P.loc[a, b] = P.loc[b, a] = p
    np.fill_diagonal(P.values, np.nan)
    return L_obs, P

def pair_components(ct: pd.DataFrame, a: str, b: str):
    ni = ct.loc[a].sum()
    nj = ct.loc[b].sum()
    comp = (ct.loc[a]/ni) * (ct.loc[b]/nj)
    return comp.sort_values(ascending=False).to_frame("contribution")

def fisher_cellwise(ct: pd.DataFrame):
    """Cell vs rest Fisher exact with log2OR (Haldane for OR only), Pearson residuals, FDR."""
    N = int(ct.values.sum())
    row_tot = ct.sum(axis=1).astype(int)
    col_tot = ct.sum(axis=0).astype(int)
    exp = np.outer(row_tot, col_tot) / N
    pearson = (ct - exp) / np.sqrt(exp)
    pearson = pearson.replace([np.inf, -np.inf], np.nan)

    recs = []
    for r in ct.index:
        for c in ct.columns:
            a = int(ct.loc[r, c])
            b = int(row_tot[r] - a)
            c_val = int(col_tot[c] - a)
            d = int(N - (a + b + c_val))
            table_int = np.array([[a, b], [c_val, d]], dtype=int)
            _, p = fisher_exact(table_int, alternative="two-sided")
            table_float = table_int.astype(float)
            if (table_float == 0).any():
                table_float += 0.5
            OR = (table_float[0,0]*table_float[1,1])/(table_float[0,1]*table_float[1,0])
            recs.append({"cancer": r, "cluster": c, "log2OR": np.log2(OR), "p": p})
    enrich = pd.DataFrame(recs)
    enrich["q"] = multipletests(enrich["p"], method="fdr_bh")[1]
    log2OR = enrich.pivot(index="cancer", columns="cluster", values="log2OR")
    qmat = enrich.pivot(index="cancer", columns="cluster", values="q")
    return pearson, log2OR, qmat, enrich

# ----------------------------- main -----------------------------
if __name__ == "__main__":
    parser = ArgumentParser(description="Comprehensive cluster analysis")
    parser.add_argument("--selected_cancers", "-c", nargs="+", default=["BRCA","LUAD","STAD","BLCA","COAD","THCA"])
    parser.add_argument("--walk_amount", "-a", type=int, default=3)
    parser.add_argument("--walk_distance", "-w", type=int, default=3)
    parser.add_argument("--modalities", "-m", nargs="+",
                        default=["annotations","images","mutations","rna"],
                        choices=["rna","annotations","mutations","images"])
    parser.add_argument("--auto_k", action="store_true", help="Use BIC to choose GMM k and covariance type")
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--perm", type=int, default=0, help="Number of permutations for LIFT p-values (0=skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", action="store_true", help="Standardize + L2-normalize before metrics")
    args = parser.parse_args()

    cancers = args.selected_cancers
    selected_cancers = "_".join(cancers)
    walk_amount = args.walk_amount
    walk_distance = args.walk_distance
    modalities = args.modalities
    selected_modalities = "_".join(modalities)

    if len(modalities) < 2:
        raise ValueError("At least two modalities must be selected for summing embeddings.")

    results_dir = Path("figures", "classifier_holdout", selected_cancers, selected_modalities, "clustering")
    results_dir.mkdir(parents=True, exist_ok=True)

    h5_file = Path("results", "classifier_holdout", "summed_embeddings",
                   selected_cancers, selected_modalities, f"{walk_amount}_{walk_distance}", "0", "summed_embeddings.h5")
    logging.info(f"Loading embeddings from {h5_file}")
    with h5py.File(h5_file, "r") as f:
        X = f["X"][:]
        y_bytes = f["y"][:]
    y = np.array([s.decode("utf-8") for s in y_bytes])
    uniq = np.unique(y)
    logging.info(f"X: {X.shape}, cancers: {uniq.tolist()}")

    # Optional scaling + L2
    Xe = X.copy()
    if args.scale:
        Xe = StandardScaler().fit_transform(Xe)
        Xe = normalize(Xe)  # L2 row-normalize

    # --- GMM ---
    if args.auto_k:
        gmm = fit_gmm_auto_k(Xe, k_min=max(2, len(uniq)-2), k_max=args.k_max, random_state=args.seed)
        logging.info(f"Selected GMM: k={gmm.n_components}, cov={gmm.covariance_type}")
    else:
        gmm = GaussianMixture(n_components=len(uniq), random_state=args.seed)
        gmm.fit(Xe)
    labels = gmm.predict(Xe)

    # --- metrics: silhouettes ---
    # use Euclidean on Xe; also cosine (often better for embeddings)
    try:
        sil_true_eu = silhouette_score(Xe, pd.Categorical(y).codes, metric="euclidean")
        sil_gmm_eu  = silhouette_score(Xe, labels, metric="euclidean")
        sil_true_co = silhouette_score(Xe, pd.Categorical(y).codes, metric="cosine")
        sil_gmm_co  = silhouette_score(Xe, labels, metric="cosine")
    except Exception as e:
        logging.warning(f"Silhouette failed: {e}")
        sil_true_eu = sil_gmm_eu = sil_true_co = sil_gmm_co = np.nan

    # --- alignment metrics ---
    y_codes = pd.Categorical(y).codes
    ari = adjusted_rand_score(y_codes, labels)
    nmi = normalized_mutual_info_score(y_codes, labels)
    homog, compl, vmeas = homogeneity_completeness_v_measure(y_codes, labels)

    # --- contingency, purity/coverage ---
    ct = contingency(y, labels)
    cluster_purity = ct.max(axis=0).sum() / ct.values.sum()
    class_coverage = ct.max(axis=1).sum() / ct.values.sum()

    # log cluster majorities
    totals = ct.sum(axis=0)
    majority = ct.idxmax(axis=0)
    top3 = ct.apply(lambda col: (col/col.sum()).sort_values(ascending=False).head(3).to_dict(), axis=0)
    for cl in ct.columns:
        frac = ct.loc[majority[cl], cl] / totals[cl]
        logging.info(f"Cluster {cl}: majority={majority[cl]} ({frac:.2%}) | top3={top3[cl]}")

    # --- Hungarian mapping ---
    mapping, mapped, hung_acc, hung_macro_f1, cls_report, ct_sorted = hungarian_map(y, labels)

    # --- per-cancer silhouettes + distances ---
    sil_samples_true = silhouette_samples(Xe, y_codes, metric="euclidean")
    per_cancer_sil = (pd.DataFrame({"cancer": y, "silhouette": sil_samples_true})
                      .groupby("cancer")["silhouette"].agg(["mean","std","count"]).reset_index()
                      .rename(columns={"mean":"silhouette_mean","std":"silhouette_std","count":"n"}))
    per_cancer_dist = per_cancer_distances(Xe, y, metric="euclidean")
    per_cancer_summary = per_cancer_sil.merge(per_cancer_dist, on="cancer", how="outer")

    # --- LIFT + optional permutations ---
    L = lift_matrix(ct)
    if args.perm and args.perm > 0:
        L_obs, P = lift_perm_pvalues(y, labels, n_perm=args.perm, rng=args.seed)
    else:
        L_obs, P = L, None

    # per-pair cluster contribution table (all pairs)
    pairs = list(combinations(ct.index.tolist(), 2))
    comps = []
    for a, b in pairs:
        contrib_df = pair_components(ct, a, b)  # DataFrame with 'contribution'
        contrib_df["pair"] = f"{a}-{b}"
        contrib_df = contrib_df.reset_index().rename(columns={"index": "cluster"})
        comps.append(contrib_df)
    pair_comp_df = pd.concat(comps, ignore_index=True)

    # --- cellwise enrichment ---
    pearson_resid, log2OR_mat, q_mat, enrich_long = fisher_cellwise(ct)

    # --- chi-square association (global) ---
    chi2, chi2_p, _, _ = chi2_contingency(ct.values)

    # --- per-sample mapping table ---
    sample_map = pd.DataFrame({
        "sample_idx": np.arange(len(y)),
        "true": y,
        "cluster": labels
    })

    # ----------------------------- save -----------------------------
    prefix = f"{walk_distance}_{walk_amount}"
    out = results_dir

    # text summary
    with open(out / f"{prefix}_summary.txt", "w") as f:
        f.write(f"GMM: k={gmm.n_components}, cov={getattr(gmm,'covariance_type','full')}\n")
        f.write(f"Silhouette EUC (true/gmm): {sil_true_eu:.6f} / {sil_gmm_eu:.6f}\n")
        f.write(f"Silhouette COS (true/gmm): {sil_true_co:.6f} / {sil_gmm_co:.6f}\n")
        f.write(f"ARI: {ari:.6f} | NMI: {nmi:.6f} | Hom: {homog:.6f} | Comp: {compl:.6f} | V: {vmeas:.6f}\n")
        f.write(f"Purity: {cluster_purity:.6f} | ClassCoverage: {class_coverage:.6f}\n")
        f.write(f"Hungarian accuracy: {hung_acc:.6f} | Macro-F1: {hung_macro_f1:.6f}\n")
        f.write(f"Chi2 assoc p-value: {chi2_p:.6e}\n")
        f.write(f"Cluster→Class mapping: {mapping}\n")

    with open(out / f"{prefix}_hungarian_classification_report.txt", "w") as f:
        f.write(cls_report)

    ct.to_csv(out / f"{prefix}_contingency.csv")
    L.round(3).to_csv(out / f"{prefix}_lift.csv")
    if P is not None:
        P.to_csv(out / f"{prefix}_lift_perm_p.csv")

    pair_comp_df.to_csv(out / f"{prefix}_pairwise_components.csv", index=False)
    per_cancer_summary.to_csv(out / f"{prefix}_per_cancer_summary.csv", index=False)
    pearson_resid.to_csv(out / f"{prefix}_pearson_residuals.csv")
    log2OR_mat.to_csv(out / f"{prefix}_log2OR.csv")
    q_mat.to_csv(out / f"{prefix}_fdr.csv")
    enrich_long.to_csv(out / f"{prefix}_enrichment_long.csv", index=False)
    sample_map.to_csv(out / f"{prefix}_sample_map.csv", index=False)

    logging.info("== DONE ==")
    logging.info(f"Saved results to: {out.resolve()}")