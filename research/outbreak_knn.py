"""
Disease Outbreak Risk Classification using K-Nearest Neighbours (KNN)
AI & ML Lab (CCE3131) - Mini-Project #6

Task : classify a patient's symptom vector into LOW / MEDIUM / HIGH outbreak-risk.
Data : synthetic symptom dataset generated below (default), OR your own CSV:
       python outbreak_knn.py --csv data.csv --target risk
       (all other columns must be numeric / 0-1 symptom flags)

Run  : python outbreak_knn.py
Out  : figures + results.json inside ./outputs/
Tip  : the "# %%" markers let VS Code / Jupytext treat this as a notebook.
"""

# %% Imports
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = "outputs"

LABELS = ["Low", "Medium", "High"]
SYMPTOMS = ["fever", "cough", "fatigue", "headache", "body_ache", "sore_throat",
            "breathlessness", "loss_of_smell", "diarrhea", "vomiting", "rash",
            "joint_pain", "chills", "nausea"]


# %% 1. Data: synthetic generator (stand-in for a UCI / Kaggle symptom dataset)
def make_synthetic(n=3000, seed=SEED):
    """Each risk class has its own symptom-probability profile. Profiles overlap
    on purpose (e.g. fever is common in all classes) so the problem is not trivial."""
    rng = np.random.default_rng(seed)
    priors = [0.50, 0.30, 0.20]                       # low, medium, high
    #            fev  cou  fat  hea  bod  sor  bre  sme  dia  vom  ras  joi  chi  nau
    p = np.array([
        [0.15, 0.25, 0.25, 0.20, 0.15, 0.20, 0.03, 0.03, 0.10, 0.08, 0.04, 0.08, 0.08, 0.10],  # low
        [0.45, 0.45, 0.45, 0.35, 0.35, 0.30, 0.15, 0.15, 0.20, 0.18, 0.10, 0.20, 0.30, 0.22],  # medium
        [0.70, 0.60, 0.65, 0.45, 0.55, 0.35, 0.40, 0.35, 0.30, 0.30, 0.20, 0.35, 0.55, 0.35],  # high
    ])
    y = rng.choice(3, size=n, p=priors)
    X = (rng.random((n, len(SYMPTOMS))) < p[y]).astype(int)
    df = pd.DataFrame(X, columns=SYMPTOMS)
    # numeric / contextual features
    df["age"] = np.clip(rng.normal([35, 42, 50][0], 15, n) + y * 6, 1, 90).round()
    df["temperature_c"] = np.round(rng.normal(36.8 + 0.6 * y, 0.5, n), 1)
    df["days_since_onset"] = np.clip(rng.poisson(2 + y, n), 0, 14)
    df["contact_with_case"] = (rng.random(n) < np.array([0.05, 0.25, 0.55])[y]).astype(int)
    df["risk"] = y
    return df


def load_data(csv=None, target="risk"):
    if csv:
        df = pd.read_csv(csv)
        if not pd.api.types.is_numeric_dtype(df[target]):                # map text labels -> 0/1/2
            order = {"low": 0, "medium": 1, "high": 2}
            df[target] = df[target].str.lower().map(order)
        df = df.rename(columns={target: "risk"}).apply(pd.to_numeric, errors="raise")
        if len(df) < 100 or not np.isfinite(df.to_numpy()).all():
            raise ValueError("CSV needs at least 100 rows and finite numeric values")
        if set(df.risk.unique()) != {0, 1, 2} or df.groupby("risk").size().min() < 10:
            raise ValueError("CSV needs all three risk classes and at least 10 rows per class")
        return df
    return make_synthetic()


# %% 2. Exploratory plots
def eda(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    df["risk"].map(dict(enumerate(LABELS))).value_counts().reindex(LABELS).plot(
        kind="bar", ax=ax[0], color=["#4c9f70", "#e0a030", "#c0392b"])
    ax[0].set_title("Class distribution"); ax[0].set_ylabel("patients"); ax[0].tick_params(axis="x", rotation=0)
    sym = [c for c in SYMPTOMS if c in df.columns]
    if not sym:
        sym = [c for c in df.columns if c != "risk"]
    prev = df.groupby("risk")[sym].mean().T
    prev.columns = LABELS
    sns.heatmap(prev, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax[1], cbar=False)
    ax[1].set_title("Symptom prevalence per risk class")
    plt.tight_layout(); plt.savefig(f"{OUT}/01_eda.png", dpi=150); plt.close()


# %% 3. Main experiment
def main(csv=None, target="risk"):
    os.makedirs(OUT, exist_ok=True)
    df = load_data(csv, target)
    df.to_csv(f"{OUT}/dataset_used.csv", index=False)
    eda(df)

    X, y = df.drop(columns="risk"), df["risk"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    results = {"n_samples": len(df), "n_features": X.shape[1],
               "train_size": len(X_tr), "test_size": len(X_te)}

    # --- 3a. Baseline: KNN WITHOUT scaling vs WITH scaling (k=5)
    raw = KNeighborsClassifier(5).fit(X_tr, y_tr)
    scaled = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(5))]).fit(X_tr, y_tr)
    results["baseline_no_scaling_acc"] = accuracy_score(y_te, raw.predict(X_te))
    results["baseline_scaled_acc"] = accuracy_score(y_te, scaled.predict(X_te))

    # --- 3b. Accuracy vs k (5-fold CV) -> choose k
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    ks = list(range(1, 41))
    from sklearn.model_selection import cross_val_score
    cv_means, cv_stds = [], []
    for k in ks:
        pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(k))])
        s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro")
        cv_means.append(s.mean()); cv_stds.append(s.std())
    cv_means, cv_stds = np.array(cv_means), np.array(cv_stds)
    plt.figure(figsize=(7, 4))
    plt.plot(ks, cv_means, marker="o", ms=3, color="#1a2b4c")
    plt.fill_between(ks, cv_means - cv_stds, cv_means + cv_stds, alpha=0.2, color="#1a2b4c")
    plt.axvline(ks[int(cv_means.argmax())], ls="--", color="#c0392b", label=f"best k = {ks[int(cv_means.argmax())]}")
    plt.xlabel("k (number of neighbours)"); plt.ylabel("5-fold CV macro-F1")
    plt.title("Choosing k: under-fit (large k) vs over-fit (small k)"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/02_k_vs_f1.png", dpi=150); plt.close()

    # --- 3c. Grid search: k, distance metric, weighting
    pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier())])
    grid = {"knn__n_neighbors": list(range(3, 32, 2)),
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan", "cosine"]}
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=1).fit(X_tr, y_tr)
    best = gs.best_estimator_
    import joblib
    joblib.dump(best, f"{OUT}/model.joblib")
    with open(f"{OUT}/cv_curve.json", "w") as curve:
        json.dump({"k":ks,"mean":cv_means.tolist(),"std":cv_stds.tolist()},curve)
    results["best_params"] = {k.replace("knn__", ""): v for k, v in gs.best_params_.items()}
    results["cv_best_macro_f1"] = gs.best_score_

    # --- 3d. Final evaluation on held-out test set
    pred = best.predict(X_te)
    results["test_accuracy"] = accuracy_score(y_te, pred)
    results["test_macro_f1"] = f1_score(y_te, pred, average="macro")
    results["test_recall_per_class"] = dict(zip(LABELS, recall_score(y_te, pred, average=None).round(3).tolist()))
    results["classification_report"] = classification_report(y_te, pred, target_names=LABELS, output_dict=True)
    print(classification_report(y_te, pred, target_names=LABELS))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig(f"{OUT}/03_confusion_matrix.png", dpi=150); plt.close()

    # --- 3e. Decision-surface view: PCA to 2D, refit KNN there (visual only)
    sc = StandardScaler().fit(X_tr)
    pca = PCA(2, random_state=SEED).fit(sc.transform(X_tr))
    Z_tr, Z_te = pca.transform(sc.transform(X_tr)), pca.transform(sc.transform(X_te))
    vis = KNeighborsClassifier(results["best_params"]["n_neighbors"]).fit(Z_tr, y_tr)
    xx, yy = np.meshgrid(np.linspace(Z_tr[:, 0].min() - 1, Z_tr[:, 0].max() + 1, 300),
                         np.linspace(Z_tr[:, 1].min() - 1, Z_tr[:, 1].max() + 1, 300))
    zz = vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(["#bfe3cc", "#f6dfa8", "#f0b6ae"])
    plt.figure(figsize=(6.5, 5))
    plt.contourf(xx, yy, zz, cmap=cm, alpha=0.8)
    for c, col, lab in zip(range(3), ["#2e7d4f", "#c68a12", "#a93226"], LABELS):
        m = y_te.values == c
        plt.scatter(Z_te[m, 0], Z_te[m, 1], s=10, c=col, label=lab, alpha=0.7)
    plt.legend(title="True risk"); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"KNN decision regions in 2-D PCA space (explained var = {pca.explained_variance_ratio_.sum():.0%})")
    plt.tight_layout(); plt.savefig(f"{OUT}/04_decision_regions.png", dpi=150); plt.close()
    results["pca_2d_explained_variance"] = float(pca.explained_variance_ratio_.sum())

    # --- 3f. Predict for a brand-new patient
    demo_features = set(SYMPTOMS + ["age", "temperature_c", "days_since_onset", "contact_with_case"])
    if set(X.columns).issubset(demo_features):
        new = pd.DataFrame([{**{s: 0 for s in SYMPTOMS}, "fever": 1, "cough": 1, "breathlessness": 1,
                             "loss_of_smell": 1, "fatigue": 1, "age": 58, "temperature_c": 38.9,
                             "days_since_onset": 4, "contact_with_case": 1}])[X.columns]
        proba = best.predict_proba(new)[0]
        results["demo_patient_prediction"] = LABELS[int(proba.argmax())]
        results["demo_patient_probabilities"] = dict(zip(LABELS, proba.round(3).tolist()))
    
    else:
        results["demo_patient_notice"] = "Fixed symptom example skipped for custom feature schema"

    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (np.floating, float)): return round(float(o), 4)
        if isinstance(o, (np.integer,)): return int(o)
        return o
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(clean(results), f, indent=2)
    print(json.dumps(clean({k: v for k, v in results.items() if k != "classification_report"}), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--target", default="risk")
    a = ap.parse_args()
    main(a.csv, a.target)
