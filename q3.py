# ------------------------------------------------------------
# Soru 3 – 2-fold CV ile dört MLP öğrenme algoritması
# ------------------------------------------------------------
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':

    # --- 1. Veri kümesini oku -------------------
    cols = ["class", "T3_resin", "thyroxine", "triiodothyronine", "thyroid_stimulating", "basal_TSH"]

    df = pd.read_csv("new-thyroid.data", header=None, names=cols, sep=",")

    X = df.drop("class", axis=1).values
    y = df["class"].astype(int).values
    classes = np.unique(y)  # [1,2,3]

    # ---------- 1. Dört MLP varyasyonu -----------------------------------------
    best_arch = (5,)  # Soru 2’deki mimari
    base_args = dict(hidden_layer_sizes=best_arch,
                     activation='relu', alpha=1e-4,
                     max_iter=2000, random_state=42)

    algos = {
        "SGD": MLPClassifier(solver='sgd', momentum=0, learning_rate_init=0.01, **base_args),
        "SGD+Momentum": MLPClassifier(solver='sgd', momentum=0.9, learning_rate_init=0.01, **base_args),
        "Adam": MLPClassifier(solver='adam', **base_args),
        "LBFGS": MLPClassifier(solver='lbfgs', **base_args)
    }

    # ---------- 2. 2-katlı çapraz geçerleme ------------------------------------
    kf = KFold(n_splits=2, shuffle=True, random_state=1)

    results_avg = {name: {'train': {c: [] for c in classes},
                          'test': {c: [] for c in classes}}
                   for name in algos}
    results_folds = []  # her satır: (fold, algo, set, class, acc)

    for fold, (tri, tes) in enumerate(kf.split(X), start=1):
        for name, model in algos.items():
            pipe = Pipeline([("sc", StandardScaler()),
                             ("mlp", model)])
            pipe.fit(X[tri], y[tri])

            for split, idx in [("train", tri), ("test", tes)]:
                y_pred = pipe.predict(X[idx])
                for cls in classes:
                    acc = accuracy_score(y[idx] == cls, y_pred == cls)
                    results_folds.append((fold, name, split, cls, acc))
                    results_avg[name][split][cls].append(acc)

    # ---------- 3. Ortalamaları derle ------------------------------------------
    avg_rows = []
    for name in algos:
        for split in ("train", "test"):
            per_class = [np.mean(results_avg[name][split][c]) for c in classes]
            avg_rows.append((name, split, *per_class))

    avg_df = pd.DataFrame(avg_rows,
                          columns=["Algo", "Set", "Class1", "Class2", "Class3"])
    fold_df = pd.DataFrame(results_folds,
                           columns=["Fold", "Algo", "Set", "Class", "Acc"])

    print("\n=== Ortalama sonuçlar ===")
    print(avg_df)
    print("\n=== Fold bazlı ===")
    print(fold_df.head())

    # ---------- 4. Öğrenme oranı taraması (standart SGD) -----------------------
    etas = np.arange(0.1, 1.01, 0.1)
    eta_acc = []
    for eta in etas:
        clf = MLPClassifier(solver='sgd', momentum=0,
                            learning_rate_init=eta, **base_args)
        pipe = Pipeline([("sc", StandardScaler()), ("mlp", clf)])
        pipe.fit(X, y)
        eta_acc.append(pipe.score(X, y))

    pivot = fold_df.pivot_table(
        index=["Fold", "Algo", "Set"],
        columns="Class",
        values="Acc") * 100  # yüzdeye çevir
    print(pivot.round(1))

    plt.figure()
    plt.plot(etas, eta_acc, marker='o')
    plt.xlabel("Öğrenme oranı (η)")
    plt.ylabel("Doğruluk (tam veri)")
    plt.title("Standart SGD – öğrenme oranı taraması")
    plt.savefig("eta_scan.png", dpi=300)

    # ---------- 5. Momentum taraması (SGD+Momentum) ----------------------------
    mom = np.arange(0.0, 1.01, 0.1)
    mom_acc = []
    for m in mom:
        clf = MLPClassifier(solver='sgd', momentum=m,
                            learning_rate_init=0.01, **base_args)
        pipe = Pipeline([("sc", StandardScaler()), ("mlp", clf)])
        pipe.fit(X, y)
        mom_acc.append(pipe.score(X, y))

    plt.figure()
    plt.plot(mom, mom_acc, marker='s')
    plt.xlabel("Momentum (β)")
    plt.ylabel("Doğruluk (tam veri)")
    plt.title("SGD + Momentum – momentum taraması")
    plt.savefig("momentum_scan.png", dpi=300)


