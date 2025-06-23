# ------------------------------------------------------------
# Soru 4: 2-fold CV ile iki RBF öğrenme algoritması
# ------------------------------------------------------------
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

# ---------- 0. VERİ --------------------------------------------------------
cols = ["class", "T3_resin", "thyroxine", "triiodothyronine", "thyroid_stimulating", "basal_TSH"]
df = pd.read_csv("new-thyroid.data", header=None, names=cols, sep=",")

X = StandardScaler().fit_transform(df.drop("class",axis=1).values)
y = df["class"].astype(int).values
classes = np.unique(y)

# ---------- 1. ORTAK RBF DÖNÜŞTÜRÜCÜ -------------------------------------
class KM_RBF(BaseEstimator, TransformerMixin):
    """K-Means tabanlı RBF; isteğe bağlı sınıf-bazlı merkez."""
    def __init__(self, n_centers=10, spread=1.0, supervised=False, random_state=42):
        self.n_centers = n_centers
        self.spread    = spread
        self.supervised= supervised
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.supervised and y is not None:
            # sınıf başına eşit merkez
            centers = []
            per_cls = self.n_centers // len(np.unique(y))
            for cls in np.unique(y):
                km = KMeans(per_cls, random_state=self.random_state).fit(X[y==cls])
                centers.append(km.cluster_centers_)
            self.centers_ = np.vstack(centers)
        else:
            self.centers_ = KMeans(self.n_centers,
                                   random_state=self.random_state).fit(X).cluster_centers_
        return self

    def transform(self, X):
        diff   = X[:,None,:] - self.centers_[None,:,:]
        d2     = np.sum(diff**2, axis=2)
        return np.exp(-d2/(2*self.spread**2))

# ---------- 2. İKİ RBF ALG. TANIMI ----------------------------------------
def build_rbf_model(k, sigma, lam, supervised=False):
    return {
        "rbf": KM_RBF(n_centers=k, spread=sigma, supervised=supervised),
        "clf": RidgeClassifier(alpha=lam)
    }

algos = {
    "Unsupervised-KMeans":  build_rbf_model(k=15, sigma=0.7, lam=1e-2, supervised=False),
    "Supervised-KMeans":    build_rbf_model(k=18, sigma=0.7, lam=1e-2, supervised=True)
}

# ---------- 3. 2-KATLI CV -------------------------------------------------
kf = KFold(n_splits=2, shuffle=True, random_state=1)
avg_rows, fold_rows = [], []

for name, comp in algos.items():
    for fold,(tri,tes) in enumerate(kf.split(X),1):
        Φ_train = comp["rbf"].fit_transform(X[tri], y[tri])
        Φ_test  = comp["rbf"].transform(X[tes])
        clf     = comp["clf"].fit(Φ_train, y[tri])

        for split, Φ, idx in [("train",Φ_train, tri), ("test",Φ_test, tes)]:
            y_pred = clf.predict(Φ)
            for c in classes:
                acc = accuracy_score(y[idx]==c, y_pred==c)
                fold_rows.append((fold,name,split,c,acc))
                avg_rows.append((name,split,c,acc))

# ortalamaları grupla
import pandas as pd
avg_df  = (pd.DataFrame(avg_rows, columns=["Algo","Set","Class","Acc"])
              .groupby(["Algo","Set","Class"]).mean()
              .reset_index()
              .pivot_table(index=["Algo","Set"], columns="Class", values="Acc"))
fold_df = pd.DataFrame(fold_rows, columns=["Fold","Algo","Set","Class","Acc"])


# SADECE Acc sütununu yüzdeye çevir
fold_print = fold_df.copy()
fold_print["Acc"] = (fold_print["Acc"] * 100).round(1)


print(fold_print.to_string(index=False))

avg_print = avg_df.copy() * 100
avg_print = avg_print.round(1)
print(avg_print)


print("\n--- Ortalama ---\n", (avg_df*100).round(1))
print("\n--- Fold ---\n", (fold_df.head()*100).round(1))

# ---------- 4. SPREAD (σ) TARAMASI ----------------------------------------
sigmas = np.arange(0.1, 2.1, 0.1)
accs   = []
for s in sigmas:
    rbf  = KM_RBF(n_centers=15, spread=s).fit(X)
    Φ    = rbf.transform(X)
    accs.append(RidgeClassifier(alpha=1e-2).fit(Φ, y).score(Φ, y))

plt.figure()
plt.plot(sigmas, accs, marker='o')
plt.xlabel("Spread σ")
plt.ylabel("Doğruluk (tam veri)")
plt.title("RBF – σ parametre taraması")
plt.grid(True, alpha=.3)
plt.savefig("spread_scan.png", dpi=300)
