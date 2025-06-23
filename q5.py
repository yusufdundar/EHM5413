# s5_lofo.py  ---------------------------------------------------
import pandas as pd, numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# ---- veri ----------------------------------------------------
cols = ["class", "T3_resin", "thyroxine", "triiodothyronine", "thyroid_stimulating", "basal_TSH"]
df = pd.read_csv("new-thyroid.data", header=None, names=cols, sep=",")

X_full = df.drop("class",axis=1).values
y      = df["class"].astype(int).values
features = df.columns[1:]

# ---- en iyi MLP yapılandırması -------------------------------
best_mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='relu',
                         alpha=1e-4, solver='sgd',
                         learning_rate_init=0.01, momentum=0.9,
                         max_iter=2000, random_state=42)

kf = KFold(n_splits=2, shuffle=True, random_state=1)

def cv_score(X):
    acc = []
    for tr, ts in kf.split(X):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("mlp", best_mlp)])
        pipe.fit(X[tr], y[tr])
        acc.append(pipe.score(X[ts], y[ts]))
    return np.mean(acc)

# ---- Tam model skoru -----------------------------------------
base = cv_score(X_full)
print("Tüm özelliklerle CV doğruluğu:", round(base*100,2), "%")

# ---- Leave-one-out -------------------------------------------
for i,f in enumerate(features):
    Xi = np.delete(X_full, i, axis=1)
    s  = cv_score(Xi)
    print(f"   - {f:22s}: {round(s*100,2)} %")
