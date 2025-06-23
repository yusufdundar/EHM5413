# Soru 2(b): Tam veri üzerinde basit RBF (KMeans + Ridge)
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier

if __name__ == '__main__':

    # --- 1. Veri kümesini oku -------------------
    cols = ["class", "T3_resin", "thyroxine", "triiodothyronine", "thyroid_stimulating", "basal_TSH"]
    df = pd.read_csv("new-thyroid.data", header=None, names=cols, sep=",")

    X = StandardScaler().fit_transform(df.drop("class", axis=1).values)
    y = df["class"].astype(int).values

    # ---------- RBF parametreleri (elle seçilen) ------------------------------
    k = 15  # merkez sayısı
    sigma = 0.7  # yayılım
    lam = 1e-2  # Ridge cezası

    km = KMeans(n_clusters=k, random_state=42).fit(X)
    C = km.cluster_centers_
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
    Phi = np.exp(-d2 / (2 * sigma ** 2))

    clf = RidgeClassifier(alpha=lam).fit(Phi, y)

    train_acc = clf.score(Phi, y)  # “tam veri” doğruluğu
    print("RBF   (k=15, σ=0.7):", train_acc)