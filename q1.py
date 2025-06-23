import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

    # --- 1. OKUMA ---
    cols = ["class", "T3_resin", "thyroxine", "triiodothyronine", "thyroid_stimulating", "basal_TSH"]
    df = pd.read_csv("new-thyroid.data", header=None, names=cols, sep=",")
    df["class"] = df["class"].astype(int)

    print("Veri şekli :", df.shape)
    print("Sınıf dağılımı:\n", df["class"].value_counts(), "\n")

    # --- 2. ÖLÇEKLEME ---
    X_raw = df.drop("class", axis=1).values
    y = df["class"].values
    X = StandardScaler().fit_transform(X_raw)

    # --- 3. 3-B SAÇILIM ---
    triplets = [(0, 1, 2), (0, 1, 4), (1, 2, 4)]
    labels = [
        "T3_resin – thyroxine – triiodothyronine",
        "T3_resin – thyroxine – basal_TSH",
        "thyroxine – triiodothyronine – basal_TSH"
    ]
    markers = {1: "o", 2: "^", 3: "s"}  # hiper, hipo, normal

    for idx, (a, b, c) in enumerate(triplets, 1):
        fig = plt.figure(figsize=(6, 4.6))
        ax = fig.add_subplot(111, projection="3d")

        for cls in np.unique(y):
            sel = y == cls
            ax.scatter(
                X[sel, a], X[sel, b], X[sel, c],
                marker=markers[cls], alpha=0.75, label=f"Sınıf {cls}"
            )

        ax.set_xlabel(cols[a + 1])
        ax.set_ylabel(cols[b + 1])
        ax.set_zlabel(cols[c + 1])
        ax.set_title(f"3-B Saçılım {idx}: {labels[idx - 1]}")
        ax.legend(loc="best")
        fig.tight_layout()
        plt.savefig(f"scatter{idx}.png", dpi=300)
        plt.show()
