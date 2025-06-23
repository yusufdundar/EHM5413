# ------------------------------------------------------------
# Soru 2(a): Tam veri üzerinde MLP hiper-parametre taraması
# ------------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':

    # --- 1. Veri kümesini oku -------------------
    cols = ["class", "T3_resin", "thyroxine", "triiodothyronine", "thyroid_stimulating", "basal_TSH"]
    df = pd.read_csv("new-thyroid.data", header=None, names=cols, sep=",")

    X, y = df.drop("class", axis=1).values, df["class"].astype(int).values

    # --- 2. Boru hattı ve hiper-parametre ızgarası ---------------
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=2000, random_state=42))
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(5,), (8,), (8, 4), (10, 5)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-4, 1e-3],
    }

    gs = GridSearchCV(
        pipe, param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    ).fit(X, y)

    print("En iyi skor (CV):", gs.best_score_)
    print("En iyi parametreler:", gs.best_params_)

    # --- 3. Tam veri ile yeniden eğit, "eğitim/test" skorları ---
    best_model = gs.best_estimator_
    train_acc = best_model.score(X, y)
    print("Tam veri skoru (\"test\" dâhil):", train_acc)

