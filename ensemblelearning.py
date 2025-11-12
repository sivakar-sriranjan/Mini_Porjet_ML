# === PARTIE A : Préparation et Analyse des Données ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
import time

# 1) Chargement du fichier Excel
path = "C:\\Users\\sivas\\Documents\\Mini_Porjet_ML\\Mini projet ensemble learning 2025\\beer_quality.xlsx"
df = pd.read_excel(path)

# 2) Informations générales sur le dataset
n_rows, n_cols = df.shape
print("=== INFORMATIONS GÉNÉRALES ===")
print(f"Nombre de lignes : {n_rows}")
print(f"Nombre de colonnes : {n_cols}")
print("Colonnes :", list(df.columns))

# 3) Séparation des features et de la cible
if "quality" not in df.columns:
    raise ValueError("La colonne cible 'quality' est introuvable dans le fichier.")

X = df.drop(columns=["quality"])
y = df["quality"]

# 4) Découpage en apprentissage/test (70/30)
# On stratifie approximativement en découpant y en intervalles
y_bins = pd.cut(y, bins=np.linspace(y.min(), y.max(), 6), include_lowest=True, right=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y_bins
)

print("\n=== SPLIT DES DONNÉES ===")
print(f"Taille X_train : {X_train.shape}")
print(f"Taille X_test  : {X_test.shape}")

# 5) Statistiques descriptives du jeu d’entraînement
print("\n=== STATISTIQUES DESCRIPTIVES (X_train) ===")
print(X_train.describe().T)

# 6) Répartition de la variable cible
print("\n=== DISTRIBUTION DE y_train ===")
print(y_train.value_counts().sort_index())

print("\n=== DISTRIBUTION DE y_test ===")
print(y_test.value_counts().sort_index())

# 7) Corrélation de Pearson entre chaque feature et la qualité
corr_with_y = X_train.copy()
corr_with_y["quality"] = y_train.values
corr_series = corr_with_y.corr(numeric_only=True)["quality"].drop(labels=["quality"]).sort_values(ascending=False)

print("\n=== CORRÉLATION AVEC LA VARIABLE CIBLE ===")
print(corr_series)

# 8) Visualisation : Distribution de la cible
plt.figure()
plt.hist(y_train, bins=range(int(y.min()), int(y.max()) + 2), edgecolor="black")
plt.title("Distribution de la variable cible (y_train)")
plt.xlabel("quality")
plt.ylabel("Effectif")
plt.tight_layout()
plt.show()

# 9) Visualisation : Histogrammes des features
for col in X_train.columns:
    if is_numeric_dtype(X_train[col]):
        plt.figure()
        plt.hist(X_train[col].dropna(), bins=30, edgecolor="black")
        plt.title(f"Distribution de {col} (X_train)")
        plt.xlabel(col)
        plt.ylabel("Effectif")
        plt.tight_layout()
        plt.show()

# 10) Matrice de corrélations entre les variables explicatives
corr_matrix = X_train.corr(numeric_only=True)
plt.figure(figsize=(6, 5))
im = plt.imshow(corr_matrix, aspect='auto')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(ticks=range(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=90)
plt.yticks(ticks=range(len(corr_matrix.index)), labels=corr_matrix.index)
plt.title("Matrice de corrélation (features) - X_train")
plt.tight_layout()
plt.show()

# 11) Résumé final
print("\n=== RÉSUMÉ RAPIDE ===")
print(f"Nombre d'exemples : {n_rows}, Nombre de variables : {n_cols-1} (hors cible) + 1 cible")
print("Colonnes :", ", ".join(df.columns))
print("\nTop 5 corrélations positives :")
print(corr_series.head(5))
print("\nTop 5 corrélations négatives :")
print(corr_series.tail(5))

# === PARTIE B : Classification binaire ===
# 1) ybin via médiane (calculée uniquement sur y_train)
m = y_train.median()
print(f"\n=== PARTIE B : Classification binaire (médiane m = {m:.3f}) ===")
ybin_train = (y_train >= m).astype(int)
ybin_test = (y_test >= m).astype(int)
print("Répartition ybin_train :")
print(ybin_train.value_counts().sort_index())
print("\nRépartition ybin_test :")
print(ybin_test.value_counts().sort_index())

# 2) Optimisation rapide d'un arbre de décision par recherche aléatoire
dt = DecisionTreeClassifier(random_state=42)
param_dist = {
    "max_depth": [None] + list(range(1, 11)),
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
}
rs = RandomizedSearchCV(
    estimator=dt,
    param_distributions=param_dist,
    n_iter=40,
    scoring="accuracy",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
t0 = time.perf_counter()
rs.fit(X_train, ybin_train)
train_time_tree = time.perf_counter() - t0
best_tree = rs.best_estimator_
t1 = time.perf_counter()
ybin_pred_test_tree = best_tree.predict(X_test)
infer_time_tree = time.perf_counter() - t1
acc_test_tree = accuracy_score(ybin_test, ybin_pred_test_tree)
print("\n=== Arbre de décision optimisé (Random Search) ===")
print("Meilleurs hyperparamètres :", rs.best_params_)
print(f"Temps apprentissage : {train_time_tree:.3f}s | Temps inférence : {infer_time_tree:.3f}s")
print(f"Accuracy test : {acc_test_tree:.4f}")

# 3) AdaBoost avec arbres faibles : courbes accuracy vs n_estimators (train/val)
X_tr, X_val, ybin_tr, ybin_val = train_test_split(
    X_train, ybin_train, test_size=0.2, random_state=42, stratify=ybin_train
)

def evaluate_adaboost_over_n_estimators(max_depth_value):
    ns = list(range(10, 201, 10))
    train_accs, val_accs = [], []
    for n in ns:
        base = DecisionTreeClassifier(max_depth=max_depth_value, random_state=42)
        model = AdaBoostClassifier(
            estimator=base, n_estimators=n, algorithm="SAMME.R", random_state=42
        )
        model.fit(X_tr, ybin_tr)
        y_tr_pred = model.predict(X_tr)
        y_val_pred = model.predict(X_val)
        train_accs.append(accuracy_score(ybin_tr, y_tr_pred))
        val_accs.append(accuracy_score(ybin_val, y_val_pred))
    return ns, train_accs, val_accs

for depth in [1, 5]:
    ns, tr, va = evaluate_adaboost_over_n_estimators(depth)
    plt.figure()
    plt.plot(ns, tr, label="Train", marker="o")
    plt.plot(ns, va, label="Validation", marker="s")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.title(f"AdaBoost accuracy vs n_estimators (max_depth={depth})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Sélection du modèle optimal par validation
best_score, best_cfg, best_model = -np.inf, None, None
for depth in [1, 5]:
    for n in range(10, 201, 10):
        base = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model = AdaBoostClassifier(
            estimator=base, n_estimators=n, algorithm="SAMME.R", random_state=42
        )
        model.fit(X_tr, ybin_tr)
        val_acc = accuracy_score(ybin_val, model.predict(X_val))
        if val_acc > best_score:
            best_score = val_acc
            best_cfg = {"max_depth": depth, "n_estimators": n}
            best_model = model

print("\n=== AdaBoost : Modèle sélectionné (validation) ===")
print("Config optimale :", best_cfg, f"| Accuracy validation : {best_score:.4f}")

# Entraînement final sur X_train complet et évaluation sur test + temps
t0 = time.perf_counter()
final_base = DecisionTreeClassifier(max_depth=best_cfg["max_depth"], random_state=42)
final_ada = AdaBoostClassifier(
    estimator=final_base,
    n_estimators=best_cfg["n_estimators"],
    algorithm="SAMME.R",
    random_state=42,
)
final_ada.fit(X_train, ybin_train)
train_time_ada = time.perf_counter() - t0
t1 = time.perf_counter()
y_test_pred_ada = final_ada.predict(X_test)
infer_time_ada = time.perf_counter() - t1
acc_test_ada = accuracy_score(ybin_test, y_test_pred_ada)
print("\n=== AdaBoost (final) ===")
print(f"Temps apprentissage : {train_time_ada:.3f}s | Temps inférence : {infer_time_ada:.3f}s")
print(f"Accuracy test : {acc_test_ada:.4f}")

# Importance des variables dans AdaBoost
if hasattr(final_ada, "feature_importances_"):
    importances = final_ada.feature_importances_
    feat_importance = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    print("\n=== Importance des variables (AdaBoost) ===")
    print(feat_importance)
else:
    print("\nAdaBoost ne fournit pas d'importance de variables avec cette configuration.")

# Commentaire biais/variance
print("\n=== Biais / Variance (commentaire) ===")
print("- max_depth=1 : apprenants très faibles -> biais élevé, variance faible ;")
print("- max_depth=5 : apprenants plus expressifs -> biais plus faible, variance plus élevée ;")
print("AdaBoost réduit le biais en combinant de nombreux faibles apprenants, avec une sensibilité possible au bruit.")