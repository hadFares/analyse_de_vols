# =============================================================
# main01.py — Test du pipeline de maintenance prédictive EGT
# =============================================================
import sys
from pathlib import Path
import pandas as pd

# --- Résolution des chemins ---
try:
    project_root = Path(__file__).resolve().parent
except NameError:
    project_root = Path.cwd()

sys.path.insert(0, str(project_root / "external"))

from tabata import Opset
from config import AIRCRAFT_FILES, N_TRAIN_VOLS, VAR_CONTEXTE, VAR_MOTEUR, TARGET, VAR_CONTEXTE_FINAL, VAR_MOTEUR_FINAL
from RF_model import select_train_vols, train_rf, compute_residuals, train_final_model
import RF_model as _rf
_rf.TARGET = TARGET   # TARGET n'est pas importé dans RF_model.py — on le patch ici


# =============================================================
# ÉTAPE 1 — Chargement des données
# =============================================================
print("=" * 60)
print("ÉTAPE 1 — Chargement des données")
print("=" * 60)

aircraft_file = AIRCRAFT_FILES[0]   # On teste sur l'avion 1
print(f"Fichier utilisé : {aircraft_file}")

opset = Opset(str(aircraft_file))
n_vols = len(opset)
print(f"Nombre de vols disponibles : {n_vols}")

if n_vols == 0:
    raise RuntimeError("L'Opset est vide — vérifiez le chemin du fichier H5.")

# Aperçu du premier vol
vol_exemple = opset[0]
print(f"Colonnes disponibles ({len(vol_exemple.columns)}) : {list(vol_exemple.columns)}")
print(f"Taille d'un vol exemple : {vol_exemple.shape}")


# =============================================================
# ÉTAPE 2 — Sélection des vols d'entraînement RF
# =============================================================
print("\n" + "=" * 60)
print(f"ÉTAPE 2 — Sélection des {N_TRAIN_VOLS} vols d'entraînement RF")
print("=" * 60)

X_train, Y_train = select_train_vols(opset, n_train=N_TRAIN_VOLS)

print(f"X_train : {X_train.shape}  (variables contextuelles : {VAR_CONTEXTE})")
print(f"Y_train : {Y_train.shape}  (variables moteur : {VAR_MOTEUR})")
print(f"Valeurs manquantes dans X_train : {X_train.isnull().sum().sum()}")
print(f"Valeurs manquantes dans Y_train : {Y_train.isnull().sum().sum()}")


# =============================================================
# ÉTAPE 3 — Entraînement du Random Forest
# =============================================================
print("\n" + "=" * 60)
print("ÉTAPE 3 — Entraînement du Random Forest")
print("=" * 60)

print("Entraînement en cours...")
rf_model = train_rf(X_train, Y_train)
print(f"Modèle entraîné : {rf_model}")
print(f"Nombre d'arbres : {rf_model.n_estimators}")
print(f"Profondeur max  : {rf_model.max_depth}")

# Score in-sample pour vérification rapide
score_train = rf_model.score(X_train, Y_train)
print(f"R² in-sample (doit être proche de 1) : {score_train:.4f}")


# =============================================================
# ÉTAPE 4 — Calcul des résidus sur tous les vols
# =============================================================
print("\n" + "=" * 60)
print("ÉTAPE 4 — Calcul des résidus (tous les vols)")
print("=" * 60)

print("Calcul en cours...")
residuals, residuals_norm, norm_params, contexts = compute_residuals(rf_model, opset)

print(f"residuals      : {residuals.shape}")
print(f"residuals_norm : {residuals_norm.shape}")
print(f"contexts       : {contexts.shape}")

print("\nMoyenne des résidus bruts (doit être ~0 sur les vols train) :")
cols_moteur = [c for c in residuals.columns if c != 'id_vol']
print(residuals[cols_moteur].mean().round(4).to_string())

print("\nParamètres de normalisation — moyenne :")
print(norm_params['mean'].round(4).to_string())
print("Paramètres de normalisation — écart-type :")
print(norm_params['std'].round(4).to_string())

print(f"\nNombre de vols in-sample  (id_vol <= {N_TRAIN_VOLS}) : "
      f"{(residuals_norm['id_vol'] <= N_TRAIN_VOLS).sum()} lignes")
print(f"Nombre de vols out-sample (id_vol >  {N_TRAIN_VOLS}) : "
      f"{(residuals_norm['id_vol'] > N_TRAIN_VOLS).sum()} lignes")


# =============================================================
# ÉTAPE 5 — Nettoyage des noms de colonnes (requis par LightGBM)
# =============================================================
# LightGBM sérialise les noms de features en JSON :
# les caractères [ ] % et espaces provoquent une erreur fatale.
# On nettoie les colonnes de residuals_norm et contexts AVANT
# de les passer à train_final_model.

def clean_col_names(df):
    """Remplace les caractères spéciaux JSON par des underscores."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace(r'[\[\]{}",:\ ]', '_', regex=True)
        .str.replace('%', 'pct', regex=False)
        .str.replace(r'_+', '_', regex=True)
        .str.rstrip('_')
    )
    return df

residuals_norm = clean_col_names(residuals_norm)
contexts       = clean_col_names(contexts)

# Mise à jour des listes de variables avec les nouveaux noms
VAR_CONTEXTE_CLEAN       = list(clean_col_names(pd.DataFrame(columns=VAR_CONTEXTE)).columns)
VAR_MOTEUR_CLEAN         = list(clean_col_names(pd.DataFrame(columns=VAR_MOTEUR)).columns)
VAR_CONTEXTE_FINAL_CLEAN = list(clean_col_names(pd.DataFrame(columns=VAR_CONTEXTE_FINAL)).columns)
VAR_MOTEUR_FINAL_CLEAN   = list(clean_col_names(pd.DataFrame(columns=VAR_MOTEUR_FINAL)).columns)
TARGET_CLEAN             = clean_col_names(pd.DataFrame(columns=[TARGET])).columns[0]

print(f"Features LGBM nettoyées : {VAR_CONTEXTE_FINAL_CLEAN + VAR_MOTEUR_FINAL_CLEAN}")

# On réimporte les constantes nettoyées pour que train_final_model les utilise
import config as _cfg
_cfg.VAR_CONTEXTE       = VAR_CONTEXTE_CLEAN
_cfg.VAR_MOTEUR         = VAR_MOTEUR_CLEAN
_cfg.VAR_CONTEXTE_FINAL = VAR_CONTEXTE_FINAL_CLEAN
_cfg.VAR_MOTEUR_FINAL   = VAR_MOTEUR_FINAL_CLEAN
_cfg.TARGET             = TARGET_CLEAN

_rf.VAR_CONTEXTE       = VAR_CONTEXTE_CLEAN
_rf.VAR_MOTEUR         = VAR_MOTEUR_CLEAN
_rf.VAR_CONTEXTE_FINAL = VAR_CONTEXTE_FINAL_CLEAN
_rf.VAR_MOTEUR_FINAL   = VAR_MOTEUR_FINAL_CLEAN
_rf.TARGET             = TARGET_CLEAN


# =============================================================
# ÉTAPE 6 — Entraînement du modèle final LightGBM
# =============================================================
print("\n" + "=" * 60)
print("ÉTAPE 6 — Entraînement du modèle final LightGBM (prédiction EGT)")
print("=" * 60)

print("Entraînement en cours...")
lgbm_model = train_final_model(residuals_norm, contexts)
print(f"Modèle entraîné : {lgbm_model}")

# Vérification rapide : prédiction sur un sous-ensemble out-sample
mask_oos = contexts['id_vol'] > N_TRAIN_VOLS
X_oos = pd.concat(
    [contexts[mask_oos][VAR_CONTEXTE_FINAL_CLEAN],
     residuals_norm[mask_oos][VAR_MOTEUR_FINAL_CLEAN]],
    axis=1
)
Y_oos = contexts[mask_oos][TARGET_CLEAN]

score_oos = lgbm_model.score(X_oos, Y_oos)
print(f"R² out-of-sample sur EGT_1 : {score_oos:.4f}")


# =============================================================
# RÉSUMÉ
# =============================================================
print("\n" + "=" * 60)
print("RÉSUMÉ DU PIPELINE")
print("=" * 60)
print(f"  Avion                  : {aircraft_file.name}")
print(f"  Vols totaux            : {n_vols}")
print(f"  Vols RF (in-sample)    : {N_TRAIN_VOLS}")
print(f"  Vols LGBM (out-sample) : {n_vols - N_TRAIN_VOLS}")
print(f"  R² RF  (in-sample)     : {score_train:.4f}")
print(f"  R² LGBM (out-sample)   : {score_oos:.4f}")

print("Pipeline terminé avec succès ✓")