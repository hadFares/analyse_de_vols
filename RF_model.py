# Import des librairies
# =======================================
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from config import VAR_CONTEXTE, VAR_MOTEUR, N_TRAIN_VOLS, RF_PARAMS, LGBM_PARAMS, VAR_CONTEXTE_FINAL, VAR_MOTEUR_FINAL

try:
    project_root = Path(__file__).resolve().parent
except NameError:
    project_root = Path.cwd()

external_path = project_root / "external"

if str(external_path) not in sys.path:
    sys.path.insert(0, str(external_path))

from tabata import Opset
# =============================================




# Fonction qui extrait les N_train premiers vols, puis donne X_train et Y_train correspondant.
def select_train_vols(opset, n_train=N_TRAIN_VOLS):
    """
    Sélectionne les n_train premiers vols d'un Opset et retourne
    X_train (variables contextuelles) et Y_train (variables moteur).
    
    Args:
        opset : Opset tabata chargé
        n_train : nombre de vols d'entraînement (défaut : N_TRAIN_VOLS)
    
    Returns:
        X_train : DataFrame des variables contextuelles
        Y_train : DataFrame des variables moteur
    """
    selection = []

    for vol in opset.iterator(n_train):
        vol.insert(0, 'id_vol', opset.sigpos)
        selection.append(vol)

    selection_df = pd.concat(selection)

    X_train = selection_df[VAR_CONTEXTE]
    Y_train = selection_df[VAR_MOTEUR]

    return X_train, Y_train


def train_rf(X_train, Y_train, rf_params=RF_PARAMS):
    """
    Entraîne un RandomForestRegressor sur X_train et Y_train.
    
    Args:
        X_train : DataFrame des variables contextuelles
        Y_train : DataFrame des variables moteur
        rf_params : dictionnaire des hyperparamètres (défaut : RF_PARAMS)
    
    Returns:
        model : RandomForestRegressor entraîné
    """
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, Y_train)
    return model


def compute_residuals(model, opset):
    residuals = []
    contexts = []

    for i in range(len(opset)):
        vol = opset[i]

        pred = model.predict(vol[VAR_CONTEXTE])
        df_pred = pd.DataFrame(pred, columns=VAR_MOTEUR, index=vol.index)

        res = vol[VAR_MOTEUR] - df_pred
        res.insert(0, 'id_vol', opset.sigpos)
        residuals.append(res)

        # On sauvegarde aussi le contexte et la cible au même index
        ctx = vol[VAR_CONTEXTE + [TARGET]]
        ctx.insert(0, 'id_vol', opset.sigpos)
        contexts.append(ctx)

    residuals = pd.concat(residuals)
    contexts = pd.concat(contexts)

    # Normalisation
    cols = VAR_MOTEUR
    norm_params = {
        'mean': residuals[cols].mean(),
        'std': residuals[cols].std()
    }
    residuals_norm = residuals.copy()
    residuals_norm[cols] = (residuals[cols] - norm_params['mean']) / norm_params['std']

    return residuals, residuals_norm, norm_params, contexts


def train_final_model(residuals_norm, contexts):
    """
    Entraîne un modèle LightGBM pour prédire l'EGT à partir des variables
    contextuelles et des résidus normalisés du RF.

    On exclut les N_TRAIN_VOLS premiers vols car leurs résidus sont in-sample
    (le RF les a vus pendant l'entraînement) et donc artificiellement petits.

    Args:
        residuals_norm : DataFrame des résidus normalisés + colonne id_vol
        contexts       : DataFrame des variables contextuelles + TARGET + colonne id_vol

    Returns:
        model : LGBMRegressor entraîné
    """

    # On exclut les N_TRAIN_VOLS premiers vols des deux DataFrames
    # La condition porte sur id_vol qui est présent dans les deux
    mask = residuals_norm['id_vol'] > N_TRAIN_VOLS
    residuals_train = residuals_norm[mask]
    contexts_train = contexts[mask]

    # Construction de X_final en concaténant horizontalement (axis=1)
    # les variables contextuelles et les résidus normalisés
    # On exclut id_vol des deux côtés car ce n'est pas une feature
    X_final = pd.concat(
        [
            contexts_train[VAR_CONTEXTE_FINAL],   # 4 variables contextuelles sélectionnées
            residuals_train[VAR_MOTEUR_FINAL]      # 4 résidus moteur sélectionnés
        ],
        axis=1
    )

    # La cible est l'EGT extraite du DataFrame contexts
    Y_final = contexts_train[TARGET]

    # Vérification que X et Y ont bien le même nombre de lignes
    # Indispensable car pd.concat sur axis=1 aligne sur les index —
    # si les index ne correspondent pas, pandas introduit des NaN silencieusement
    if X_final.shape[0] != Y_final.shape[0]:
        raise ValueError(
            f"X_final ({X_final.shape[0]} lignes) et Y_final "
            f"({Y_final.shape[0]} lignes) ne sont pas alignés."
        )

    # Vérification qu'il n'y a pas de NaN dans X ou Y
    # Un NaN dans les features ferait planter ou biaiser le modèle silencieusement
    if X_final.isnull().any().any():
        raise ValueError("X_final contient des valeurs manquantes.")
    if Y_final.isnull().any():
        raise ValueError("Y_final contient des valeurs manquantes.")

    # Entraînement du modèle LightGBM
    # **LGBM_PARAMS dépacke le dictionnaire de config en arguments nommés
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_final, Y_final)

    return model