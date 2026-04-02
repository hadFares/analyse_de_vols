from pathlib import Path

# --- Chemins ---
DATA_DIR = Path(__file__).resolve().parent.parent
AIRCRAFT_FILES = [
    DATA_DIR / "aircraft_1_decollage.h5",
    DATA_DIR / "aircraft_2_decollage.h5",
    DATA_DIR / "aircraft_3_decollage.h5",
]

# --- Paramètres de sélection des données ---
N_TRAIN_VOLS = 100  # nombre de vols utilisés pour entraîner le RF

# --- Variables du modèle RF ---
VAR_CONTEXTE = [
    'ALT',
    'M [MACH]',
    'TAT [DEG C]',
    'P0_1 [PSIA]',
    'N1_1 [% RPM]',
    'N1_2 [% RPM]',
]


VAR_MOTEUR = [
    'N2_1 [% RPM]', 'N2_2 [% RPM]',
    'PS3_1 [PSIA]', 'PS3_2 [PSIA]',
    'T_OIL_1 [DEG C]', 'T_OIL_2 [DEG C]',
    'Q_1 [LB/H]', 'Q_2 [LB/H]',
]

# --- Variable cible du modèle final ---
TARGET = 'EGT_1 [DEG C]'

# --- Features sélectionnées pour le modèle LightGBM final ---
# Sélection par score d'information mutuelle avec EGT_1,
# en excluant les variables trop corrélées à EGT_1 (T5, T3, EGT_2...)
# et en se limitant à des variables présentes dans VAR_CONTEXTE ou VAR_MOTEUR.
VAR_CONTEXTE_FINAL = [
    'N1_1 [% RPM]',   # MI = 0.919 — poussée demandée moteur 1
    'N1_2 [% RPM]',   # MI = 0.915 — régime global
    'TAT [DEG C]',    # MI = 0.367 — condition thermique extérieure
    'M [MACH]',       # MI = 0.240 — vitesse vol
]

VAR_MOTEUR_FINAL = [
    'N2_1 [% RPM]',   # MI = 0.989 — vitesse corps haute pression moteur 1
    'N2_2 [% RPM]',   # MI = 0.955 — asymétrie inter-moteurs
    'Q_1 [LB/H]',     # MI = 0.521 — débit carburant moteur 1
    'PS3_1 [PSIA]',   # MI = 0.399 — pression compresseur moteur 1
]

# --- Hyperparamètres Random Forest ---
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
}

LGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_jobs': -1,  # utilise tous les coeurs disponibles
}