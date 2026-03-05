# Import des librairies
import os
import sys
from pathlib import Path

try:
    project_root = Path(__file__).resolve().parent
except NameError:
    project_root = Path.cwd()

external_path = project_root / "external"
if str(external_path) not in sys.path:
    sys.path.insert(0, str(external_path))

import tabata as tbt

# ============================================================
# PARAMETRES MODIFIABLES
# ============================================================

parent = Path(__file__).resolve().parent.parent
AIRCRAFT_FILES = [
    parent / "Aircraft_01.h5",
    parent / "Aircraft_02.h5",
    parent / "Aircraft_03.h5",
]

ALT_COLUMN_ORIGINAL = "ALT [FT]"
ALT_COLUMN_NEW      = "ALT"

MIN_STD_ALT          = 10
MIN_ALT_MAX          = 2000
CRUISE_MARGIN_METERS = 150
ROLLING_WINDOW       = 20
TAKEOFF_SLOPE_THRESHOLD = 1
TAKEOFF_BEFORE       = 50
TAKEOFF_AFTER        = 400

# ============================================================
# FONCTIONS
# ============================================================

def remove_existing_files(prefix):
    """Supprime les anciens fichiers H5 générés (décollage et croisière)."""
    for suffix in ["decollage.h5", "croisiere.h5"]:
        filename = f"{prefix}_{suffix}"
        if os.path.exists(filename):
            os.remove(filename)


def build_clean_flights(storename):
    """
    Charge un fichier H5, filtre les vols invalides,
    convertit l'altitude ft -> m, et ajoute la colonne CR.
    Retourne une liste de DataFrames propres (en mémoire, sans fichier intermédiaire).
    """
    ds = tbt.Opset(storename)
    clean_flights = []

    for df in ds:

        # 1. DataFrame vide ou sans colonne
        if df.empty or df.shape[1] == 0:
            continue

        # 2. Normalisation des noms de colonnes
        df.columns = [c.strip().upper() for c in df.columns]

        # 3. Colonne altitude absente
        if ALT_COLUMN_ORIGINAL not in df.columns:
            continue

        alt = df[ALT_COLUMN_ORIGINAL]

        # 4. Trop de valeurs manquantes
        if alt.isna().mean() > 0.2:
            continue

        # 5. Signal quasi constant (pas un vrai vol)
        if alt.std() < MIN_STD_ALT:
            continue

        # 6. Avion n'atteint pas 2000 ft
        if alt.max() < MIN_ALT_MAX:
            continue

        # 7. Conversion ft -> m
        df[ALT_COLUMN_ORIGINAL] = alt * 0.3048

        # 8. Renommage de la colonne
        df.rename(columns={ALT_COLUMN_ORIGINAL: ALT_COLUMN_NEW}, inplace=True)

        # 9. Ajout de la colonne CR (croisière)
        mx = df[ALT_COLUMN_NEW].max()
        df["CR"] = df[ALT_COLUMN_NEW] > (mx - CRUISE_MARGIN_METERS)

        clean_flights.append(df)

    return clean_flights


def extract_takeoff_and_cruise(clean_flights, prefix):
    """
    Extrait les phases de décollage et de croisière depuis une liste de DataFrames.
    Produit deux fichiers : {prefix}_decollage.h5 et {prefix}_croisiere.h5.
    """
    ds_decollage = tbt.Opset(f"{prefix}_decollage.h5")
    ds_croisiere = tbt.Opset(f"{prefix}_croisiere.h5")

    for df in clean_flights:

        # --- DÉCOLLAGE ---
        pente        = df[ALT_COLUMN_NEW].diff()
        pente_lissee = pente.rolling(window=ROLLING_WINDOW).mean()
        indices_montee = df.index[pente_lissee > TAKEOFF_SLOPE_THRESHOLD]

        if len(indices_montee) > 0:
            idx_debut = indices_montee[0]
            start     = max(0, idx_debut - TAKEOFF_BEFORE)
            end       = min(len(df), idx_debut + TAKEOFF_AFTER)
            ds_decollage.put(df.iloc[start:end])

        # --- CROISIÈRE ---
        indices_croisiere = df.index[df["CR"] == True]

        if len(indices_croisiere) > 0:
            idx_start = indices_croisiere[0]
            idx_end   = indices_croisiere[-1]
            ds_croisiere.put(df.loc[idx_start:idx_end])

    print(f"  {prefix}_decollage.h5 : {len(ds_decollage)} segments")
    print(f"  {prefix}_croisiere.h5 : {len(ds_croisiere)} segments")


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    for idx, aircraft_file in enumerate(AIRCRAFT_FILES, start=1):
        prefix = f"aircraft_{idx}"
        print(f"\n--- Traitement de Aircraft_0{idx} ---")

        remove_existing_files(prefix)

        clean_flights = build_clean_flights(aircraft_file)
        print(f"  Vols valides après nettoyage : {len(clean_flights)}")

        extract_takeoff_and_cruise(clean_flights, prefix)

    print("\nExtraction terminée. Fichiers générés :")
    for idx in range(1, 4):
        print(f"  aircraft_{idx}_decollage.h5")
        print(f"  aircraft_{idx}_croisiere.h5")


if __name__ == "__main__":
    main()