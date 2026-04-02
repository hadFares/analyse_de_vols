# Import des librairies
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
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

# On place dans le tableu AIRCRAFT_FILES les données extraites par le précédent scrypte.
parent = Path(__file__).resolve().parent.parent
AIRCRAFT_FILES = [
    parent / "aircraft_1_decollage.h5",
    parent / "aircraft_2_decollage.h5",
    parent / "aircraft_3_decollage.h5",
]

""" Plan :
1- On crée un dataframe extrait des fichiers AIRCRAFT_FILES beaucoup plus petit,
    sur lequel on peut faire des stat.  
2- On Boucle sur les colones. A chaque itération, On calcule l'information mutuelle entre
    la colone courante et la colone EGT
3- On peut faire un classement du la plus haute à la plus faible.
"""

"""
structure des fichiers .h5 :
Aircraft_01.h5
├── record_0/     ← vol 0 → un DataFrame (lignes = instants de temps, colonnes = paramètres)
├── record_1/     ← vol 1 → un DataFrame
├── record_2/     ← vol 2 → un DataFrame
└── ...
"""

"""
1er étape : 
"""
dfs = [] # On initialise la liste qui va contenir les données à analyser
store = pd.HDFStore(AIRCRAFT_FILES[1], mode="r") # On place dans store la fichier h5,
# mais en mode lecture simplement : On ne charge pas tout le fichier dans la RAM
# Nature de l'objet :HDF5store de la biblioteque pandas

for key in store.keys() : # store.keys contient la liste de toutes les keys

    df = store[key] # dataframe du fichier hdf5
    n = min(25, len(df))
    dfs.append(df.sample(n=n, random_state=42))

store.close()
df = pd.concat(dfs, ignore_index=True) # dataframe final, concaréné, pour lequel on va pouvoir
#calculer les informations mutuelles

# On sépare la colonne EGT du reste pour pouvoir utiliser mutual_info_regression proprement :
y = df["EGT_1 [DEG C]"]
X = df.drop(columns=["EGT_1 [DEG C]"])

# Fonction qui renvoie les informations mutuelles entre une matrice et un vecteur
Influences = mutual_info_regression(X, y, discrete_features='auto', n_neighbors=3, copy=False)

# On associe chaque score au nom de sa colonne
mi_series = pd.Series(Influences, index=X.columns)

print(mi_series.sort_values(ascending=True))
# On trie par ordre décroissant et on prend les 5 premiers
top5 = mi_series.sort_values(ascending=True).head(5)

#print(top5)



