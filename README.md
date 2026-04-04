# Analyse de données moteurs avion

## Description

Ce projet a pour objectif d'analyser des données de vols d'avions et de construire un modèle permettant de prédire la température des gaz d’échappement (EGT) à partir de variables de vol et de paramètres moteur.

L’approche repose sur une pipeline en deux étapes combinant :
- un modèle de type Random Forest pour modéliser le comportement nominal
- un modèle LightGBM pour affiner la prédiction à partir des résidus

---

## Structure du projet

- `extraction_donnees.py`  
  Nettoyage des données brutes et extraction des phases de vol (décollage et croisière)

- `config.py`  
  Définition des variables utilisées et des hyperparamètres des modèles

- `RF_model.py`  
  Implémentation de la pipeline de modélisation (Random Forest + LightGBM)
  
- `visu01.ipynb`  
  Notebook de visualisation des résultats sortis par le model lightGBM

- `external/`  
  Dépendances externes (notamment `tabata` pour la manipulation des fichiers de vols)

---

## Pipeline

### 1. Prétraitement des données

Le script `extraction_donnees.py` réalise les étapes suivantes :

- Chargement des vols depuis les fichiers `.h5`
- Filtrage des vols non exploitables :
  - données manquantes
  - altitude insuffisante
  - signal peu variable
- Conversion de l’altitude (ft → m)
- Détection des phases de vol :
  - décollage (basé sur la pente de l’altitude)
  - croisière (plateau d’altitude)
- Export des segments dans de nouveaux fichiers :
  - `*_decollage.h5`
  - `*_croisiere.h5`

### 2. Modélisation

#### a) Modèle Random Forest

- Entrée : variables de contexte (conditions de vol)
- Sortie : variables moteur

Ce modèle apprend le comportement nominal du moteur en fonction du contexte.


#### b) Calcul des résidus

- Calcul de l’écart entre :
  - valeurs réelles moteur
  - prédictions du Random Forest
- Normalisation des résidus (centrage et réduction)

Ces résidus représentent les écarts au comportement attendu.


#### c) Modèle final (LightGBM)

- Entrée :
  - variables de contexte sélectionnées
  - résidus normalisés sélectionnés
- Sortie :
  - EGT

Important :
Les vols utilisés pour entraîner le Random Forest sont exclus afin d’éviter un biais (résidus artificiellement faibles).


#### d) visualisation des résultats
- visualisation des résultats renvoyés par lightGBM, notament les résidus entre EGT prédit et EGT réel.
---



