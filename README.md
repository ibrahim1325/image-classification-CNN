# 🧠 Classification d'images par Deep Learning

> Projet académique réalisé dans le cadre du cours INF5082 (UQAM) — Session Hiver 2026

## 📋 Aperçu

Pipeline complet de classification d'images utilisant des réseaux de neurones convolutifs (CNN) et du Transfer Learning, appliqué à 4 datasets réels :

- 🔥 **Détection de feux de forêt** — imagerie satellite, 2 classes
- 🏞️ **Classification de scènes naturelles** — 25 000 images, 6 classes (Intel Image Classification)
- ♻️ **Tri automatique de déchets** — 15 000 images, 6 classes (Garbage Classification)
- 🏥 **Imagerie médicale histopathologique** — 25 000 images, 5 classes (cancer poumon/côlon)

## 🛠️ Stack technique

- **Langage :** Python 3.9+
- **Deep Learning :** PyTorch, torchvision
- **Machine Learning :** scikit-learn, XGBoost, LightGBM
- **Visualisation :** Matplotlib, Seaborn
- **Optimisation :** GridSearchCV
- **Environnement :** Jupyter Notebook

## 🎯 Méthodologie

### Partie 1 : CNN personnalisé

Conception et entraînement d'un CNN à 3 blocs convolutifs sur chaque dataset :

- **Bloc 1 :** Conv2d(3→32) + BatchNorm + ReLU + MaxPool
- **Bloc 2 :** Conv2d(32→64) + BatchNorm + ReLU + MaxPool
- **Bloc 3 :** Conv2d(64→128) + BatchNorm + ReLU + MaxPool
- **Classifieur :** FC(128×28×28 → 256) + Dropout(0.5) + FC(256 → nb_classes)

Entraînement : 20 époques, optimiseur Adam (lr=0.001), CrossEntropyLoss, augmentation de données (flip, rotation, color jitter).

### Partie 2 : Transfer Learning

Utilisation de 5 architectures pré-entraînées sur ImageNet comme extracteurs de features :

| Architecture | Taille des features |
|---|---|
| AlexNet | 4096 |
| VGGNet (VGG16) | 4096 |
| ResNet-50 | 2048 |
| GoogLeNet | 1024 |
| MobileNet V2 | 1280 |

Combinées à 5 classifieurs ML : Decision Tree, AdaBoost, XGBoost, LightGBM, KNN.
**25 combinaisons** testées par dataset.

### Partie 3 : Optimisation

Recherche d'hyperparamètres optimaux via **GridSearchCV** avec validation croisée pour chaque combinaison architecture/classifieur, suivie d'une analyse comparative critique.

## 📊 Métriques évaluées

Pour chaque modèle et chaque dataset :
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Matrice de confusion
- Courbes d'entraînement (loss + accuracy)

## 📥 Installation et utilisation

### 1. Cloner le repo

```bash
git clone https://github.com/ibrahim1325/image-classification-cnn.git
cd image-classification-cnn
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger les datasets

Les datasets ne sont pas inclus dans ce repo (trop volumineux). Téléchargez-les depuis Kaggle :

1. [Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)
2. [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
3. [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
4. [Lung and Colon Cancer](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

### 4. Organiser les datasets

Extrayez les archives dans le dossier `datasets/`, puis exécutez :

```bash
python scripts/organize_datasets.py
```

### 5. Lancer le notebook

```bash
jupyter notebook notebook.ipynb
```

## 📂 Structure du projet

```
image-classification-cnn/
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
├── .gitignore                   # Fichiers ignorés par Git
├── notebook.ipynb               # Notebook principal (3 parties)
├── rapport.pdf                  # Rapport d'analyse complet
└── scripts/
    └── organize_datasets.py     # Préparation des datasets
```

## 👤 Auteur

**Ibrahima Berete**
Étudiant en génie logiciel à l'UQAM

• 💻 [GitHub](https://github.com/ibrahim1325)

## 📄 Licence

Ce projet est réalisé à des fins éducatives dans le cadre d'un cours universitaire.
