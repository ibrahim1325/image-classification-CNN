"""
Script pour organiser les datasets dans la structure attendue par le notebook.

Ce script copie un échantillon aléatoire d'images depuis les datasets extraits
vers une structure standardisée utilisable par le notebook principal.

Usage :
    python organize_datasets.py

Avant exécution, modifier les chemins dans la section CONFIGURATION ci-dessous.
"""

import os
import shutil
import random
from pathlib import Path

# Seed pour la reproductibilité
random.seed(42)


def copier_images(source: str, destination: str, max_images: int = 333) -> None:
    """
    Copie un échantillon aléatoire d'images depuis source vers destination.

    Args:
        source: Dossier source contenant les images
        destination: Dossier de destination
        max_images: Nombre maximum d'images à copier
    """
    os.makedirs(destination, exist_ok=True)

    extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

    try:
        images = [f for f in os.listdir(source) if f.endswith(extensions)]
    except FileNotFoundError:
        print(f"   [ERREUR] Dossier introuvable : {source}")
        return

    random.shuffle(images)
    images = images[:max_images]

    for img in images:
        shutil.copy2(os.path.join(source, img), os.path.join(destination, img))

    print(f"   [OK] {len(images)} images copiées vers {destination}")


def organiser_intel(chemin_source: str, chemin_sortie: str, max_par_classe: int = 333) -> None:
    """Organise le dataset Intel Image Classification."""
    print("\n[INFO] Organisation du dataset Intel Image...")

    source_train = os.path.join(chemin_source, "seg_train", "seg_train")
    if not os.path.exists(source_train):
        source_train = os.path.join(chemin_source, "seg_train")

    if not os.path.exists(source_train):
        print(f"   [ERREUR] Dataset Intel introuvable dans {chemin_source}")
        return

    classes = [d for d in os.listdir(source_train)
               if os.path.isdir(os.path.join(source_train, d))]

    print(f"   Classes trouvées : {classes}")

    for classe in classes:
        src = os.path.join(source_train, classe)
        dst = os.path.join(chemin_sortie, "intel_image", classe)
        copier_images(src, dst, max_par_classe)

    print("[OK] Intel Image organisé !")


def organiser_wildfire(chemin_source: str, chemin_sortie: str, max_par_classe: int = 1000) -> None:
    """Organise le dataset Wildfire."""
    print("\n[INFO] Organisation du dataset Wildfire...")

    source = os.path.join(chemin_source, "train")
    if not os.path.exists(source):
        source = chemin_source

    if not os.path.exists(source):
        print(f"   [ERREUR] Dataset Wildfire introuvable dans {chemin_source}")
        return

    classes = [d for d in os.listdir(source)
               if os.path.isdir(os.path.join(source, d))]

    print(f"   Classes trouvées : {classes}")

    for classe in classes:
        src = os.path.join(source, classe)
        dst = os.path.join(chemin_sortie, "wildfire", classe)
        copier_images(src, dst, max_par_classe)

    print("[OK] Wildfire organisé !")


def organiser_garbage(chemin_source: str, chemin_sortie: str, max_par_classe: int = 333) -> None:
    """Organise le dataset Garbage Classification."""
    print("\n[INFO] Organisation du dataset Garbage Classification...")

    source = chemin_source
    for root, dirs, _ in os.walk(chemin_source):
        sous = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        if len(sous) >= 4:
            a_images = any(
                any(f.endswith(('.jpg', '.jpeg', '.png'))
                    for f in os.listdir(os.path.join(root, d)))
                for d in sous
            )
            if a_images:
                source = root
                break

    classes = [d for d in os.listdir(source)
               if os.path.isdir(os.path.join(source, d))]

    print(f"   Classes trouvées : {classes}")

    for classe in classes:
        src = os.path.join(source, classe)
        dst = os.path.join(chemin_sortie, "garbage_classification", classe)
        copier_images(src, dst, max_par_classe)

    print("[OK] Garbage Classification organisé !")


def organiser_lung_colon(chemin_source: str, chemin_sortie: str, max_par_classe: int = 400) -> None:
    """Organise le dataset Lung and Colon Cancer."""
    print("\n[INFO] Organisation du dataset Lung & Colon Cancer...")

    source = chemin_source
    for root, dirs, _ in os.walk(chemin_source):
        sous = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        if len(sous) == 5:
            source = root
            break

    classes = [d for d in os.listdir(source)
               if os.path.isdir(os.path.join(source, d))]

    print(f"   Classes trouvées : {classes}")

    for classe in classes:
        src = os.path.join(source, classe)
        dst = os.path.join(chemin_sortie, "lung_colon_cancer", classe)
        copier_images(src, dst, max_par_classe)

    print("[OK] Lung & Colon Cancer organisé !")


def main():
    """Point d'entrée principal du script."""

    # ============================================================
    # CONFIGURATION : modifier ces chemins selon votre installation
    # ============================================================

    # Dossier de sortie où seront créées les données pour le notebook
    chemin_sortie = "./data"

    # Chemins vers les dossiers extraits de chaque dataset
    intel_source = "./datasets/intel_image"
    wildfire_source = "./datasets/wildfire"
    garbage_source = "./datasets/garbage_classification"
    lung_source = "./datasets/lung_colon_cancer"

    # ============================================================
    # EXÉCUTION : décommentez les datasets à traiter
    # ============================================================

    organiser_intel(intel_source, chemin_sortie, max_par_classe=333)
    organiser_wildfire(wildfire_source, chemin_sortie, max_par_classe=1000)
    organiser_garbage(garbage_source, chemin_sortie, max_par_classe=333)
    organiser_lung_colon(lung_source, chemin_sortie, max_par_classe=400)

    print(f"\n[SUCCÈS] Structure créée dans : {chemin_sortie}")


if __name__ == "__main__":
    main()