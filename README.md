# Image Prompter

## Description
Image Prompter est un outil d'analyse d'images qui utilise l'intelligence artificielle pour identifier les éléments visuels correspondant le mieux à une description textuelle donnée. Il combine deux modèles puissants :
- **SAM (Segment Anything Model)** pour la segmentation d'objets dans les images
- **CLIP (Contrastive Language-Image Pre-Training)** pour l'analyse sémantique

## Fonctionnalités
- Segmentation automatique des objets dans les images
- Analyse sémantique des segments détectés
- Comparaison avec un prompt textuel
- Identification des éléments les plus pertinents
- Support multi-plateforme (CPU/GPU)

## Prérequis
- Python 3.8+
- PyTorch
- CUDA (optionnel, pour l'accélération GPU)
- 16GB RAM minimum recommandé

## Installation
1. Cloner le repository
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation
### Mode GPU (CUDA)
Pour utiliser le GPU, assurez-vous que CUDA est installé et configurez la variable `SELECTED_GPU` dans le code :
```python
SELECTED_GPU = "0"  # Utilise le premier GPU
```

### Mode CPU
Pour utiliser le CPU uniquement, définissez :
```python
SELECTED_GPU = ""  # Force l'utilisation du CPU
```

## Performance
- **Mode GPU** : Recommandé pour le traitement de grands volumes d'images
- **Mode CPU** : Adapté pour le test et le développement, mais plus lent pour le traitement

## Limitations
- Le traitement sur CPU peut être significativement plus lent
- Nécessite le téléchargement des modèles (~2.4GB pour SAM)