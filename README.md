# Projet de Fin de Module – Deep Learning

**EMSI Casablanca – École Marocaine des Sciences de l'Ingénieur**  
**Filière : Informatique | Module : Deep Learning | Année : 2025–2026**

---

## Idée Générale du Projet

Ce projet démontre l'adaptation des architectures de deep learning à trois grandes familles de données :

- **Données tabulaires** → MLP (Perceptron Multi-Couche)
- **Images** → CNN (Réseau Neuronal Convolutionnel)
- **Séquences textuelles** → RNN / LSTM / GRU / Seq2Seq

Chaque partie contient une étude théorique, une implémentation PyTorch complète, une expérimentation comparative et une analyse critique. L'ensemble est conçu pour s'exécuter directement sur **Google Colab** (GPU T4).

---

## Structure des Dossiers et Fichiers

```
DEEP_LEARNING/
│
├── README.md                               ← Ce fichier
│
├── Part_I_MLP/
│   ├── MLP_Tabular_Classification.ipynb   ← Notebook principal (Google Colab)
│   └── src/
│       ├── __init__.py
│       ├── data_prep.py               ← Chargement/normalisation Breast Cancer
│       ├── models.py                  ← MLP Sequential + Classe perso + inits
│       └── utils.py                   ← Boucle entraîn., métriques, sauvegarde
│
├── Part_II_CNN/
│   ├── CNN_Image_Classification.ipynb     ← Notebook principal (Google Colab)
│   └── src/
│       ├── __init__.py
│       ├── conv_ops.py                ← Corrél. croisée, max/avg-pool manuels
│       ├── models.py                  ← MLP baseline, LeNet-5, CNN amélioré
│       └── utils.py                   ← Entraîn., visualisation feature maps
│
├── Part_III_RNN_Seq2Seq/
│   ├── RNN_LSTM_GRU_Seq2Seq.ipynb         ← Notebook principal (Google Colab)
│   └── src/
│       ├── __init__.py
│       ├── data_prep.py               ← Vocabulaire, CharLM, TranslationDataset
│       ├── models.py                  ← RecurrentLM, Encoder, Decoder, Seq2Seq
│       └── evaluate.py                ← Perplexité, BLEU, Greedy, Beam Search
│
└── docs/
    └── Rapport_Scientifique.md        ← Rapport + question transversale finale
```

---

## Partie I – MLP et Ingénierie PyTorch

**Dataset** : Breast Cancer Wisconsin (569 échantillons, 30 features, 2 classes)

**Contenu du notebook** :
- Concepts : `nn.Module`, `state_dict`, `named_parameters`, device, propagation avant/arrière
- Préparation : nettoyage, normalisation Z-score, split 70/10/20
- Deux architectures MLP : `nn.Sequential` vs classe `nn.Module` personnalisée
- Trois initialisations : Gaussienne, Constante, **Xavier** (meilleure)
- Entraînement avec early stopping, ReduceLROnPlateau, weight decay
- Métriques : Accuracy, Precision, Recall, F1-Score, Matrice de confusion
- Sauvegarde/rechargement `state_dict`

**Résultat attendu** : ~97–98% accuracy sur le test set avec initialisation Xavier.

---

## Partie II – CNN et Vision par Ordinateur

**Dataset** : MNIST (70 000 images 28×28, 10 classes de chiffres manuscrits)

**Contenu du notebook** :
- Pourquoi le MLP est inadapté aux images (localité, partage des poids, hiérarchie)
- Calculs manuels : taille de sortie convolution, pooling, dimensionnement LeNet
- Implémentations manuelles : corrélation croisée 2D, max-pooling, average-pooling
- Validation contre les couches PyTorch (`nn.Conv2d`, `F.max_pool2d`)
- Trois modèles : MLP Baseline, LeNet-5, CNN Amélioré (BN + MaxPool + Conv 1×1)
- Étude architecturale : padding, stride, type de pooling, nombre de filtres
- Visualisation des feature maps via forward hooks
- Comparaison finale MLP vs CNN sur accuracy test

**Résultat attendu** : MLP ~97.5% | LeNet-5 ~98.5% | CNN Amélioré ~99.2%

---

## Partie III – RNN, LSTM, GRU et Seq2Seq

**Dataset** : Corpus textuel char-level (LM) + paires EN→FR (Seq2Seq démonstration)

**Contenu du notebook** :
- Théorie : modèle de langage, règle de chaîne, perplexité, BPTT
- Équations LSTM (4 portes) et GRU (2 portes)
- Implémentation unifiée `RecurrentLM` (cell_type = rnn/lstm/gru)
- Entraînement avec gradient clipping (`clip_grad_norm_`)
- Démonstration de l'effet du clipping sur la norme des gradients
- Génération de texte par température sampling
- Architecture Seq2Seq : Encodeur GRU bidirectionnel + Attention Bahdanau + Décodeur
- Teacher forcing (ratio configurable)
- Deux stratégies de décodage : **glouton** et **beam search** (k=3)
- Évaluation : perplexité (LM) et score BLEU (traduction)

**Résultat attendu** : GRU ≈ LSTM >> RNN simple en perplexité ; Beam Search améliore BLEU.

---

## Question Transversale Finale

> *Comment le deep learning adapte-t-il ses architectures à la structure des données – tabulaire, image et séquentielle ?*

Voir `docs/Rapport_Scientifique.md` pour la discussion scientifique complète.

| Données | Architecture | Inductive Bias principal | Mécanisme Clé |
|---------|-------------|--------------------------|---------------|
| Tabulaire | MLP | Aucun (approximation universelle) | Couches denses + BatchNorm |
| Image | CNN | Localité + équivariance translation | Filtres partagés + Pooling |
| Séquence | RNN/LSTM/GRU | Ordre temporel + mémoire | État caché récurrent + Portes |
| Séq→Séq | Seq2Seq | Longueur variable | Encodeur-Décodeur + Attention |

---

## Utilisation sur Google Colab

1. Ouvrir [Google Colab](https://colab.research.google.com/)
2. `Fichier → Ouvrir le notebook → GitHub` → coller l'URL du repo
3. Sélectionner le notebook souhaité (`Part_I_MLP/`, `Part_II_CNN/`, `Part_III_RNN_Seq2Seq/`)
4. `Exécution → Tout exécuter` (activer GPU : `Exécution → Modifier le type d'exécution → T4 GPU`)

Chaque notebook est **autonome** : toutes les classes et fonctions sont définies directement dans les cellules.

---

## Dépendances

```
torch >= 2.0
torchvision >= 0.15
scikit-learn >= 1.3
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
pandas >= 2.0
```

Toutes installées automatiquement via `!pip install` au début de chaque notebook.
