# Analyse de sentiments sur 1 million de tweets

> Détecter automatiquement les **tweets négatifs** pour aider une entreprise à repérer un **bad buzz** avant qu'il ne prenne de l'ampleur, en comparant des modèles de complexité croissante, de la régression logistique jusqu'à **BERT**.

![Python](https://img.shields.io/badge/Python-3-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00) ![Transformers](https://img.shields.io/badge/Hugging%20Face-BERT-yellow) ![Azure](https://img.shields.io/badge/Azure-App%20Service-0078D4)

## Le problème

Une compagnie aérienne veut surveiller ce qui se dit d'elle sur les réseaux sociaux. Il faut un modèle capable de prédire si un tweet est **positif ou négatif**, puis le mettre à disposition sous la forme d'une API.

## Résultats clés

| | |
|---|---|
| Données | Environ **1 million de tweets** annotés positif / négatif |
| Modèles comparés | Régression logistique, réseau avec couche d'embedding, LSTM bidirectionnel, BERT |
| Meilleur modèle | **BERT** : 79,8 % de tweets bien classés sur le jeu de test |
| Modèle mis en production | **Régression logistique** : la plus rapide, dans une API Flask déployée sur Azure |

## Démarche

### 1. Exploration et nettoyage des données : notebook 2.1

Nettoyage des tweets (mentions, liens, ponctuation, mots vides), en gardant les négations comme « not » et « never », qui changent le sens d'une phrase. Comparaison de deux façons de ramener les mots à leur forme de base : la **racinisation** (stemming) et la **lemmatisation**. Séparation des données en jeux d'entraînement, de validation et de test.

### 2. Modèle simple : notebook 2.2

Chaque tweet est représenté par la moyenne des vecteurs **Word2Vec** de ses mots, puis classé par une **régression logistique**. Ce modèle sert de point de comparaison.

### 3. Réseau de neurones avec une couche d'embedding : notebook 2.3

Réseau **TensorFlow / Keras** qui apprend à partir des vecteurs de mots, testé avec **Word2Vec** et **GloVe**, sur des textes racinisés ou lemmatisés.

### 4. LSTM bidirectionnel : notebook 2.4

Le LSTM lit le tweet mot par mot, dans les deux sens, et tient compte de l'ordre des mots. Il comprend mieux des tournures comme « not good ».

### 5. BERT : notebook 2.5

Adaptation (fine-tuning) du modèle pré-entraîné **bert-base-uncased**, qui tient compte du contexte de chaque mot dans la phrase. Entraînement sur GPU (Google Colab).

### 6. Comparaison des modèles : notebook 2.6

Tous les modèles sont évalués sur les **99 333 tweets du jeu de test**, jamais utilisés pendant l'entraînement :

| Modèle | Tweets bien classés | Temps d'entraînement |
|---|---:|---:|
| Régression logistique (Word2Vec) | 74,2 % | 20 secondes |
| Réseau avec couche d'embedding | 75,4 à 75,6 % | 11 minutes |
| LSTM bidirectionnel | 77,2 à 77,6 % | 1 h 45 |
| **BERT** | **79,8 %** | 1 h 47 |

### 7. Mise en production : dossier `1_deploiement`

BERT donne le meilleur score, mais il est lourd et a besoin d'un GPU. La **régression logistique**, qui s'entraîne en quelques secondes et répond instantanément, a été choisie pour un premier prototype :
- une **API Flask** qui reçoit un tweet et renvoie son sentiment ;
- un **déploiement sur Azure App Service**, mis à jour automatiquement à chaque envoi de code grâce à **GitHub Actions**.

## Mise à jour de novembre 2026

Le projet est en cours de reprise :
- **nettoyage des données amélioré** ;
- **notebooks entièrement réexpliqués**, avec une conclusion « À retenir » après chaque résultat ;
- nouveaux résultats sur le jeu de validation : **75,7 %** de tweets bien classés pour la régression logistique (AUC de 0,835) et **77,6 %** pour le réseau avec embedding (AUC de 0,857) ;
- analyse des erreurs : les **négations** restent la principale difficulté des modèles qui ne tiennent pas compte de l'ordre des mots.

Prochaines étapes : réentraîner le LSTM et BERT sur les données nettoyées, puis comparer tous les modèles sur le jeu de test.

## Stack

Python, pandas, NumPy, spaCy, NLTK, Gensim (Word2Vec), GloVe, scikit-learn, TensorFlow / Keras, Hugging Face Transformers (BERT), Matplotlib, Seaborn, Flask, Azure App Service, GitHub Actions

## Données

Les données ne sont pas incluses dans le dépôt. Elles proviennent du jeu public [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) sur Kaggle (1,6 million de tweets annotés).

## Contenu du dépôt

| Fichier ou dossier | Contenu |
|---|---|
| `..._1_deploiement_122022/` | API Flask et fichiers de déploiement |
| `..._2_1_notebook_exploration_des_donnees_122022.ipynb` | Exploration et nettoyage |
| `..._2_2_modele_simple_122022.ipynb` | Word2Vec et régression logistique |
| `..._2_3_modele_avance_embedding_122022.ipynb` | Réseau avec couche d'embedding |
| `..._2_4_modele_avance_LSTM_122022.ipynb` | LSTM bidirectionnel |
| `..._2_5_modele_BERT_122022.ipynb` | BERT |
| `..._2_6_comparaison_modeles_122022.ipynb` | Comparaison sur le jeu de test |
| `..._3_Article de blog_122022.pdf` | Article de blog qui présente la démarche |

---

Projet réalisé de 10/2022 à 12/2022 dans le cadre du parcours d'OpenClassrooms, mis à jour en 11/2026 · [Khadija ER-REGYBY](https://github.com/khadijaerregyby)
