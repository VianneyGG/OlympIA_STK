"""
Single Player Training Script for Lunar Lander
==============================================

Ce script entraîne un agent PPO (Proximal Policy Optimization) sur l'environnement 
LunarLander-v3 de Gymnasium. Il utilise Stable Baselines3 pour l'implémentation.

Auteur: [Votre nom]
Date: Août 2025
"""

# =============================================================================
# IMPORTATIONS
# =============================================================================

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from tqdm import tqdm
import os

# =============================================================================
# CONFIGURATION DE L'ENVIRONNEMENT
# =============================================================================

# Création d'un environnement vectorisé pour un entraînement parallèle
# LunarLander-v3 : Environnement classique de contrôle de fusée lunaire
# n_envs=16 : Utilisation de 16 environnements parallèles pour accélérer l'entraînement
training_env = make_vec_env("LunarLander-v3", n_envs=16)

# =============================================================================
# CONFIGURATION DU MODÈLE PPO
# =============================================================================

# Initialisation du modèle PPO avec une politique MLP (Multi-Layer Perceptron)
model = PPO(
    policy="MlpPolicy",                    # Type de politique : réseau de neurones simple
    env=training_env,                      # Environnement d'entraînement
    learning_rate=3e-4,                    # Taux d'apprentissage (3e-4 est une valeur standard)
    n_steps=1024,                          # Nombre d'étapes par mise à jour (batch de trajectoires)
    batch_size=64,                         # Taille du mini-batch pour la mise à jour
    n_epochs=5,                            # Nombre d'époques d'entraînement par mise à jour
    clip_range=0.2,                        # Plage de clipping pour PPO (valeur standard)
    gamma=0.99,                            # Facteur de discount pour les récompenses futures
    gae_lambda=0.95,                       # Paramètre lambda pour Generalized Advantage Estimation
    ent_coef=0.01,                         # Coefficient d'entropie (encourage l'exploration)
    verbose=1                              # Niveau de verbosité (1 = progression de l'entraînement)
)

# =============================================================================
# ENTRAÎNEMENT DU MODÈLE
# =============================================================================

# Définition du nom du modèle sauvegardé
model_name = "ppo_lunarlander"

# Phase d'entraînement principal
# total_timesteps=500000 : Nombre total d'étapes d'interaction avec l'environnement
# Cela correspond à environ 500000/16 ≈ 31250 épisodes (avec 16 envs parallèles)
print("Début de l'entraînement PPO sur LunarLander-v3...")
model.learn(total_timesteps=500000)

# Sauvegarde du modèle entraîné
print(f"Sauvegarde du modèle sous le nom : {model_name}")
model.save(model_name)

print("Entraînement terminé avec succès !")


