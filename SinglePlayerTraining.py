"""
Single Player Training Scritraining_env = make_vec_env(
    "supertuxkart/flattened-v0", 
    n_envs=4,  # Réduit pour éviter les problèmes de mémoire
    env_kwargs={
        "render_mode": None,  # Pas de rendu pendant l'entraînement
        "track": "hacienda"   # Piste par défaut
    }
)perTuxKart
==============================================

Ce script entraîne un agent PPO (Proximal Policy Optimization) sur l'environnement 
SuperTuxKart de Gymnasium. Il utilise Stable Baselines3 pour l'implémentation.

Auteur: [Votre nom]
Date: Août 2025
"""

# =============================================================================
# IMPORTATIONS
# =============================================================================

import re
import gymnasium as gym
import pystk2_gymnasium as pystk_gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import os

# =============================================================================
# CONFIGURATION DE L'ENVIRONNEMENT
# =============================================================================

# Création d'un environnement vectorisé pour un entraînement parallèle
# SuperTuxKart : Environnement de course avec voitures
# n_envs=4 : Réduire le nombre d'environnements pour SuperTuxKart (plus lourd)
training_env = gym.make(
    "supertuxkart/flattened_discrete-v0", 
    render_mode=None,
    track="hacienda"
)

# =============================================================================
# CONFIGURATION DU MODÈLE PPO
# =============================================================================

# Initialisation du modèle PPO avec une politique MultiInput (pour espaces d'observation dict)
model = PPO(
    policy="MultiInputPolicy",             # Type de politique : réseau de neurones pour espaces d'observation dict
    env=training_env,                      # Environnement d'entraînement
    learning_rate=3e-4,                    # Taux d'apprentissage (3e-4 est une valeur standard)
    n_steps=2048,                          # Nombre d'étapes par mise à jour (batch de trajectoires)
    batch_size=64,                         # Taille du mini-batch pour la mise à jour
    n_epochs=8,                            # Nombre d'époques d'entraînement par mise à jour
    clip_range=0.2,                        # Plage de clipping pour PPO (valeur standard)
    gamma=0.995,                           # Facteur de discount légèrement plus élevé
    gae_lambda=0.95,                       # Paramètre lambda pour Generalized Advantage Estimation
    ent_coef=0.01,                         # Coefficient d'entropie (encourage l'exploration)
    vf_coef=0.5,                           # Coefficient de la fonction de valeur
    max_grad_norm=0.5,                     # Normalisation du gradient
    verbose=1                              # Niveau de verbosité (1 = progression de l'entraînement)
)

# =============================================================================
# ENTRAÎNEMENT DU MODÈLE
# =============================================================================

# Définition du nom du modèle sauvegardé
model_name = "ppo_stk"

# Vérifier si un modèle entraîné existe déjà
if os.path.exists(f"{model_name}.zip"):
    print(f"Modèle {model_name} trouvé. Chargement du modèle existant...")
    model = PPO.load(model_name)
    print("Modèle chargé avec succès !")
else:
    print("Aucun modèle trouvé. Début de l'entraînement PPO sur SuperTuxKart...")
    model.learn(total_timesteps=int(1e6))

    # Sauvegarde du modèle entraîné
    print(f"Sauvegarde du modèle sous le nom : {model_name}")
    model.save(model_name)
    print("Entraînement terminé avec succès !")

# =============================================================================
# ÉVALUATION DU MODÈLE
# =============================================================================

print("Évaluation du modèle...")
eval_env = Monitor(
    gym.make("supertuxkart/flattened_discrete-v0", 
             render_mode=None,
             track="hacienda"), 
    filename=None
)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
print(f"Récompense moyenne sur 5 épisodes : {mean_reward:.2f} ± {std_reward:.2f}")

# Test avec rendu visuel (optionnel) - seulement si le modèle est prometteur
if mean_reward > 0:  # Pour STK, un score positif est déjà bon
    print("Test visuel du modèle...")
    eval_env_render = Monitor(
        gym.make("supertuxkart/flattened_discrete-v0", 
                 render_mode="human",
                 track="hacienda"), 
        filename=None
    )
    evaluate_policy(model, eval_env_render, n_eval_episodes=2, deterministic=True)
