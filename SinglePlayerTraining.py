import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque, namedtuple
from tqdm import tqdm
import random
import heapq
import math
import csv
import os

# =============================================================================
# RÉSEAU DE NEURONES DUELING DQN (Deep Q-Network)
# =============================================================================
class DuelingDQN(nn.Module):
    """
    Réseau de neurones Dueling DQN qui sépare l'estimation de la valeur d'état V(s)
    et des avantages d'action A(s,a) pour améliorer l'apprentissage.
    Architecture: V(s) + A(s,a) - mean(A(s,:))
    """
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # Couches partagées (feature extraction)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Branche pour l'estimation de la valeur d'état V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Une seule valeur scalaire pour V(s)
        )
        
        # Branche pour l'estimation des avantages d'action A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Une valeur par action pour A(s,a)
        )

    def forward(self, x):
        """
        Propagation avant du réseau Dueling DQN.
        
        Args:
            x: État de l'environnement (tensor)
            
        Returns:
            Q-values calculées comme Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        """
        # Extraction des caractéristiques communes
        features = self.feature_layer(x)
        
        # Calcul de la valeur d'état V(s)
        value = self.value_stream(features)
        
        # Calcul des avantages d'action A(s,a)
        advantage = self.advantage_stream(features)
        
        # Formule Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        # La soustraction de la moyenne assure l'identifiabilité
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

# =============================================================================
# INTRINSIC CURIOSITY MODULE (ICM)
# =============================================================================
class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module for exploration.
    Consists of:
    1. Feature encoder: Extracts meaningful features from states
    2. Inverse model: Predicts action given current and next state features
    3. Forward model: Predicts next state features given current features and action
    
    Intrinsic reward = prediction error of forward model
    """
    def __init__(self, input_dim, action_dim, feature_dim=64, hidden_dim=128):
        super(ICMModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # Feature encoder: state -> feature representation
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Inverse model: phi(s_t), phi(s_{t+1}) -> a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Forward model: phi(s_t), a_t -> phi(s_{t+1})
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def encode_state(self, state):
        """Encode state into feature representation."""
        return self.feature_encoder(state)
    
    def predict_action(self, state_features, next_state_features):
        """Inverse model: predict action given state transitions."""
        combined_features = torch.cat([state_features, next_state_features], dim=1)
        return self.inverse_model(combined_features)
    
    def predict_next_state(self, state_features, action):
        """Forward model: predict next state features given current state and action."""
        # Convert action to one-hot encoding
        action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        combined_input = torch.cat([state_features, action_onehot], dim=1)
        return self.forward_model(combined_input)
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """
        Compute intrinsic reward based on forward model prediction error.
        Higher prediction error = higher curiosity = higher intrinsic reward
        """
        with torch.no_grad():
            # Encode states
            state_features = self.encode_state(state)
            next_state_features = self.encode_state(next_state)
            
            # Predict next state features
            predicted_next_features = self.predict_next_state(state_features, action)
            
            # Compute prediction error (intrinsic reward)
            prediction_error = F.mse_loss(predicted_next_features, next_state_features, reduction='none')
            intrinsic_reward = prediction_error.mean(dim=1)
            
        return intrinsic_reward
    
    def compute_losses(self, state, action, next_state):
        """
        Compute ICM losses for training.
        Returns: forward_loss, inverse_loss
        """
        # Encode states
        state_features = self.encode_state(state)
        next_state_features = self.encode_state(next_state)
        
        # Forward model loss
        predicted_next_features = self.predict_next_state(state_features, action)
        forward_loss = F.mse_loss(predicted_next_features, next_state_features.detach())
        
        # Inverse model loss
        predicted_action = self.predict_action(state_features.detach(), next_state_features.detach())
        inverse_loss = F.cross_entropy(predicted_action, action.long())
        
        return forward_loss, inverse_loss
# =============================================================================
# BUFFER DE REPLAY D'EXPÉRIENCE AVEC PRIORITÉ (PER)
# =============================================================================
# Structure pour stocker les transitions avec priorité
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """
    Buffer de replay avec priorité (Prioritized Experience Replay).
    Les transitions avec des erreurs TD importantes sont échantillonnées plus fréquemment.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Initialise le buffer PER.
        
        Args:
            capacity: Nombre maximum de transitions à stocker
            alpha: Degré de priorisation (0=uniforme, 1=priorité pure)
            beta: Degré de correction d'importance sampling (0=pas de correction, 1=correction complète)
            beta_increment: Incrément de beta à chaque échantillonnage
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Petite constante pour éviter les priorités nulles
        
        # Stockage des transitions et priorités
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, transition, td_error=None):
        """
        Ajoute une transition avec sa priorité.
        
        Args:
            transition: Tuple (state, action, reward, next_state, done)
            td_error: Erreur TD pour calculer la priorité (si None, priorité max)
        """
        # Calcul de la priorité basée sur l'erreur TD
        if td_error is None:
            # Nouvelle transition: priorité maximale pour être sûr qu'elle soit échantillonnée
            priority = self.priorities.max() if self.size > 0 else 1.0
        else:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        
        # Stockage de la transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        # Mise à jour de la priorité
        self.priorities[self.position] = priority
        
        # Mise à jour des indices
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Échantillonne un batch selon les priorités.
        
        Args:
            batch_size: Taille du batch
            
        Returns:
            batch: Liste des transitions
            indices: Indices des transitions échantillonnées
            weights: Poids d'importance sampling
        """
        # Normalisation des priorités
        priorities = self.priorities[:self.size]
        probabilities = priorities / priorities.sum()
        
        # Échantillonnage selon les probabilités
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calcul des poids d'importance sampling
        sampling_probs = probabilities[indices]
        weights = (self.size * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalisation
        
        # Incrément de beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extraction des transitions
        batch = [self.buffer[idx] for idx in indices]
        
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        """
        Met à jour les priorités des transitions échantillonnées.
        
        Args:
            indices: Indices des transitions
            td_errors: Nouvelles erreurs TD
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return self.size

# =============================================================================
# AGENT DQN AMÉLIORÉ (PER + Double DQN + Dueling DQN)
# =============================================================================
class EnhancedLunarLanderDQN:
    """
    Agent Deep Q-Network amélioré pour résoudre l'environnement LunarLander-v3.
    Implémente les améliorations suivantes:
    - Prioritized Experience Replay (PER)
    - Double DQN
    - Dueling DQN
    - Intrinsic Curiosity Module (ICM) pour l'exploration
    """
    
    # =========================================================================
    # HYPERPARAMÈTRES DE L'AGENT
    # =========================================================================
    learning_rate = 0.0005          # Taux d'apprentissage réduit pour plus de stabilité
    discount_factor = 0.99          # Facteur de remise gamma pour les récompenses futures
    batch_size = 128                # Taille du batch augmentée pour GPU
    network_sync_frequency = 500    # Fréquence de mise à jour du réseau cible
    replay_buffer_capacity = 100000 # Capacité du buffer de replay augmentée
    
    # Paramètres ICM (remplace epsilon-greedy)
    icm_learning_rate = 0.0005      # Taux d'apprentissage pour ICM
    intrinsic_reward_scale = 0.01   # Échelle des récompenses intrinsèques
    forward_loss_weight = 0.2       # Poids du forward model dans ICM
    inverse_loss_weight = 0.8       # Poids de l'inverse model dans ICM
    
    # Hyperparamètres PER
    per_alpha = 0.6                 # Degré de priorisation
    per_beta_start = 0.4            # Beta initial pour importance sampling
    per_beta_increment = 0.0001     # Incrément de beta

    # =========================================================================
    # COMPOSANTS DU RÉSEAU DE NEURONES
    # =========================================================================
    loss_function = nn.MSELoss()    # Fonction de perte Mean Squared Error

    # =========================================================================
    # INITIALISATION DE L'AGENT
    # =========================================================================
    def __init__(self):
        """
        Initialise l'agent DQN amélioré avec l'environnement et tous les composants nécessaires.
        """
        # Configuration du device (GPU si disponible, sinon CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU détecté: {torch.cuda.get_device_name()}")
            print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Configuration de l'environnement LunarLander avec paramètres réalistes
        self.env = gym.make("LunarLander-v3",
                             continuous=False,        # Actions discrètes
                             gravity=-9.81,          # Gravité terrestre
                             enable_wind=True,       # Vent activé pour plus de difficulté
                             wind_power=15.0,        # Puissance du vent
                             turbulence_power=1.5,   # Turbulences
                             render_mode=None)       # Pas d'affichage pendant l'entraînement (GPU friendly)

        # Dimensions des espaces d'état et d'action
        self.input_dim = self.env.observation_space.shape[0]  # 8 observations
        self.output_dim = self.env.action_space.n             # 4 actions possibles
        
        # Création des réseaux de neurones Dueling DQN sur GPU
        self.policy_net = DuelingDQN(self.input_dim, self.output_dim).to(self.device)  # Réseau principal
        self.target_net = DuelingDQN(self.input_dim, self.output_dim).to(self.device)  # Réseau cible
        
        # Initialisation du réseau cible avec les poids du réseau principal
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Configuration de l'optimiseur pour DQN
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialisation du module ICM pour l'exploration intrinsèque sur GPU
        self.icm = ICMModule(self.input_dim, self.output_dim).to(self.device)
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=self.icm_learning_rate)
        
        # Initialisation du buffer de replay avec priorité (PER)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.replay_buffer_capacity,
            alpha=self.per_alpha,
            beta=self.per_beta_start,
            beta_increment=self.per_beta_increment
        )
        self.steps_done = 0
        
        # Liste pour stocker les récompenses de chaque épisode
        self.episode_rewards = []
        self.intrinsic_rewards = []  # Track intrinsic rewards for analysis
        
    # =========================================================================
    # BOUCLE D'ENTRAÎNEMENT PRINCIPAL
    # =========================================================================
    def train(self, num_episodes=1000, render=False):
        """
        Entraîne l'agent DQN sur un nombre donné d'épisodes.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            render: Si True, affiche l'environnement visuellement (ralentit l'entraînement)
        """
        # Préparer le fichier CSV pour la journalisation
        csv_file = "training_log.csv"
        write_header = not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0
        csv_f = open(csv_file, mode="a", newline="")
        csv_writer = csv.writer(csv_f)
        if write_header:
            csv_writer.writerow(["episode", "step", "total_reward", "extrinsic_reward", "intrinsic_reward", "action"])
        # Reconfigurer l'environnement avec ou sans rendu
        if render and self.env.spec.kwargs.get('render_mode') != 'human':
            self.env.close()
            self.env = gym.make("LunarLander-v3",
                               continuous=False,
                               gravity=-9.81,
                               enable_wind=True,
                               wind_power=15.0,
                               turbulence_power=1.5,
                               render_mode="human")
            
        # Barre de progression pour les épisodes
        episode_pbar = tqdm(range(num_episodes), desc="Entraînement DQN", unit="épisode")
        
        for episode in episode_pbar:
            # Réinitialiser l'environnement pour un nouvel épisode
            observation, _ = self.env.reset()
            total_reward = 0
            done = False
            truncated = False
            step_count = 0

            # Boucle principale de l'épisode
            while not done and not truncated:
                # =====================================================
                # SÉLECTION D'ACTION (ICM-based exploration)
                # =====================================================
                # Utilisation du réseau DQN pour sélectionner l'action
                # L'exploration est gérée par les récompenses intrinsèques ICM
                state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                    action = q_values.argmax().item()
                
                # =====================================================
                # EXÉCUTION DE L'ACTION DANS L'ENVIRONNEMENT
                # =====================================================
                new_observation, reward, done, truncated, _ = self.env.step(action)
                
                # =====================================================
                # CALCUL DE LA RÉCOMPENSE INTRINSÈQUE (ICM)
                # =====================================================
                # Calcul de la récompense intrinsèque basée sur la curiosité
                state_tensor_icm = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                next_state_tensor_icm = torch.FloatTensor(new_observation).unsqueeze(0).to(self.device)
                action_tensor = torch.tensor([action]).to(self.device)
                
                intrinsic_reward = self.icm.compute_intrinsic_reward(
                    state_tensor_icm, action_tensor, next_state_tensor_icm
                ).item()
                
                # Combinaison des récompenses extrinsèque et intrinsèque
                total_step_reward = reward + self.intrinsic_reward_scale * intrinsic_reward
                total_reward += total_step_reward
                # Stockage pour analyse
                self.intrinsic_rewards.append(intrinsic_reward)
                # Journalisation CSV de l'étape
                csv_writer.writerow([
                    episode,
                    step_count,
                    total_reward,
                    reward,
                    intrinsic_reward,
                    action
                ])
                self.steps_done += 1
                step_count += 1

                # =====================================================
                # STOCKAGE DE LA TRANSITION DANS LE BUFFER DE REPLAY
                # =====================================================
                # Créer les tenseurs pour stockage (CPU pour le buffer)
                state_tensor_cpu = torch.FloatTensor(observation).unsqueeze(0)
                next_state_tensor_cpu = torch.FloatTensor(new_observation).unsqueeze(0)
                
                transition = Transition(state_tensor_cpu, action, total_step_reward, next_state_tensor_cpu, done)
                # Stocker sans erreur TD pour l'instant (sera calculée lors de l'optimisation)
                self.replay_buffer.push(transition)

                # =====================================================
                # APPRENTISSAGE PAR BATCH (Experience Replay + ICM)
                # =====================================================
                if len(self.replay_buffer) >= self.batch_size:
                    # Échantillonner un batch de transitions du buffer (avec PER)
                    transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
                    
                    # Optimiser le réseau DQN avec ce batch
                    td_errors = self._optimize(transitions, weights, policy_net=self.policy_net, target_net=self.target_net)
                    
                    # Optimiser le module ICM
                    self._optimize_icm(transitions)
                    
                    # Mise à jour des priorités avec les nouvelles erreurs TD
                    self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

                    # Mise à jour périodique du réseau cible
                    if self.steps_done % self.network_sync_frequency == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                # Mise à jour de l'observation pour le prochain step
                observation = new_observation

            # =====================================================
            # FIN D'ÉPISODE - STOCKAGE ET MISE À JOUR
            # =====================================================
            # Stocker la récompense de cet épisode
            self.episode_rewards.append(total_reward)
            
            # Calculer la récompense moyenne des 100 derniers épisodes
            recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            avg_reward = np.mean(recent_rewards)
            
            # Calculer la récompense intrinsèque moyenne de cet épisode
            episode_intrinsic_rewards = self.intrinsic_rewards[-step_count:] if len(self.intrinsic_rewards) >= step_count else self.intrinsic_rewards
            avg_intrinsic_reward = np.mean(episode_intrinsic_rewards) if episode_intrinsic_rewards else 0
            
            # Informations sur l'utilisation GPU
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                gpu_info = f', GPU: {gpu_memory:.1f}GB'
            
            episode_pbar.set_postfix({
                'Récompense': f'{total_reward:.1f}',
                'Moy100': f'{avg_reward:.1f}',
                'ICM_Int': f'{avg_intrinsic_reward:.3f}',
                'Steps': step_count
            })
            
            # Nettoyage mémoire GPU périodique
            if episode % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Ne pas fermer l'environnement à chaque épisode pendant l'entraînement

    # Sauvegarde du modèle entraîné
    torch.save(self.policy_net.state_dict(), "enhanced_lunar_lander_dqn.pth")
    print("\nModèle sauvegardé dans 'enhanced_lunar_lander_dqn.pth'")
    # Fermer l'environnement d'entraînement à la fin
    self.env.close()
    # Fermer le fichier CSV
    csv_f.close()
    # =====================================================
    # VISUALISATION DES RÉSULTATS D'ENTRAÎNEMENT
    # =====================================================
    self._plot_training_results()

    def _plot_training_results(self):
        """
        Affiche les graphiques de progression de l'entraînement avec les récompenses intrinsèques ICM.
        """
        if len(self.episode_rewards) == 0:
            print("Aucune donnée de récompense à afficher")
            return
            
        plt.figure(figsize=(20, 10))

        # Graphique 1: Récompenses par épisode
        plt.subplot(2, 3, 1)
        plt.plot(self.episode_rewards, alpha=0.6, color='lightblue', label='Récompense par épisode')
        
        # Moyenne mobile sur 100 épisodes
        if len(self.episode_rewards) >= 100:
            moving_avg = []
            for i in range(len(self.episode_rewards)):
                start_idx = max(0, i - 99)
                moving_avg.append(np.mean(self.episode_rewards[start_idx:i+1]))
            plt.plot(moving_avg, color='red', linewidth=2, label='Moyenne mobile (100 épisodes)')
        
        plt.xlabel('Épisodes')
        plt.ylabel('Récompense')
        plt.title('Récompenses par Épisode')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Graphique 2: Distribution des récompenses
        plt.subplot(2, 3, 2)
        plt.hist(self.episode_rewards, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Récompense')
        plt.ylabel('Fréquence')
        plt.title('Distribution des Récompenses')
        plt.grid(True, alpha=0.3)

        # Graphique 3: Progression de l'apprentissage (derniers 20% des épisodes)
        plt.subplot(2, 3, 3)
        last_20_percent = int(len(self.episode_rewards) * 0.8)
        recent_rewards = self.episode_rewards[last_20_percent:]
        episodes_recent = range(last_20_percent, len(self.episode_rewards))
        
        plt.plot(episodes_recent, recent_rewards, color='purple', alpha=0.7)
        if len(recent_rewards) > 10:
            # Ligne de tendance pour les derniers épisodes
            z = np.polyfit(episodes_recent, recent_rewards, 1)
            p = np.poly1d(z)
            plt.plot(episodes_recent, p(episodes_recent), color='orange', linewidth=2, label='Tendance')
        
        plt.xlabel('Épisodes')
        plt.ylabel('Récompense')
        plt.title('Progression Récente (20% finaux)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Graphique 4: Récompenses intrinsèques ICM
        if len(self.intrinsic_rewards) > 0:
            plt.subplot(2, 3, 4)
            plt.plot(self.intrinsic_rewards, alpha=0.6, color='coral', label='Récompense intrinsèque')
            
            # Moyenne mobile pour les récompenses intrinsèques
            if len(self.intrinsic_rewards) >= 100:
                intrinsic_moving_avg = []
                for i in range(len(self.intrinsic_rewards)):
                    start_idx = max(0, i - 99)
                    intrinsic_moving_avg.append(np.mean(self.intrinsic_rewards[start_idx:i+1]))
                plt.plot(intrinsic_moving_avg, color='darkred', linewidth=2, label='Moyenne mobile ICM')
            
            plt.xlabel('Steps')
            plt.ylabel('Récompense Intrinsèque')
            plt.title('Évolution de la Curiosité (ICM)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Graphique 5: Distribution des récompenses intrinsèques
        if len(self.intrinsic_rewards) > 0:
            plt.subplot(2, 3, 5)
            plt.hist(self.intrinsic_rewards, bins=50, alpha=0.7, color='coral', edgecolor='black')
            plt.xlabel('Récompense Intrinsèque')
            plt.ylabel('Fréquence')
            plt.title('Distribution de la Curiosité')
            plt.grid(True, alpha=0.3)

        # Graphique 6: Comparaison récompenses extrinsèques vs intrinsèques (moyennées par épisode)
        if len(self.intrinsic_rewards) > 0 and len(self.episode_rewards) > 0:
            plt.subplot(2, 3, 6)
            
            # Calculer les récompenses intrinsèques moyennes par épisode (approximation)
            steps_per_episode = len(self.intrinsic_rewards) // len(self.episode_rewards) if len(self.episode_rewards) > 0 else 1
            episode_intrinsic_rewards = []
            
            for i in range(len(self.episode_rewards)):
                start_idx = i * steps_per_episode
                end_idx = min((i + 1) * steps_per_episode, len(self.intrinsic_rewards))
                if start_idx < len(self.intrinsic_rewards):
                    episode_avg = np.mean(self.intrinsic_rewards[start_idx:end_idx])
                    episode_intrinsic_rewards.append(episode_avg)
            
            if episode_intrinsic_rewards:
                plt.plot(range(len(episode_intrinsic_rewards)), episode_intrinsic_rewards, 
                        color='coral', alpha=0.7, label='Curiosité moyenne/épisode')
                
                # Normaliser les récompenses d'environnement pour comparaison
                normalized_env_rewards = [(r - np.min(self.episode_rewards)) / 
                                        (np.max(self.episode_rewards) - np.min(self.episode_rewards)) 
                                        for r in self.episode_rewards[:len(episode_intrinsic_rewards)]]
                plt.plot(range(len(normalized_env_rewards)), normalized_env_rewards, 
                        color='blue', alpha=0.7, label='Récompense env. (normalisée)')
                
                plt.xlabel('Épisodes')
                plt.ylabel('Valeur')
                plt.title('Curiosité vs Performance')
                plt.legend()
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Statistiques finales
        print(f"\n{'='*60}")
        print("STATISTIQUES FINALES D'ENTRAÎNEMENT")
        print(f"{'='*60}")
        print(f"Nombre total d'épisodes: {len(self.episode_rewards)}")
        print(f"Récompense moyenne: {np.mean(self.episode_rewards):.2f}")
        print(f"Récompense médiane: {np.median(self.episode_rewards):.2f}")
        print(f"Meilleure récompense: {np.max(self.episode_rewards):.2f}")
        print(f"Pire récompense: {np.min(self.episode_rewards):.2f}")
        print(f"Écart-type: {np.std(self.episode_rewards):.2f}")
        
        if len(self.intrinsic_rewards) > 0:
            print(f"\nSTATISTIQUES ICM (CURIOSITÉ):")
            print(f"Récompense intrinsèque moyenne: {np.mean(self.intrinsic_rewards):.4f}")
            print(f"Récompense intrinsèque médiane: {np.median(self.intrinsic_rewards):.4f}")
            print(f"Récompense intrinsèque max: {np.max(self.intrinsic_rewards):.4f}")
            print(f"Écart-type intrinsèque: {np.std(self.intrinsic_rewards):.4f}")
        
        # Performance des 100 derniers épisodes
        if len(self.episode_rewards) >= 100:
            last_100 = self.episode_rewards[-100:]
            print(f"\nPerformance des 100 derniers épisodes:")
            print(f"Récompense moyenne: {np.mean(last_100):.2f}")
            print(f"Pourcentage de succès (>200): {sum(1 for r in last_100 if r > 200)/100*100:.1f}%")

    # =========================================================================
    # OPTIMISATION DU RÉSEAU (Double DQN + PER)
    # =========================================================================
    def _optimize(self, transitions, weights, policy_net, target_net):
        """
        Optimise le réseau de neurones en utilisant Double DQN et PER.
        
        Args:
            transitions: Batch de transitions (state, action, reward, next_state, done)
            weights: Poids d'importance sampling du PER
            policy_net: Réseau principal à optimiser
            target_net: Réseau cible pour calculer les Q-values stables
            
        Returns:
            td_errors: Erreurs TD pour mise à jour des priorités
        """
        # Extraction des composants des transitions et transfert vers GPU
        states = torch.cat([t.state for t in transitions]).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.cat([t.next_state for t in transitions]).to(self.device)
        dones = torch.BoolTensor([t.done for t in transitions]).to(self.device)
        weights = weights.to(self.device)
        
        # Q-values actuelles pour les actions prises
        current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # =====================================================
        # DOUBLE DQN: Sélection d'action avec policy_net, évaluation avec target_net
        # =====================================================
        with torch.no_grad():
            # Sélection des meilleures actions avec le réseau principal
            next_actions = policy_net(next_states).argmax(1)
            # Évaluation de ces actions avec le réseau cible
            next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Calcul des Q-values cibles
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # =====================================================
        # CALCUL DE LA PERTE AVEC IMPORTANCE SAMPLING (PER)
        # =====================================================
        # Erreurs TD pour la mise à jour des priorités
        td_errors = target_q_values - current_q_values
        
        # Perte pondérée par les poids d'importance sampling
        weighted_loss = (weights * (td_errors ** 2)).mean()
        
        # Rétropropagation
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping pour la stabilité
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        return td_errors

    # =========================================================================
    # OPTIMISATION DU MODULE ICM
    # =========================================================================
    def _optimize_icm(self, transitions):
        """
        Optimise le module ICM (Intrinsic Curiosity Module).
        
        Args:
            transitions: Batch de transitions pour l'entraînement ICM
        """
        # Extraction des composants des transitions et transfert vers GPU
        states = torch.cat([t.state for t in transitions]).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        next_states = torch.cat([t.next_state for t in transitions]).to(self.device)
        
        # Calcul des pertes ICM
        forward_loss, inverse_loss = self.icm.compute_losses(states, actions, next_states)
        
        # Perte totale ICM avec pondération
        icm_loss = self.forward_loss_weight * forward_loss + self.inverse_loss_weight * inverse_loss
        
        # Rétropropagation pour ICM
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        
        # Gradient clipping pour la stabilité
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), max_norm=10.0)
        
        self.icm_optimizer.step()

    # =========================================================================
    # TEST DE L'AGENT ENTRAÎNÉ
    # =========================================================================
    def test(self, num_episodes=10, render=True):
        """
        Test l'agent entraîné sur plusieurs épisodes sans exploration.
        
        Args:
            num_episodes: Nombre d'épisodes de test
            render: Si True, affiche l'environnement pendant les tests
        """
        # Création d'un nouvel environnement pour les tests
        test_env = gym.make("LunarLander-v3",
                            continuous=False,
                            gravity=-9.81,
                            enable_wind=True,
                            wind_power=15.0,
                            turbulence_power=1.5,
                            render_mode="human" if render else None)
        
        # Chargement du modèle entraîné
        self.policy_net.load_state_dict(torch.load("enhanced_lunar_lander_dqn.pth", map_location=self.device))
        self.policy_net.eval()  # Mode évaluation (pas d'entraînement)

        print(f"Test de l'agent entraîné sur {num_episodes} épisodes...")
        
        # Barre de progression pour les tests
        test_rewards = []
        test_pbar = tqdm(range(num_episodes), desc="Test de l'agent", unit="épisode")
        
        for episode in test_pbar:
            observation, _ = test_env.reset()
            total_reward = 0
            done = False
            truncated = False
            step_count = 0

            # Boucle de test (pas d'exploration, seulement exploitation)
            while not done and not truncated:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                    # Sélection de la meilleure action selon le réseau entraîné
                    action = self.policy_net(state_tensor).argmax().item()

                # Exécution de l'action dans l'environnement
                observation, reward, done, truncated, _ = test_env.step(action)
                total_reward += reward
                step_count += 1

            test_rewards.append(total_reward)
            test_pbar.set_postfix({
                'Récompense': f'{total_reward:.1f}',
                'Moyenne': f'{np.mean(test_rewards):.1f}',
                'Steps': step_count
            })
            
            test_env.close()

        # Statistiques de test
        print(f"\n{'='*50}")
        print("RÉSULTATS DU TEST")
        print(f"{'='*50}")
        print(f"Récompense moyenne: {np.mean(test_rewards):.2f}")
        print(f"Écart-type: {np.std(test_rewards):.2f}")
        print(f"Meilleure récompense: {np.max(test_rewards):.2f}")
        print(f"Pire récompense: {np.min(test_rewards):.2f}")
        print(f"Taux de succès (>200): {sum(1 for r in test_rewards if r > 200)/len(test_rewards)*100:.1f}%")
        print("Test terminé!")

# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================
# ENTRAÎNEMENT PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    # Configuration pour optimiser l'utilisation GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Nettoyer la mémoire GPU
        torch.backends.cudnn.benchmark = True  # Optimiser les convolutions
        print("Optimisations GPU activées")
    
    print("Démarrage de l'entraînement DQN AMÉLIORÉ pour LunarLander-v3")
    print("Améliorations: PER + Double DQN + Dueling DQN + ICM")
    print("="*70)
    
    try:
        # Création et entraînement de l'agent amélioré
        lunar_lander = EnhancedLunarLanderDQN()
        lunar_lander.train(num_episodes=3000)  # Plus d'épisodes pour le PER
        
        print("\nEntraînement terminé! Démarrage des tests...")
        print("="*70)
        
        # Test de l'agent entraîné (sans rendu pour performance)
        lunar_lander.test(num_episodes=10, render=False)
        
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur!")
        # Sauvegarder le modèle même en cas d'interruption
        if 'lunar_lander' in locals():
            torch.save(lunar_lander.policy_net.state_dict(), "enhanced_lunar_lander_dqn_interrupted.pth")
            print("Modèle sauvegardé dans 'enhanced_lunar_lander_dqn_interrupted.pth'")
    
    except Exception as e:
        print(f"Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Nettoyer la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Mémoire GPU nettoyée")
