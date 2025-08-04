import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import random

# =============================================================================
# RÉSEAU DE NEURONES POUR DQN (Deep Q-Network)
# =============================================================================
class DQN(nn.Module):
    """
    Réseau de neurones pour approximer la fonction Q(s,a).
    Architecture simple: 2 couches cachées de 64 neurones chacune.
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Couche d'entrée: observation_space -> 64 neurones
        self.fc1 = nn.Linear(input_dim, 64)
        # Couche cachée: 64 -> 64 neurones
        self.fc2 = nn.Linear(64, 64)
        # Couche de sortie: 64 -> nombre d'actions possibles
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Propagation avant du réseau.
        
        Args:
            x: État de l'environnement (tensor)
            
        Returns:
            Q-values pour chaque action possible
        """
        x = F.relu(self.fc1(x))  # Activation ReLU pour la première couche
        x = F.relu(self.fc2(x))  # Activation ReLU pour la deuxième couche
        return self.out(x)       # Sortie linéaire (Q-values)
# =============================================================================
# BUFFER DE REPLAY D'EXPÉRIENCE
# =============================================================================
class ReplayBuffer:
    """
    Stocke les transitions (s, a, r, s', done) pour l'apprentissage par replay.
    Utilise une deque avec capacité limitée pour un accès efficace.
    """
    def __init__(self, capacity):
        """
        Initialise le buffer avec une capacité maximale.
        
        Args:
            capacity: Nombre maximum de transitions à stocker
        """
        self.capacity = capacity
        # deque avec maxlen supprime automatiquement les anciens éléments
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """
        Ajoute une nouvelle transition au buffer.
        
        Args:
            transition: Tuple (state, action, reward, next_state, done)
        """
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        """
        Échantillonne aléatoirement un batch de transitions.
        
        Args:
            batch_size: Nombre de transitions à échantillonner
            
        Returns:
            Liste de transitions sélectionnées aléatoirement
        """
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        """Retourne le nombre de transitions actuellement dans le buffer."""
        return len(self.buffer)

# =============================================================================
# AGENT DQN POUR LUNAR LANDER
# =============================================================================
class LunarLanderDQN:
    """
    Agent Deep Q-Network pour résoudre l'environnement LunarLander-v3.
    Implémente l'algorithme DQN avec replay d'expérience et réseau cible.
    """
    
    # =========================================================================
    # HYPERPARAMÈTRES DE L'AGENT
    # =========================================================================
    learning_rate = 0.001           # Taux d'apprentissage pour l'optimiseur
    discount_factor = 0.99          # Facteur de remise gamma pour les récompenses futures
    batch_size = 64                 # Taille du batch pour l'apprentissage
    network_sync_frequency = 10     # Fréquence de mise à jour du réseau cible
    replay_buffer_capacity = 1000   # Capacité du buffer de replay
    exploration_start = 1.0         # Epsilon initial (exploration maximale)
    exploration_end = 0.01          # Epsilon final (exploration minimale)
    exploration_decay = 200         # Vitesse de décroissance d'epsilon

    # =========================================================================
    # COMPOSANTS DU RÉSEAU DE NEURONES
    # =========================================================================
    loss_function = nn.MSELoss()    # Fonction de perte Mean Squared Error

    # =========================================================================
    # INITIALISATION DE L'AGENT
    # =========================================================================
    def __init__(self):
        """
        Initialise l'agent DQN avec l'environnement et tous les composants nécessaires.
        """
        # Configuration de l'environnement LunarLander avec paramètres réalistes
        self.env = gym.make("LunarLander-v3",
                             continuous=False,        # Actions discrètes
                             gravity=-9.81,          # Gravité terrestre
                             enable_wind=True,       # Vent activé pour plus de difficulté
                             wind_power=15.0,        # Puissance du vent
                             turbulence_power=1.5,   # Turbulences
                             render_mode=None)       # Pas d'affichage pendant l'entraînement
        
        # Dimensions des espaces d'état et d'action
        self.input_dim = self.env.observation_space.shape[0]  # 8 observations
        self.output_dim = self.env.action_space.n             # 4 actions possibles
        
        # Création des réseaux de neurones
        self.policy_net = DQN(self.input_dim, self.output_dim)  # Réseau principal
        self.target_net = DQN(self.input_dim, self.output_dim)  # Réseau cible
        
        # Initialisation du réseau cible avec les poids du réseau principal
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Configuration de l'optimiseur
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialisation du buffer de replay et compteur d'étapes
        self.replay_buffer = ReplayBuffer(self.replay_buffer_capacity)
        self.steps_done = 0
        
        # Liste pour stocker les récompenses de chaque épisode
        self.episode_rewards = []
        
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
                # SÉLECTION D'ACTION (Epsilon-Greedy)
                # =====================================================
                # Calcul d'epsilon décroissant pour l'exploration
                epsilon = self.exploration_end + (self.exploration_start - self.exploration_end) * \
                          np.exp(-1. * self.steps_done / self.exploration_decay)
                
                if np.random.rand() < epsilon:
                    # EXPLORATION: Action aléatoire
                    action = self.env.action_space.sample()
                else:
                    # EXPLOITATION: Meilleure action selon le réseau actuel
                    state_tensor = torch.FloatTensor(observation).unsqueeze(0)
                    with torch.no_grad():
                        action = self.policy_net(state_tensor).argmax().item()
                
                # =====================================================
                # EXÉCUTION DE L'ACTION DANS L'ENVIRONNEMENT
                # =====================================================
                new_observation, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                self.steps_done += 1
                step_count += 1

                # =====================================================
                # STOCKAGE DE LA TRANSITION DANS LE BUFFER DE REPLAY
                # =====================================================
                # Convertir les observations en tenseurs
                current_state_tensor = torch.FloatTensor(observation).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(new_observation).unsqueeze(0)
                
                transition = (current_state_tensor, action, reward, next_state_tensor, done)
                self.replay_buffer.push(transition)

                # =====================================================
                # APPRENTISSAGE PAR BATCH (Experience Replay)
                # =====================================================
                if len(self.replay_buffer) >= self.batch_size:
                    # Échantillonner un batch de transitions du buffer
                    transitions = self.replay_buffer.sample(self.batch_size)
                    # Optimiser le réseau avec ce batch
                    self._optimize(transitions, policy_net=self.policy_net, target_net=self.target_net)

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
            
            # Mise à jour de la barre de progression avec les métriques
            current_epsilon = self.exploration_end + (self.exploration_start - self.exploration_end) * \
                             np.exp(-1. * self.steps_done / self.exploration_decay)
            
            # Calculer la récompense moyenne des 100 derniers épisodes
            recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            avg_reward = np.mean(recent_rewards)
            
            episode_pbar.set_postfix({
                'Récompense': f'{total_reward:.1f}',
                'Moy100': f'{avg_reward:.1f}',
                'Epsilon': f'{current_epsilon:.3f}',
                'Steps': step_count
            })
            
            # Ne pas fermer l'environnement à chaque épisode pendant l'entraînement

        # Sauvegarde du modèle entraîné
        torch.save(self.policy_net.state_dict(), "lunar_lander_dqn.pth")
        print("\nModèle sauvegardé dans 'lunar_lander_dqn.pth'")
        
        # Fermer l'environnement d'entraînement à la fin
        self.env.close()

        # =====================================================
        # VISUALISATION DES RÉSULTATS D'ENTRAÎNEMENT
        # =====================================================
        self._plot_training_results()

    def _plot_training_results(self):
        """
        Affiche les graphiques de progression de l'entraînement.
        """
        if len(self.episode_rewards) == 0:
            print("Aucune donnée de récompense à afficher")
            return
            
        plt.figure(figsize=(15, 5))

        # Graphique 1: Récompenses par épisode
        plt.subplot(1, 3, 1)
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
        plt.subplot(1, 3, 2)
        plt.hist(self.episode_rewards, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Récompense')
        plt.ylabel('Fréquence')
        plt.title('Distribution des Récompenses')
        plt.grid(True, alpha=0.3)

        # Graphique 3: Progression de l'apprentissage (derniers 20% des épisodes)
        plt.subplot(1, 3, 3)
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
        
        # Performance des 100 derniers épisodes
        if len(self.episode_rewards) >= 100:
            last_100 = self.episode_rewards[-100:]
            print(f"\nPerformance des 100 derniers épisodes:")
            print(f"Récompense moyenne: {np.mean(last_100):.2f}")
            print(f"Pourcentage de succès (>200): {sum(1 for r in last_100 if r > 200)/100*100:.1f}%")

    # =========================================================================
    # OPTIMISATION DU RÉSEAU (Algorithme DQN)
    # =========================================================================
    def _optimize(self, transitions, policy_net, target_net):
        """
        Optimise le réseau de neurones en utilisant un batch de transitions.
        Implémente l'équation de Bellman pour calculer les Q-values cibles.
        
        Args:
            transitions: Batch de transitions (state, action, reward, next_state, done)
            policy_net: Réseau principal à optimiser
            target_net: Réseau cible pour calculer les Q-values stables
        """
        current_q_list = []  # Liste des Q-values actuelles
        target_q_list = []   # Liste des Q-values cibles

        # Traitement de chaque transition du batch
        for state, action, reward, next_state, done in transitions:

            if done:
                # Épisode terminé: Q-value cible = récompense immédiate seulement
                target_q_value = reward
            else:
                # Épisode en cours: Appliquer l'équation de Bellman
                # Q(s,a) = reward + γ * max(Q(s',a'))
                with torch.no_grad():
                    next_q_values = target_net(next_state)
                    # Q-value cible selon l'équation de Bellman
                    target_q_value = reward + self.discount_factor * next_q_values.max().item()

            # Extraction de la Q-value actuelle pour l'action prise
            current_q_value = policy_net(state)[0][action]
            current_q_list.append(current_q_value)
            
            # Stockage de la Q-value cible
            target_q_list.append(torch.FloatTensor([target_q_value]))
        
        # =====================================================
        # CALCUL DE LA PERTE ET RÉTROPROPAGATION
        # =====================================================
        # Conversion des listes en tenseurs pour le calcul de la perte
        current_q_tensor = torch.stack(current_q_list)
        target_q_tensor = torch.stack(target_q_list).squeeze()
        
        # Calcul de l'erreur quadratique moyenne entre Q-values actuelles et cibles
        loss = self.loss_function(current_q_tensor, target_q_tensor)

        # Mise à zéro des gradients précédents
        self.optimizer.zero_grad()
        # Calcul des nouveaux gradients
        loss.backward()
        # Mise à jour des poids du réseau
        self.optimizer.step()

    # =========================================================================
    # TEST DE L'AGENT ENTRAÎNÉ
    # =========================================================================
    def test(self, num_episodes=10):
        """
        Test l'agent entraîné sur plusieurs épisodes sans exploration.
        
        Args:
            num_episodes: Nombre d'épisodes de test
        """
        # Création d'un nouvel environnement pour les tests
        test_env = gym.make("LunarLander-v3",
                            continuous=False,
                            gravity=-9.81,
                            enable_wind=True,
                            wind_power=15.0,
                            turbulence_power=1.5,
                            render_mode="human")
        
        # Chargement du modèle entraîné
        self.policy_net.load_state_dict(torch.load("lunar_lander_dqn.pth"))
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
                    state_tensor = torch.FloatTensor(observation).unsqueeze(0)
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
if __name__ == '__main__':
    print("Démarrage de l'entraînement DQN pour LunarLander-v3")
    print("="*60)
    
    # Création et entraînement de l'agent
    lunar_lander = LunarLanderDQN()
    lunar_lander.train(num_episodes=1000)
    
    print("\nEntraînement terminé! Démarrage des tests...")
    print("="*60)
    
    # Test de l'agent entraîné
    lunar_lander.test(num_episodes=10)
