#!/usr/bin/env python3
"""
Script de lancement rapide pour l'entraînement DQN amélioré.
Usage: python start_training.py [--episodes N] [--gpu] [--batch-size N]
"""

import argparse
import torch
from SinglePlayerTraining import EnhancedLunarLanderDQN

def main():
    parser = argparse.ArgumentParser(description='Entraînement DQN Enhanced avec PER, Double DQN, Dueling DQN et ICM')
    parser.add_argument('--episodes', type=int, default=1000, help='Nombre d\'épisodes d\'entraînement (défaut: 1000)')
    parser.add_argument('--test-episodes', type=int, default=10, help='Nombre d\'épisodes de test (défaut: 10)')
    parser.add_argument('--no-render', action='store_true', help='Désactiver l\'affichage pendant les tests')
    parser.add_argument('--force-cpu', action='store_true', help='Forcer l\'utilisation du CPU même si GPU disponible')
    
    args = parser.parse_args()
    
    # Configuration GPU
    if args.force_cpu:
        print("Forçage CPU demandé")
        # Note: Dans notre implémentation, le device est géré automatiquement
    
    print("="*80)
    print("🚀 ENTRAÎNEMENT DQN ENHANCED - LunarLander-v3")
    print("="*80)
    print(f"📊 Épisodes d'entraînement: {args.episodes}")
    print(f"🧪 Épisodes de test: {args.test_episodes}")
    print(f"💻 Device: {'CPU forcé' if args.force_cpu else 'Auto-détection'}")
    print(f"🖥️  Rendu test: {'Désactivé' if args.no_render else 'Activé'}")
    
    print("\n🔧 AMÉLIORATIONS ACTIVÉES:")
    print("   ✓ Prioritized Experience Replay (PER)")
    print("   ✓ Double DQN (réduction surestimation)")
    print("   ✓ Dueling DQN (séparation V(s) et A(s,a))")
    print("   ✓ Intrinsic Curiosity Module (ICM)")
    print("="*80)
    
    try:
        # Création de l'agent
        print("🏗️  Initialisation de l'agent...")
        agent = EnhancedLunarLanderDQN()
        
        # Entraînement
        print(f"🎯 Début de l'entraînement sur {args.episodes} épisodes...")
        agent.train(num_episodes=args.episodes)
        
        # Test
        print(f"🧪 Test de l'agent sur {args.test_episodes} épisodes...")
        agent.test(num_episodes=args.test_episodes, render=not args.no_render)
        
        print("\n🎉 Entraînement et test terminés avec succès!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu par l'utilisateur!")
        if 'agent' in locals():
            torch.save(agent.policy_net.state_dict(), "enhanced_dqn_interrupted.pth")
            print("💾 Modèle sauvegardé: enhanced_dqn_interrupted.pth")
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Nettoyage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Mémoire GPU nettoyée")

if __name__ == "__main__":
    main()
