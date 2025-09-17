# OlympIA_STK
SuperTux kart pour OlympIA

## Entraînement PPO (actions continues) sur Windows

Prérequis: Windows 10/11, conda, GPU optionnel. L'environnement Python cible est 3.11.

### Créer l'environnement conda

```powershell
conda env create -f environment.yml ; conda activate OlympIA_STK_py311
```

### Lancer l'entraînement

```powershell
python src/train_continuous.py --config configs\ppo_continuous.yaml
```

Le script configure automatiquement les DLLs Windows (`Library/bin` de conda et `C:\Program Files\SuperTuxKart*` si présent). Les logs TensorBoard sont écrits dans `tensorboard/ppo_stk_continuous/<run_name>`.

### Jouer avec le meilleur modèle

```powershell
python src/eval_play.py --model q-supertuxkart\continuous\best_model.zip --stats q-supertuxkart\continuous\vecnormalize.pkl
```

Si le fichier `vecnormalize.pkl` n'existe pas encore, omettez `--stats`.

### Ouvrir TensorBoard

```powershell
tensorboard --logdir tensorboard\ppo_stk_continuous
```

## Structure

- `src/env_utils.py`: gestion DLL Windows, patch `tarfile`, fabrique d'environnements.
- `src/callbacks.py`: callbacks d'évaluation, checkpoints, et logger des composantes de récompense.
- `src/train_continuous.py`: script d'entraînement principal, config YAML.
- `src/eval_play.py`: chargement modèle et exécution avec `render_mode="human"`.
- `configs/ppo_continuous.yaml`: hyperparamètres et chemins.
- `tests/test_env_boot.py`: test de boot/reset/step/close (sans rendu).
- `tools/analyze_obs_continuous.py`: analyse corrélations obs/actions/progression.

Les scripts existants (`SinglePlayerTraining(...)`, `test.py`) restent utilisables.

