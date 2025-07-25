import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Chemin vers le fichier trainer_state.json
path = "/media/hdd/maxime_carlier/checkpoints/vlac/train-vlm-encode-decode2/checkpoint-9984/trainer_state.json"

# Chargement du fichier JSON
with open(path, 'r') as f:
    trainer_state = json.load(f)

# Extraction des logs d'entrainement (ils sont généralement sous 'log_history')
log_history = trainer_state.get('log_history', [])

# Extraire grad_norm, learning_rate et loss à chaque étape/epoch
steps = []
grad_norms = []
learning_rates = []
losses = []

for entry in log_history:
    # Les clefs peuvent varier selon la version de transformers
    if 'step' in entry:
        steps.append(entry['step'])
    elif 'epoch' in entry:
        # Parfois pas de step, utiliser l'epoch
        steps.append(entry['epoch'])
    else:
        continue

    grad_norms.append(entry.get('grad_norm', None))
    learning_rates.append(entry.get('learning_rate', None))
    losses.append(entry.get('loss', None))


# Nettoyer les valeurs None (optionnel, pour éviter les erreurs lors du plot)
def clean_data(x, y):
    cleaned_x = []
    cleaned_y = []
    for xi, yi in zip(x, y):
        if yi is not None:
            cleaned_x.append(xi)
            cleaned_y.append(yi)
    return cleaned_x, cleaned_y


steps_gn, grad_norms = clean_data(steps, grad_norms)
steps_lr, learning_rates = clean_data(steps, learning_rates)
steps_loss, losses = clean_data(steps, losses)

# Tracé
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot loss
color_loss = 'tab:red'
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss', color=color_loss)
ax1.plot(steps_loss, losses, color=color_loss, label='Loss')
ax1.tick_params(axis='y', labelcolor=color_loss)

# Plot learning rate sur ax2 (axe y droit)
ax2 = ax1.twinx()
color_lr = 'tab:blue'
ax2.set_ylabel('Learning Rate', color=color_lr)
ax2.plot(steps_lr, learning_rates, color=color_lr, linestyle='--', label='Learning Rate')
ax2.tick_params(axis='y', labelcolor=color_lr)

# Plot grad_norm sur un 3e axe y optionnel
fig, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(steps_gn, grad_norms, color='tab:green', label='Grad Norm')
ax3.set_xlabel('Step')
ax3.set_ylabel('Grad Norm')
ax3.legend()

plt.show()
