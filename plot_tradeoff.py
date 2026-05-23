import pandas as pd
import matplotlib.pyplot as plt
import os

# Configurazione stile grafici per la tesi usando stili standard di matplotlib
plt.style.use('tabular') if 'tabular' in plt.style.available else plt.style.use('ggplot')
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'figure.titlesize': 14})

# Carica i dati
csv_path = "benchmark_results/benchmark_summary_aggregate.csv"
if not os.path.exists(csv_path):
    csv_path = "benchmark_summary_aggregate.csv" 

df = pd.read_csv(csv_path)

# Calcoliamo la media su tutte le immagini per fare un grafico aggregato pulito
df_grouped = df.groupby(['Model', 'Threshold']).mean(numeric_only=True).reset_index()

# Creazione della figura con due sotto-grafici (Over-spraying vs Under-spraying)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

models = df_grouped['Model'].unique()
# Mappa di colori standard nativa di matplotlib
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
model_colors = dict(zip(models, colors[:len(models)]))

for model in models:
    data = df_grouped[df_grouped['Model'] == model]
    
    # Grafico 1: Over-spraying (Spreco di fitofarmaco)
    ax1.plot(data['Threshold'], data['Over_spraying'] * 100, 
             marker='o', linewidth=2, label=model, color=model_colors.get(model, '#000000'))
    
    # Grafico 2: Under-spraying (Erbacce sopravvissute)
    ax2.plot(data['Threshold'], data['Under_spraying'] * 100, 
             marker='s', linewidth=2, linestyle='--', label=model, color=model_colors.get(model, '#000000'))

# Configurazione grafico Over-spraying
ax1.set_title("Spreco di Erbicida (Over-spraying Rate)")
ax1.set_xlabel(r"Soglia Operativa $\tau$")
ax1.set_ylabel("Pixel Sani Trattati (%)")
ax1.grid(True, linestyle=':', alpha=0.6)

# Configurazione grafico Under-spraying
ax2.set_title("Erbacce Non Trattate (Under-spraying Rate)")
ax2.set_xlabel(r"Soglia Operativa $\tau$")
ax2.set_ylabel("Pixel Infestati Mancati (%)")
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Modelli / Calibrazioni")

plt.tight_layout()

# Crea la cartella di output se non esiste
os.makedirs("evaluation_outputs", exist_ok=True)
output_plot = "evaluation_outputs/tradeoff_curves_thesis.png"

plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"🎉 Grafico salvato con successo in: {output_plot}")
plt.show()