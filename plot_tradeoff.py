import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ==============================================================================
# CONFIGURAZIONE STILE GRAFICI (STILE ACCADEMICO E PULITO)
# ==============================================================================
# Usiamo il default per rimuovere sfondi grigi, poi personalizziamo
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'figure.titlesize': 16,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2
})

# Funzione per rimuovere i bordi superflui (spines)
def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Carica i dati
csv_path = "benchmark_results/benchmark_summary_aggregate.csv"
if not os.path.exists(csv_path):
    csv_path = "benchmark_summary_aggregate.csv" 

df = pd.read_csv(csv_path)

# Calcoliamo la media
df_grouped = df.groupby(['Model', 'Threshold']).mean(numeric_only=True).reset_index()

# Creazione della figura 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

models = df_grouped['Model'].unique()
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
model_colors = dict(zip(models, colors[:len(models)]))

for model in models:
    data = df_grouped[df_grouped['Model'] == model]
    
    # Grafico 1: Over-spraying
    ax1.plot(data['Threshold'], data['Over_spraying'] * 100, 
             marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1.5,
             linewidth=2.5, label=model, color=model_colors.get(model, '#000000'))
    
    # Grafico 2: Under-spraying
    ax2.plot(data['Threshold'], data['Under_spraying'] * 100, 
             marker='s', markersize=6, markeredgecolor='white', markeredgewidth=1.5,
             linewidth=2.5, linestyle='--', label=model, color=model_colors.get(model, '#000000'))

# Configurazione Grafico Over-spraying
ax1.set_title("Spreco di Erbicida (Over-spraying Rate)", pad=15)
ax1.set_xlabel(r"Soglia Operativa $\tau$", labelpad=10)
ax1.set_ylabel("Pixel Sani Trattati (%)")
clean_spines(ax1)

# Configurazione Grafico Under-spraying
ax2.set_title("Erbacce Non Trattate (Under-spraying Rate)", pad=15)
ax2.set_xlabel(r"Soglia Operativa $\tau$", labelpad=10)
ax2.set_ylabel("Pixel Infestati Mancati (%)")
clean_spines(ax2)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Modelli / Calibrazioni", frameon=False)

plt.tight_layout()

# Salvataggio
os.makedirs("evaluation_outputs", exist_ok=True)
output_plot = "evaluation_outputs/tradeoff_curves_thesis.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"🎉 Grafico 1 salvato con successo in: {output_plot}")
plt.show()
plt.close(fig) # Chiude la figura per evitare sovrapposizioni

# ==============================================================================
# SEZIONE 2: DECISION CURVE ANALYSIS (NET BENEFIT MEDIATO)
# ==============================================================================
print("📊 Calcolo e generazione del grafico di Net Benefit mediato...")

PREVALENZA_INFESTANTI = 0.0418  # Prevalenza media delle infestanti (4.18%)

dca_colors = {
    'Base (non calibrato)': '#e41a1c',       
    'Temp Scaling': '#4daf4a',               
    'Focal Loss γ=1.0': '#377eb8',           
    'Focal Loss γ=2.0': '#ff7f00',           
    'Focal + Temp Scaling': '#984ea3'        
}

fig_dca, ax3 = plt.subplots(figsize=(10, 7))

# Ciclo sui modelli
for model in models:
    data_model = df_grouped[df_grouped['Model'] == model].sort_values('Threshold')
    soglie = data_model['Threshold'].values
    over_spraying_medio = data_model['Over_spraying'].values
    under_spraying_medio = data_model['Under_spraying'].values

    nb_modello = []
    for tau, over, under in zip(soglie, over_spraying_medio, under_spraying_medio):
        w_weight = tau / (1 - tau)
        tp_rate = PREVALENZA_INFESTANTI * (1 - under)
        fp_rate = (1 - PREVALENZA_INFESTANTI) * over
        nb = tp_rate - (w_weight * fp_rate)
        nb_modello.append(nb)
    
    color = dca_colors.get(model, '#555555')
    ax3.plot(soglie, nb_modello, marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1.2, 
             linewidth=2.5, label=f'Strategia: {model}', color=color)

# Baseline
soglie_baseline = np.linspace(df_grouped['Threshold'].min(), df_grouped['Threshold'].max(), 100)
nb_all = [PREVALENZA_INFESTANTI - ((t / (1 - t)) * (1 - PREVALENZA_INFESTANTI)) for t in soglie_baseline]

ax3.plot(soglie_baseline, nb_all, label='Baseline: Tratta Tutto (Cieca)', color='#7f7f7f', linestyle=':', lw=2.5)
ax3.plot(soglie_baseline, np.zeros_like(soglie_baseline), label='Baseline: Non Trattare', color='#111111', lw=1.5, linestyle='--')

# Aree evidenziate (Alpha ridotto per maggiore eleganza)
ax3.axvspan(0.02, 0.35, color='#fef0d9', alpha=0.2, label='Zona Vantaggio (Base)')
ax3.axvspan(0.35, 0.50, color='#e5f5f9', alpha=0.3, label='Zona Inversione (Calibrazione)')

# Estetica DCA
ax3.set_title("Decision Curve Analysis (DCA)\nValutazione del Beneficio Netto Agronomico Mediato", fontweight='bold', pad=18)
ax3.set_xlabel(r"Soglia di Rischio Decisionale ($\tau$)", labelpad=12)
ax3.set_ylabel("Net Benefit (Indice di Utilità Spaziale)", labelpad=12)
ax3.set_xlim(df_grouped['Threshold'].min() - 0.01, df_grouped['Threshold'].max() + 0.01)
ax3.set_ylim(min(nb_all) * 0.5, PREVALENZA_INFESTANTI * 1.15)
clean_spines(ax3)

# Legenda riposizionata per non coprire i dati
ax3.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#dddddd', framealpha=0.9, fontsize=9.5)

# Riquadro riassuntivo (Summary Box) migliorato
props = dict(boxstyle='round,pad=0.8', facecolor='#fbf8cc', edgecolor='#e3d5ca', alpha=0.85)

plt.tight_layout()
output_dca = "evaluation_outputs/decision_curve_analysis_net_benefit.png"
plt.savefig(output_dca, dpi=300, bbox_inches='tight')

print(f"🎉 Grafico 2 (DCA Net Benefit Mediato) salvato con successo in: {output_dca}")
plt.show()