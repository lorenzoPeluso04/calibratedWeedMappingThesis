import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Configurazione dello stile
sns.set_style("whitegrid")
sns.set_palette("husl")

# Caricamento del file CSV
csv_path = Path(__file__).parent / 'superpixel_recommendation_14_segformer.csv'
df = pd.read_csv(csv_path)

print(f"File caricato: {csv_path}")
print(f"Shape del dataframe: {df.shape}\n")

# ==================== STATISTICHE DESCRITTIVE ====================
mean_weed_prob = df['mean_weed_prob']

print("=== STATISTICHE DESCRITTIVE ===\n")
print(f"Numero di superpixel: {len(mean_weed_prob)}")
print(f"\nMedia: {mean_weed_prob.mean():.6f}")
print(f"Mediana: {mean_weed_prob.median():.6f}")
print(f"Deviazione Standard: {mean_weed_prob.std():.6f}")
print(f"Minimo: {mean_weed_prob.min():.6f}")
print(f"Massimo: {mean_weed_prob.max():.6f}")
print(f"\nQuartili:")
print(f"  25°: {mean_weed_prob.quantile(0.25):.6f}")
print(f"  50°: {mean_weed_prob.quantile(0.50):.6f}")
print(f"  75°: {mean_weed_prob.quantile(0.75):.6f}")
print(f"  90°: {mean_weed_prob.quantile(0.90):.6f}")
print(f"  95°: {mean_weed_prob.quantile(0.95):.6f}\n")

# ==================== CREAZIONE GRAFICI ====================
fig = plt.figure(figsize=(16, 12))

# --- Grafico 1: Istogramma della distribuzione ---
ax1 = plt.subplot(2, 3, 1)
ax1.hist(mean_weed_prob, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(mean_weed_prob.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {mean_weed_prob.mean():.4f}')
ax1.axvline(mean_weed_prob.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {mean_weed_prob.median():.4f}')
ax1.set_xlabel('Mean Weed Probability', fontsize=11)
ax1.set_ylabel('Frequenza', fontsize=11)
ax1.set_title('Istogramma Distribuzione Mean Weed Prob', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Grafico 2: Grafico a barre per intervalli ---
ax2 = plt.subplot(2, 3, 2)

# Definizione intervalli
intervals = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.0)]
interval_labels = ['0.00-0.05', '0.05-0.10', '0.10-0.15', '0.15-0.20', '>0.20']
counts = []

for start, end in intervals:
    count = len(mean_weed_prob[(mean_weed_prob >= start) & (mean_weed_prob < end)])
    counts.append(count)

colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b', '#8b0000']
bars = ax2.bar(interval_labels, counts, color=colors, edgecolor='black', alpha=0.8)

# Aggiunta dei valori sopra le barre
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}\n({count/len(mean_weed_prob)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Numero di Superpixel', fontsize=11)
ax2.set_title('Distribuzione per Intervalli di Probabilità', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# --- Grafico 3: Box plot ---
ax3 = plt.subplot(2, 3, 3)
bp = ax3.boxplot(mean_weed_prob, vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
ax3.set_ylabel('Mean Weed Probability', fontsize=11)
ax3.set_title('Box Plot Distribuzione', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# --- Grafico 4: Distribuzione per Zona Blu ---
ax4 = plt.subplot(2, 3, 4)
zona_blu = df[df['zone'] == 'Zona Blu']['mean_weed_prob']
ax4.hist(zona_blu, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
ax4.axvline(zona_blu.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {zona_blu.mean():.4f}')
ax4.set_xlabel('Mean Weed Probability', fontsize=11)
ax4.set_ylabel('Frequenza', fontsize=11)
ax4.set_title(f'Zona Blu (n={len(zona_blu)})', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# --- Grafico 5: Distribuzione per Zona Gialla ---
ax5 = plt.subplot(2, 3, 5)
zona_gialla = df[df['zone'] == 'Zona Gialla']['mean_weed_prob']
ax5.hist(zona_gialla, bins=20, color='#f1c40f', edgecolor='black', alpha=0.7)
ax5.axvline(zona_gialla.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {zona_gialla.mean():.4f}')
ax5.set_xlabel('Mean Weed Probability', fontsize=11)
ax5.set_ylabel('Frequenza', fontsize=11)
ax5.set_title(f'Zona Gialla (n={len(zona_gialla)})', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# --- Grafico 6: Distribuzione per Zona Rossa ---
ax6 = plt.subplot(2, 3, 6)
zona_rossa = df[df['zone'] == 'Zona Rossa']['mean_weed_prob']
ax6.hist(zona_rossa, bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
ax6.axvline(zona_rossa.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Media: {zona_rossa.mean():.4f}')
ax6.set_xlabel('Mean Weed Probability', fontsize=11)
ax6.set_ylabel('Frequenza', fontsize=11)
ax6.set_title(f'Zona Rossa (n={len(zona_rossa)})', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
print("Grafico salvato come: distribution_analysis.png")
plt.show()

# ==================== STATISTICHE PER ZONA ====================
print("\n=== STATISTICHE PER ZONA ===\n")
for zona in ['Zona Blu', 'Zona Gialla', 'Zona Rossa']:
    zona_data = df[df['zone'] == zona]['mean_weed_prob']
    print(f"{zona} (n={len(zona_data)}):")
    print(f"  Media: {zona_data.mean():.6f}")
    print(f"  Mediana: {zona_data.median():.6f}")
    print(f"  Std Dev: {zona_data.std():.6f}")
    print(f"  Min-Max: [{zona_data.min():.6f}, {zona_data.max():.6f}]")
    
    # Conteggio per intervalli
    interval_counts = []
    for start, end in intervals:
        count = len(zona_data[(zona_data >= start) & (zona_data < end)])
        interval_counts.append(count)
    print(f"  Intervalli: {list(zip(interval_labels, interval_counts))}\n")

print("✓ Analisi completata!")
