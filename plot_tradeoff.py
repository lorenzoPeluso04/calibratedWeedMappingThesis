import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ==============================================================================
# CONFIGURAZIONE STILE GRAFICI (STILE ACCADEMICO E PULITO)
# ==============================================================================
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
    'legend.fontsize': 9.5,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2
})

def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ==============================================================================
# MAPPATURA COLORI E STILI
# I 6 modelli CE/FL condividono colori; CE = linea continua, FL = tratteggiata
# ==============================================================================
VARIANT_COLORS = {
    'Base (CE)':           '#377eb8',  # blu
    'Temp Scaling (CE)':   '#ff7f00',  # arancione
    'Matrix Scaling (CE)': '#4daf4a',  # verde
    'Base (FL)':           '#377eb8',  # blu (stesso, linea diversa)
    'Temp Scaling (FL)':   '#ff7f00',  # arancione
    'Matrix Scaling (FL)': '#4daf4a',  # verde
}
VARIANT_LINESTYLES = {
    'Base (CE)':           '-',
    'Temp Scaling (CE)':   '-',
    'Matrix Scaling (CE)': '-',
    'Base (FL)':           '--',
    'Temp Scaling (FL)':   '--',
    'Matrix Scaling (FL)': '--',
}
VARIANT_MARKERS = {
    'Base (CE)':           'o',
    'Temp Scaling (CE)':   's',
    'Matrix Scaling (CE)': '^',
    'Base (FL)':           'o',
    'Temp Scaling (FL)':   's',
    'Matrix Scaling (FL)': '^',
}

BACKBONES = ['SegFormer', 'MobileNetV4']

def get_variant(label: str) -> str:
    """Estrae la parte variante dal label completo, es. 'SegFormer Base (CE)' -> 'Base (CE)'."""
    for prefix in ('SegFormer ', 'MobileNetV4 '):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label

# ==============================================================================
# CARICAMENTO DATI
# ==============================================================================
csv_path = "benchmark_results/benchmark_summary_aggregate_merged_400.csv"
if not os.path.exists(csv_path):
    csv_path = "benchmark_summary_aggregate.csv"

df = pd.read_csv(csv_path)
df_grouped = df.groupby(['Model', 'Threshold']).mean(numeric_only=True).reset_index()

# --------------------------------------------------------------------------
# Compatibilità con vecchi label (benchmark eseguito prima del refactoring).
# Se nessun modello inizia con "SegFormer " o "MobileNetV4 ", si applica
# la mappatura automatica (tutto viene trattato come SegFormer).
# --------------------------------------------------------------------------
_OLD_TO_NEW = {
    "Base (non calibrato)":  "SegFormer Base (CE)",
    "Temp Scaling":          "SegFormer Temp Scaling (CE)",
    "Matrix Scaling":        "SegFormer Matrix Scaling (CE)",
    "Focal Loss γ=2.0":      "SegFormer Base (FL)",
    "Focal + Temp Scaling":  "SegFormer Temp Scaling (FL)",
    "Focal + Matrix Scaling": "SegFormer Matrix Scaling (FL)",
    # vecchi label con γ=1.0 non hanno corrispondenza → vengono scartati
    "Focal Loss γ=1.0": None,
}

_has_new_format = any(
    m.startswith(("SegFormer ", "MobileNetV4 "))
    for m in df_grouped["Model"].unique()
)
if not _has_new_format:
    df_grouped["Model"] = df_grouped["Model"].map(_OLD_TO_NEW)
    df_grouped = df_grouped.dropna(subset=["Model"])

all_models = df_grouped['Model'].unique()

segformer_models  = [m for m in all_models if m.startswith('SegFormer')]
mobilenet_models  = [m for m in all_models if m.startswith('MobileNetV4')]

os.makedirs("evaluation_outputs", exist_ok=True)

# ==============================================================================
# FIGURA 1 — TRADEOFF CURVES  (2 righe × 2 colonne)
# Riga 0: SegFormer  |  Riga 1: MobileNetV4
# Col 0: Over-spraying  |  Col 1: Under-spraying
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
fig.suptitle("Tradeoff Over/Under-spraying per Soglia Operativa $\\tau$ (400 segmenti)", fontsize=15, y=1.01)

for row_idx, (backbone, model_list) in enumerate(zip(BACKBONES, [segformer_models, mobilenet_models])):
    ax_over  = axes[row_idx][0]
    ax_under = axes[row_idx][1]

    if not model_list:
        for ax in (ax_over, ax_under):
            ax.text(0.5, 0.5, f"Nessun dato per {backbone}\n(eseguire il benchmark)", 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='#888888', style='italic')
            ax.set_title(f"{backbone} — {'Over' if ax is ax_over else 'Under'}-spraying Rate", pad=10)
            clean_spines(ax)
        continue

    for model in model_list:
        data    = df_grouped[df_grouped['Model'] == model].sort_values('Threshold')
        variant = get_variant(model)
        color   = VARIANT_COLORS.get(variant, '#000000')
        ls      = VARIANT_LINESTYLES.get(variant, '-')
        marker  = VARIANT_MARKERS.get(variant, 'o')

        ax_over.plot(
            data['Threshold'], data['Over_spraying'] * 100,
            marker=marker, markersize=6, markeredgecolor='white', markeredgewidth=1.5,
            linewidth=2.5, label=variant, color=color, linestyle=ls,
        )
        ax_under.plot(
            data['Threshold'], data['Under_spraying'] * 100,
            marker=marker, markersize=6, markeredgecolor='white', markeredgewidth=1.5,
            linewidth=2.5, label=variant, color=color, linestyle=ls,
        )

    ax_over.set_title(f"{backbone} — Over-spraying Rate", pad=10)
    ax_over.set_ylabel("Pixel Sani Trattati (%)")
    if row_idx == 1:
        ax_over.set_xlabel(r"Soglia Operativa $\tau$", labelpad=8)
    clean_spines(ax_over)

    ax_under.set_title(f"{backbone} — Under-spraying Rate", pad=10)
    ax_under.set_ylabel("Pixel Infestati Mancati (%)")
    if row_idx == 1:
        ax_under.set_xlabel(r"Soglia Operativa $\tau$", labelpad=8)
    clean_spines(ax_under)

    # Legenda solo nella colonna destra
    handles, labels = ax_under.get_legend_handles_labels()
    ax_under.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc='upper left',
                    title="Variante\n(— CE  ·  -- FL)", frameon=False)

plt.tight_layout()
output_tradeoff = "evaluation_outputs/tradeoff_curves_400segments.png"
plt.savefig(output_tradeoff, dpi=300, bbox_inches='tight')
print(f"Grafico 1 (Tradeoff) salvato in: {output_tradeoff}")
plt.show()
plt.close(fig)

# ==============================================================================
# FIGURA 2 — DECISION CURVE ANALYSIS  (1 riga × 2 colonne)
# Col 0: SegFormer  |  Col 1: MobileNetV4
# ==============================================================================
print("Calcolo e generazione del grafico di Net Benefit mediato...")

PREVALENZA_INFESTANTI = 0.0418  # Prevalenza media delle infestanti (4.18%)

fig_dca, dca_axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
fig_dca.suptitle("Decision Curve Analysis — Net Benefit Agronomico Mediato (400 segmenti)",
                 fontsize=15, fontweight='bold', y=1.02)

soglie_baseline = np.linspace(df_grouped['Threshold'].min(), df_grouped['Threshold'].max(), 100)
nb_treat_all    = [PREVALENZA_INFESTANTI - (t / (1 - t)) * (1 - PREVALENZA_INFESTANTI) for t in soglie_baseline]

for ax, (backbone, model_list) in zip(dca_axes, zip(BACKBONES, [segformer_models, mobilenet_models])):
    if not model_list:
        ax.text(0.5, 0.5, f"Nessun dato per {backbone}\n(eseguire il benchmark)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#888888', style='italic')
        ax.set_title(f"DCA — {backbone}", fontweight='bold', pad=14)
        clean_spines(ax)
        continue

    all_nb = []

    for model in model_list:
        data_model = df_grouped[df_grouped['Model'] == model].sort_values('Threshold')
        soglie     = data_model['Threshold'].values
        over_vals  = data_model['Over_spraying'].values
        under_vals = data_model['Under_spraying'].values

        nb_list = []
        for tau, over, under in zip(soglie, over_vals, under_vals):
            w  = tau / (1 - tau)
            tp = PREVALENZA_INFESTANTI * (1 - under)
            fp = (1 - PREVALENZA_INFESTANTI) * over
            nb_list.append(tp - w * fp)

        variant = get_variant(model)
        color   = VARIANT_COLORS.get(variant, '#555555')
        ls      = VARIANT_LINESTYLES.get(variant, '-')
        marker  = VARIANT_MARKERS.get(variant, 'o')

        ax.plot(soglie, nb_list,
                marker=marker, markersize=6, markeredgecolor='white', markeredgewidth=1.2,
                linewidth=2.5, label=variant, color=color, linestyle=ls)
        all_nb.extend(nb_list)

    ax.plot(soglie_baseline, nb_treat_all,
            label='Baseline: Tratta Tutto', color='#7f7f7f', linestyle=':', lw=2.5)
    ax.plot(soglie_baseline, np.zeros_like(soglie_baseline),
            label='Baseline: Non Trattare', color='#111111', lw=1.5, linestyle='--')

    ax.axvspan(df_grouped['Threshold'].min(), 0.35, color='#fef0d9', alpha=0.2)
    ax.axvspan(0.35, df_grouped['Threshold'].max(), color='#e5f5f9', alpha=0.25)

    ax.set_title(f"DCA — {backbone}", fontweight='bold', pad=14)
    ax.set_xlabel(r"Soglia di Rischio Decisionale ($\tau$)", labelpad=10)
    ax.set_ylabel("Net Benefit (Indice di Utilità Spaziale)", labelpad=10)
    ax.set_xlim(df_grouped['Threshold'].min() - 0.01, df_grouped['Threshold'].max() + 0.01)
    ymin = min(all_nb + list(nb_treat_all))
    ax.set_ylim(ymin * 1.15 if ymin < 0 else -0.005, PREVALENZA_INFESTANTI * 1.2)
    clean_spines(ax)
    ax.legend(loc='lower right', frameon=True, facecolor='white',
              edgecolor='#dddddd', framealpha=0.9,
              title="Variante\n(— CE  ·  -- FL)")

plt.tight_layout()
output_dca = "evaluation_outputs/decision_curve_analysis_400segments.png"
plt.savefig(output_dca, dpi=300, bbox_inches='tight')
print(f"Grafico 2 (DCA) salvato in: {output_dca}")
plt.show()