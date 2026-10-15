"""Figuras usadas na monografia e na apresentação (PNG, 300 dpi)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

CORES = {"Regressão Logística": "#1f77b4", "Rede Neural (MLP)": "#2ca02c", "XGBoost": "#d62728"}
ROTULOS = {"precisao": "Precisão", "recall": "Recall", "f1": "F1",
           "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
    })


def _fechar(fig, caminho):
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho)
    plt.show()
    plt.close(fig)


def matriz_confusao(c, titulo, caminho=None, classes=("Normal", "Fraude")):
    m = np.array([[c["VN"], c["FP"]], [c["FN"], c["VP"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(m, cmap="Blues")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j]):,}".replace(",", "."), ha="center", va="center",
                    fontsize=13, color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], [f"Previsto\n{k}" for k in classes])
    ax.set_yticks([0, 1], [f"Real\n{k}" for k in classes])
    ax.set_title(titulo)
    _fechar(fig, caminho)


def curva_roc(y, probs, titulo, caminho=None):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for nome, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=CORES.get(nome), lw=1.8, label=f"{nome} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title(titulo)
    ax.legend(loc="lower right", fontsize=9)
    _fechar(fig, caminho)


def curva_pr(y, probs, titulo, caminho=None):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for nome, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, color=CORES.get(nome), lw=1.8, label=f"{nome} (PR-AUC = {ap:.3f})")
    ax.axhline(np.mean(y), ls="--", color="gray", lw=1, label=f"Aleatório ({np.mean(y):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.set_ylim(0, 1.02)
    ax.set_title(titulo)
    ax.legend(loc="upper right", fontsize=9)
    _fechar(fig, caminho)


def barras_metricas(media, titulo, caminho=None, desvio=None,
                    metricas=("precisao", "recall", "f1", "pr_auc")):
    metricas = list(metricas)
    nomes = list(media.index)
    x = np.arange(len(metricas))
    largura = 0.8 / len(nomes)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, nome in enumerate(nomes):
        valores = media.loc[nome, metricas].astype(float).to_numpy()
        erro = desvio.loc[nome, metricas].astype(float).to_numpy() if desvio is not None else None
        barras = ax.bar(x + (k - (len(nomes) - 1) / 2) * largura, valores, largura,
                        yerr=erro, capsize=3, color=CORES.get(nome), label=nome)
        ax.bar_label(barras, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [ROTULOS[m] for m in metricas])
    ax.set_ylim(0, 1.15)
    ax.set_title(titulo)
    ax.legend(fontsize=9, ncol=len(nomes), loc="upper center")
    _fechar(fig, caminho)


def boxplot_folds(resultados, metrica, titulo, caminho=None):
    nomes = list(dict.fromkeys(resultados["modelo"]))
    dados = [resultados.loc[resultados["modelo"] == n, metrica].dropna().to_numpy() for n in nomes]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    caixas = ax.boxplot(dados, patch_artist=True, widths=0.5)
    for caixa, nome in zip(caixas["boxes"], nomes):
        caixa.set_facecolor(CORES.get(nome))
        caixa.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, (d, nome) in enumerate(zip(dados, nomes), 1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(d)), d, s=18, color=CORES.get(nome), zorder=3)
    ax.set_xticks(range(1, len(nomes) + 1), nomes)
    ax.set_ylabel(ROTULOS.get(metrica, metrica))
    ax.set_title(titulo)
    _fechar(fig, caminho)


def composicao_folds(comp, caminho=None):
    c = comp[comp["rep"] == 0]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(c))
    ax.bar(x, c["janelas_fraude"], color="#7f7f7f")
    for xi, (n, conc) in enumerate(zip(c["com_fraude"], c["concentracao"])):
        ax.text(xi, c["janelas_fraude"].iloc[xi], f"{n} rastr.\n{conc:.0%} em 1",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"Fold {f + 1}" for f in c["fold"]])
    ax.set_ylabel("Janelas com fraude no teste")
    ax.set_title("Distribuição das fraudes entre os folds")
    _fechar(fig, caminho)


def importancia(modelo, nome, features, X, y, caminho=None, n=15, seed=42):
    if nome == "Regressão Logística":
        v = pd.Series(modelo.named_steps["clf"].coef_[0], index=features)
        v = v.reindex(v.abs().sort_values().index).tail(n)
        eixo = "Coeficiente (variáveis padronizadas)"
    elif nome == "XGBoost":
        v = pd.Series(modelo.feature_importances_, index=features).sort_values().tail(n)
        eixo = "Importância (ganho)"
    else:
        # a MLP não tem coeficientes interpretáveis: usa importância por permutação
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        amostra = np.concatenate([rng.choice(pos, min(len(pos), 3000), replace=False),
                                  rng.choice(neg, min(len(neg), 17000), replace=False)])
        r = permutation_importance(modelo, X[amostra], y[amostra], scoring="average_precision",
                                   n_repeats=3, random_state=seed, n_jobs=1)
        v = pd.Series(r.importances_mean, index=features).sort_values().tail(n)
        eixo = "Redução média da PR-AUC (permutação)"

    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(v) + 1.3))
    ax.barh(v.index, v.to_numpy(), color=[CORES.get(nome) if x >= 0 else "#9e9e9e" for x in v])
    ax.set_xlabel(eixo)
    ax.set_title(f"{nome}: variáveis mais relevantes")
    ax.grid(axis="y", alpha=0)
    _fechar(fig, caminho)
    return v
