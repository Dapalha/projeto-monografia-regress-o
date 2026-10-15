"""Classificação de um arquivo de auditoria com os modelos treinados."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import agregar_janelas, decisao_combinada, features_registro
from .figuras import CORES
from .tratamento import tratar_arquivo


def carregar(caminho):
    """Aceita a planilha bruta de auditoria (.csv/.xlsx) ou um *_treated.csv."""
    fp = Path(caminho)
    if fp.name.endswith("_treated.csv"):
        return pd.read_csv(fp, sep=";", low_memory=False)
    df, _ = tratar_arquivo(fp)
    return df


def matriz(janelas, features):
    X = janelas.reindex(columns=features)
    contagens = [c for c in features if c.startswith("n_motivo_")]
    X[contagens] = X[contagens].fillna(0)
    X = X.to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    return X


def analisar_arquivo(caminho, pacote, modelos=None):
    """Classifica o arquivo de um rastreador com cada modelo pedido.

    Retorna (informações do arquivo, resultado por modelo, tabela de janelas).
    """
    params, decisao = pacote["params"], pacote["decisao"]
    janelas = agregar_janelas(features_registro(carregar(caminho), params), params)
    janelas = janelas.reset_index(drop=True)
    X = matriz(janelas, pacote["features"])

    resultados = {}
    for nome in modelos or list(pacote["modelos"]):
        prob = pacote["modelos"][nome]["modelo"].predict_proba(X)[:, 1]
        veredito, risco, episodios, _ = decisao_combinada(janelas, prob, params, decisao)
        confianca = risco if veredito == "FRAUDE" else 1 - risco
        resultados[nome] = dict(
            veredito=veredito,
            risco_pct=round(100 * risco, 1),
            confianca_pct=round(100 * confianca, 1),
            episodios_fraude=int(episodios["fraude"].sum()) if len(episodios) else 0,
            episodios_com_indicios=len(episodios),
            episodios=episodios,
            prob=prob,
        )

    votos = sum(r["veredito"] == "FRAUDE" for r in resultados.values())
    info = dict(
        arquivo=Path(caminho).name,
        rastreador=str(janelas["rastreador"].iloc[0]),
        periodo=f"{janelas['inicio_janela'].min():%d/%m/%Y} a {janelas['inicio_janela'].max():%d/%m/%Y}",
        janelas=len(janelas),
        consenso="FRAUDE" if votos > len(resultados) / 2 else "NÃO FRAUDE",
        votos_fraude=f"{votos} de {len(resultados)}",
    )
    return info, resultados, janelas


def tabela_resumo(resultados):
    return pd.DataFrame([
        {"Modelo": nome, "Veredito": r["veredito"], "Risco (%)": r["risco_pct"],
         "Confiança (%)": r["confianca_pct"], "Episódios de fraude": r["episodios_fraude"],
         "Episódios com indícios": r["episodios_com_indicios"]}
        for nome, r in resultados.items()
    ])


def tabela_episodios(resultados, so_fraude=False):
    partes = []
    for nome, r in resultados.items():
        ep = r["episodios"]
        if len(ep):
            partes.append(ep.assign(modelo=nome))
    if not partes:
        return pd.DataFrame(columns=["modelo", "inicio", "fim", "duracao_h", "n_indicios",
                                     "indicios", "confianca", "fraude"])
    ep = pd.concat(partes, ignore_index=True)
    if so_fraude:
        ep = ep[ep["fraude"]]
    ep = ep.sort_values(["fraude", "confianca"], ascending=False)
    ep["inicio"] = pd.to_datetime(ep["inicio"]).dt.strftime("%d/%m/%Y %H:%M")
    ep["fim"] = pd.to_datetime(ep["fim"]).dt.strftime("%d/%m/%Y %H:%M")
    ep["duracao_h"] = ep["duracao_h"].astype(float).round(1)
    ep["confianca"] = (100 * ep["confianca"].astype(float)).round(1)
    ep["fraude"] = ep["fraude"].map({True: "sim", False: "não"})
    return ep[["modelo", "fraude", "confianca", "inicio", "fim", "duracao_h",
               "n_indicios", "indicios", "lat", "lng"]].reset_index(drop=True)


def grafico_risco(janelas, resultados, decisao, titulo="", caminho=None):
    """Risco ao longo do tempo: um painel por modelo.

    Faixas vermelhas = episódios classificados como fraude;
    faixas laranja = indícios que não chegaram ao limite.
    """
    nomes = list(resultados)
    fig, eixos = plt.subplots(len(nomes), 1, figsize=(13, 2.4 * len(nomes) + 0.8),
                              sharex=True, squeeze=False)
    for ax, nome in zip(eixos[:, 0], nomes):
        r = resultados[nome]
        ax.plot(janelas["inicio_janela"], r["prob"], lw=0.7, color=CORES.get(nome, "C0"))
        for _, e in r["episodios"].iterrows():
            ax.axvspan(e["inicio"], e["fim"], lw=0,
                       color="red" if e["fraude"] else "orange",
                       alpha=0.25 if e["fraude"] else 0.12)
        ax.axhline(decisao["RISCO_MIN"], color="red", ls="--", lw=0.8)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Risco")
        ax.set_title(f"{nome}: {r['veredito']} (risco {r['risco_pct']:.1f}%)", loc="left", fontsize=10)
    if titulo:
        fig.suptitle(titulo)
    fig.tight_layout()
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, dpi=300, bbox_inches="tight")
    return fig
