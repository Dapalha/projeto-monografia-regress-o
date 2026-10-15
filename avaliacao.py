"""Métricas por janela e avaliação da decisão final por rastreador."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .features import TARGET, decisao_combinada


def metricas(y, prob, limiar, grupos, janela_min=60):
    y = np.asarray(y)
    pred = (prob >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    duas_classes = len(np.unique(y)) > 1

    # um rastreador conta como sinalizado se tiver ao menos uma janela em alarme
    por_rastreador = pd.DataFrame({"r": grupos, "y": y, "p": pred}).groupby("r").max()
    rvn, rfp, rfn, rvp = confusion_matrix(por_rastreador["y"], por_rastreador["p"],
                                          labels=[0, 1]).ravel()

    meses_rastreador = len(y) * janela_min / (60 * 24 * 30)
    return dict(
        limiar=limiar,
        precisao=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        roc_auc=roc_auc_score(y, prob) if duas_classes else np.nan,
        pr_auc=average_precision_score(y, prob) if duas_classes else np.nan,
        VP=int(vp), FN=int(fn), FP=int(fp), VN=int(vn),
        falsos_por_rastreador_mes=fp / max(meses_rastreador, 1e-9),
        rast_VP=int(rvp), rast_FN=int(rfn), rast_FP=int(rfp), rast_VN=int(rvn),
    )


def imprimir(nome, experimento, m):
    print(f"[{experimento}] {nome}: {m['VP']} de {m['VP'] + m['FN']} janelas de fraude detectadas, "
          f"{m['FP']} falsos alarmes ({m['falsos_por_rastreador_mes']:.2f} por rastreador/mês) | "
          f"precisão {m['precisao']:.3f}  recall {m['recall']:.3f}  "
          f"F1 {m['f1']:.3f}  PR-AUC {m['pr_auc']:.3f}")


def avaliar_decisao(base, prob, params, decisao):
    """Aplica a decisão combinada a cada rastreador e compara com o rótulo real."""
    base = base.assign(_prob=np.nan_to_num(np.asarray(prob, dtype=float)))
    linhas = []
    for rastreador, g in base.groupby("rastreador"):
        g = g.sort_values("inicio_janela")
        veredito, risco, _, _ = decisao_combinada(g, g["_prob"].to_numpy(), params, decisao)
        linhas.append(dict(rastreador=rastreador, real=int(g[TARGET].max()),
                           previsto=int(veredito == "FRAUDE"), risco=risco))
    tabela = pd.DataFrame(linhas)
    vn, fp, fn, vp = confusion_matrix(tabela["real"], tabela["previsto"], labels=[0, 1]).ravel()
    precisao = vp / max(vp + fp, 1)
    recall = vp / max(vp + fn, 1)
    resumo = dict(VP=int(vp), FN=int(fn), FP=int(fp), VN=int(vn), precisao=precisao,
                  recall=recall, f1=2 * precisao * recall / max(precisao + recall, 1e-12))
    return tabela, resumo
