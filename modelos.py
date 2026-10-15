"""Definição e treinamento dos três classificadores comparados no trabalho."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODELOS = ["Regressão Logística", "Rede Neural (MLP)", "XGBoost"]
SIGLA = {"Regressão Logística": "rl", "Rede Neural (MLP)": "mlp", "XGBoost": "xgb"}

# Variáveis de alimentação recebem peso maior na escolha das divisões do XGBoost
TERMOS_ENERGIA = ("tensao", "ext_cortada", "bateria", "regra_energia")

# Prefixos de variáveis em que um valor maior nunca deve reduzir o risco
CRESCENTES = ("dur_max_", "frac_ext", "queda_", "salto_", "maior_gap",
              "regra_", "sinal_degradado")


def pesos_energia(features, peso):
    return np.array([peso if any(t in f for t in TERMOS_ENERGIA) else 1.0 for f in features])


def restricoes_monotonas(features):
    def sinal(f):
        if f.startswith(CRESCENTES):
            return 1
        return -1 if f == "freq_local_min" else 0
    return tuple(sinal(f) for f in features)


def criar_modelo(nome, y_treino, features, seed=42):
    imputador = SimpleImputer(strategy="median", keep_empty_features=True)

    if nome == "Regressão Logística":
        return Pipeline([
            ("imp", imputador),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", C=0.5,
                                       max_iter=3000, random_state=seed)),
        ])

    if nome == "Rede Neural (MLP)":
        return Pipeline([
            ("imp", imputador),
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                  batch_size=256, learning_rate_init=1e-3,
                                  early_stopping=True, validation_fraction=0.15,
                                  n_iter_no_change=10, max_iter=300,
                                  random_state=seed)),
        ])

    if nome == "XGBoost":
        negativos = int((y_treino == 0).sum())
        positivos = max(int((y_treino == 1).sum()), 1)
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bynode=0.6,
                             tree_method="hist", eval_metric="aucpr",
                             scale_pos_weight=negativos / positivos,
                             monotone_constraints=restricoes_monotonas(features),
                             n_jobs=2, random_state=seed)

    raise ValueError(f"modelo desconhecido: {nome}")


def sobreamostrar(idx, y, razao=4, seed=42):
    """Repete janelas de fraude no treino da MLP, que não aceita peso por classe."""
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    if len(pos) == 0:
        return idx
    alvo = max(len(neg) // razao, len(pos))
    repetidas = np.random.default_rng(seed).choice(pos, alvo, replace=True)
    return np.concatenate([neg, repetidas])


def treinar(nome, X, y, idx, features, peso_energia=3.0, seed=42):
    modelo = criar_modelo(nome, y[idx], features, seed)

    if nome == "Rede Neural (MLP)":
        idx = sobreamostrar(idx, y, seed=seed)

    if nome == "XGBoost":
        pesos = pesos_energia(features, peso_energia)
        try:
            return modelo.fit(X[idx], y[idx], feature_weights=pesos)
        except TypeError:  # xgboost >= 3 recebe os pesos no construtor
            modelo.set_params(feature_weights=pesos)
            return modelo.fit(X[idx], y[idx])

    return modelo.fit(X[idx], y[idx])


def limiar_por_precisao(y, prob, precisao_min=0.5):
    """Menor limiar que atinge a precisão mínima na validação (maximiza o recall).

    Se nenhum limiar atinge a meta, usa o que maximiza o F0,5.
    """
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, lim = precision_recall_curve(y, prob)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.where(prec >= precisao_min)[0]
    if len(ok):
        return float(lim[ok[np.argmax(rec[ok])]])
    f05 = 1.25 * prec * rec / (0.25 * prec + rec + 1e-12)
    return float(lim[np.argmax(f05)])
