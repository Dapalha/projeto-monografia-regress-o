"""Montagem da base de janelas a partir dos arquivos *_treated.csv."""
import gc

import numpy as np
import pandas as pd

from .features import PARAMS, TARGET, agregar_janelas, features_registro


def montar_base(arquivos, params=PARAMS):
    """Lê cada rastreador, calcula as features e agrega em janelas de tempo.

    Retorna a tabela de janelas de todos os rastreadores e a lista com a
    duração (min) de cada episódio de jammer encontrado.
    """
    blocos, duracoes_jammer = [], []
    for k, fp in enumerate(arquivos, 1):
        try:
            df = pd.read_csv(fp, sep=";", low_memory=False)
            if "timestamp" not in df.columns:
                print(f"{fp.name}: sem coluna timestamp, ignorado")
                continue
            registros = features_registro(df, params)
            inicio = registros["_ini_jammer"].astype(bool)
            duracoes_jammer += registros.loc[inicio, "_dur_jammer"].dropna().tolist()
            janelas = agregar_janelas(registros, params)
        except Exception as e:
            print(f"{fp.name}: {type(e).__name__}: {e}")
            continue
        janelas["arquivo"] = fp.stem.replace("_treated", "")
        blocos.append(janelas)
        del df, registros
        gc.collect()
        if k % 20 == 0:
            print(f"{k}/{len(arquivos)} arquivos processados")

    base = pd.concat(blocos, ignore_index=True)
    contagens = [c for c in base.columns if c.startswith("n_motivo_")]
    base[contagens] = base[contagens].fillna(0)
    return base, duracoes_jammer


def aplicar_fraudes_confirmadas(base, caminho, params=PARAMS):
    """Substitui o rótulo automático pelos períodos de fraude confirmados.

    O CSV deve ter as colunas rastreador;inicio;fim (datas em dd/mm/aaaa hh:mm).
    A coluna 'rastreador' pode conter o Module Id ou o nome do arquivo.
    """
    conf = pd.read_csv(caminho, sep=";", dtype={"rastreador": str})
    conf["inicio"] = pd.to_datetime(conf["inicio"], dayfirst=True)
    conf["fim"] = pd.to_datetime(conf["fim"], dayfirst=True)

    fim_janela = base["inicio_janela"] + pd.Timedelta(minutes=params["JANELA_MIN"])
    rastreador = base["rastreador"].astype(str)
    arquivo = base["arquivo"].astype(str)
    rotulo = np.zeros(len(base), dtype="int8")
    for _, c in conf.iterrows():
        mesmo = (rastreador == c["rastreador"]) | (arquivo == c["rastreador"])
        sobrepoe = (base["inicio_janela"] < c["fim"]) & (fim_janela > c["inicio"])
        rotulo[(mesmo & sobrepoe).to_numpy()] = 1
    base[TARGET] = rotulo
    return len(conf)
