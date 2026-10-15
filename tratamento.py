"""Leitura e limpeza das planilhas de auditoria exportadas pela plataforma de rastreamento.

Datas e durações são convertidas para tipos temporais antes do one-hot-encoding,
para que data, hora e minuto não se percam. Também corrige problemas da exportação:
coordenadas com separador de milhar, acentuação corrompida, ';' dentro do endereço
e cabeçalhos com nomes repetidos.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATE_FMT = "%d/%m/%Y %H:%M"   # formato de 'Location date', ex.: 02/01/2024 00:19
COORD_DECIMALS = 7            # -129.051.470 -> -12.9051470


def corrigir_texto(s):
    """Corrige acentuação corrompida (ex.: 'NÃƒO' -> 'NÃO')."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def para_datetime(serie):
    """Texto 'dd/mm/aaaa HH:MM[:SS]' ou datetime do Excel -> datetime64."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie
    s = serie.astype(str).str.strip()
    dt = pd.to_datetime(s, format=DATE_FMT, errors="coerce")
    faltando = dt.isna()
    if faltando.any():   # tenta com segundos / outros formatos
        dt[faltando] = pd.to_datetime(s[faltando], dayfirst=True, errors="coerce")
    return dt


def para_segundos(serie):
    """'HH:MM:SS' -> segundos (float). Aceita também timedelta/time do Excel."""
    s = serie.astype(str).str.strip()
    return pd.to_timedelta(s, errors="coerce").dt.total_seconds()


def corrigir_coordenada(valor, casas=COORD_DECIMALS):
    """
    A exportação gravou as coordenadas com pontos de milhar:
    '-129.051.470' -> -12.9051470  |  '-384.322.470' -> -38.4322470.
    Valores já numéricos e válidos (|x| <= 180) são mantidos. 0 vira NaN.
    """
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float)) and abs(valor) <= 180 and float(valor).is_integer() is False:
        return float(valor) if valor != 0 else np.nan
    s = str(valor).strip().replace(",", ".")
    neg = s.startswith("-")
    digitos = "".join(ch for ch in s if ch.isdigit())
    if not digitos or int(digitos) == 0:
        return np.nan
    if s.count(".") == 1 and len(s.split(".")[1]) == casas:   # já está correto (ex.: -12.9051470)
        v = float(s)
    else:
        v = int(digitos) / 10 ** casas
    v = -v if neg else v
    return v if abs(v) <= 180 else np.nan


def sim_nao(serie):
    m = {"SIM": 1, "NÃO": 0, "NAO": 0, "TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
    return serie.astype(str).str.strip().str.upper().map(m)


EVENTO_INICIO = "ENTERING GSM JAMMER AREA"
EVENTO_FIM    = "LEAVING GSM JAMMER AREA"

def rotular_fraude(df):
    estado = pd.Series(np.nan, index=df.index)
    estado[df["Reason Code"] == EVENTO_INICIO] = 1
    estado[df["Reason Code"] == EVENTO_FIM] = 1      # o próprio evento de saída ainda está no jammer
    # depois do evento de saída, volta a 0
    proximo_normal = (df["Reason Code"] == EVENTO_FIM).shift(fill_value=False)
    estado[proximo_normal & estado.isna()] = 0
    estado.iloc[0] = 0 if pd.isna(estado.iloc[0]) else estado.iloc[0]
    return estado.ffill().astype(int)


CATEGORICAS_ONEHOT = ["Reason Code", "HDOP STATUS"]
DESCARTAR = ["Place", "Plate", "Ibutton", "Creation date", "Reception date"]

LINHAS_CORRIGIDAS = {}   # arquivo -> nº de linhas com ";" extra

def ler_bruto(fp):
    """
    Lê a planilha de auditoria linha a linha.
    Linhas com mais campos que o cabeçalho (ex.: endereço em 'Place' contendo ';')
    têm o excedente reunido de volta na ÚLTIMA coluna, em vez de quebrar a leitura.
    Linhas com campos a menos são completadas com vazio.
    """
    if fp.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(fp, dtype=str)
    raw = fp.read_bytes()
    texto = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            texto = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    linhas = [l for l in texto.splitlines() if l.strip()]
    cab = [c.strip() for c in linhas[0].split(";")]
    n = len(cab)
    dados, corrigidas = [], 0
    for l in linhas[1:]:
        campos = l.split(";")
        if len(campos) > n:
            campos = campos[:n - 1] + [";".join(campos[n - 1:])]
            corrigidas += 1
        elif len(campos) < n:
            campos += [""] * (n - len(campos))
        dados.append(campos)
    LINHAS_CORRIGIDAS[fp.name] = corrigidas
    df = pd.DataFrame(dados, columns=cab, dtype=object)
    return df.replace({"": np.nan})


def limpar_texto(v):
    """Strip + correção de acentos, segura para NaN e valores não-texto."""
    if isinstance(v, str):
        v = corrigir_texto(v).strip()
        return v if v else np.nan
    return v


COLUNAS_DUPLICADAS = {}   # arquivo -> nomes repetidos no cabeçalho

def nomes_unicos(colunas, arquivo=""):
    """
    Alguns arquivos têm cabeçalho com nomes repetidos (ex.: duas colunas 'Lat'
    ou colunas vazias). Com nomes repetidos, df['col'] devolve um DataFrame
    em vez de uma Series. Aqui cada repetição recebe sufixo: Lat, Lat_2, ...
    Colunas sem nome viram 'sem_nome', 'sem_nome_2', ...
    """
    vistos, saida, repetidos = {}, [], []
    for c in colunas:
        c = str(c).strip() or "sem_nome"
        if c.startswith("Unnamed"):
            c = "sem_nome"
        if c in vistos:
            vistos[c] += 1
            repetidos.append(c)
            saida.append(f"{c}_{vistos[c]}")
        else:
            vistos[c] = 1
            saida.append(c)
    if repetidos:
        COLUNAS_DUPLICADAS[arquivo] = sorted(set(repetidos))
    return saida


def tratar_arquivo(fp):
    df = ler_bruto(fp).dropna(how="all")
    df.columns = nomes_unicos(df.columns, fp.name)
    # Remove colunas totalmente vazias e sem nome (sobras de ';' no fim do cabeçalho)
    df = df.drop(columns=[c for c in df.columns
                          if c.startswith("sem_nome") and df[c].isna().all()])
    for i in range(df.shape[1]):                      # por posição: nunca devolve DataFrame
        if df.iloc[:, i].dtype == object:
            df.iloc[:, i] = df.iloc[:, i].map(limpar_texto)
    if "Location date" not in df.columns:
        raise ValueError(f"coluna 'Location date' ausente. Colunas: {list(df.columns)}")

    # --- Tempo: mantém data + hora + minuto -------------------------
    df["timestamp"] = para_datetime(df["Location date"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    df = df.drop_duplicates().reset_index(drop=True)
    df["Data"]   = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["Hora"]   = df["timestamp"].dt.hour
    df["Minuto"] = df["timestamp"].dt.minute
    df["hora_decimal"] = df["Hora"] + df["Minuto"] / 60
    df["hora_sin"] = np.sin(2 * np.pi * df["hora_decimal"] / 24)   # codificação cíclica
    df["hora_cos"] = np.cos(2 * np.pi * df["hora_decimal"] / 24)
    df["dia_semana"] = df["timestamp"].dt.dayofweek
    for c in ["Delay", "Reception delay"]:
        if c in df:
            df[c.replace(" ", "_") + "_seg"] = para_segundos(df[c])
    df = df.drop(columns=[c for c in ["Delay", "Reception delay", "Location date"] if c in df])

    # --- Numéricas ---------------------------------------------------
    for c in ["Speed", "GPS Satellites", "GSM Signal", "HDOP",
              "External Voltage", "Internal Battery", "Hodometer"]:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    df["Lat"] = df["Lat"].map(corrigir_coordenada)
    df["Lng"] = df["Lng"].map(corrigir_coordenada)

    # --- Binárias ----------------------------------------------------
    for c in ["Ignition", "Block Car"]:
        if c in df:
            df[c] = sim_nao(df[c])

    # --- Rótulo (antes do one-hot, usa o texto do Reason Code) -------
    df["Fraude"] = rotular_fraude(df)

    # --- One-hot só das categóricas reais ----------------------------
    prefixos = {"Reason Code": "motivo", "HDOP STATUS": "hdop_status"}
    cols_oh = [c for c in CATEGORICAS_ONEHOT if c in df]
    df = pd.get_dummies(df, columns=cols_oh, prefix=[prefixos[c] for c in cols_oh], dtype=int)

    rastreador = (str(df["Module Id"].dropna().iloc[0])
                  if "Module Id" in df and df["Module Id"].notna().any()
                  else fp.stem.replace("_auditoria", ""))
    df["rastreador"] = rastreador
    df = df.drop(columns=[c for c in DESCARTAR + ["Module Id"] if c in df])
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df, rastreador
