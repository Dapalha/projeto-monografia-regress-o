"""Variáveis por registro, agregação em janelas de tempo e decisão combinada.

Todas as variáveis são calculadas dentro de um único rastreador, em ordem temporal,
e comparam o rastreador com ele mesmo (tensão relativa à própria mediana, área
habitual de circulação etc.).
"""
import re

import numpy as np
import pandas as pd

# Parâmetros de domínio
TARGET = "Fraude"
VAZAMENTO_PATTERNS = ["JAMMER"]      # colunas que definem o rótulo nunca viram feature

PARAMS = dict(
    JANELA_MIN=60,             # cada amostra do modelo = 1 janela de 60 min de um rastreador
    ROTULO_DURACAO_MIN=30,     # jammer só é FRAUDE se persistir >= 30 min (0 = rótulo antigo)
    DURACAO_MIN_SUSPEITA=30,   # persistência mínima das condições de sinal (min)
    DELAY_ALTO_SEG=600,        # atraso de recepção anormal (no exemplo, p90 normal ~ 400 s)
    HDOP_ALTO=3.0,
    SATELITES_MIN=4,
    GSM_FRACO_DBM=-105,        # no exemplo, dentro do jammer o GSM fica em -111 dBm
    GAP_TRANSMISSAO_MIN=45,    # parado, o rastreador transmite a cada ~30 min
    TENSAO_EXT_MIN=6.0,        # abaixo disso: alimentação externa cortada (V)
    PARADO_KMH=3,
    GRADE_GRAUS=0.01,          # célula de ~1 km para definir a "área habitual"
    FREQ_LOCAL_RARO=0.01,      # local onde o rastreador passa < 1% do tempo = incomum
    SALTO_KM=1.0,              # deslocamento após um apagão de transmissão
    KMH_MOVIMENTO=5,           # acima disso o veículo está andando
    # Violação de alimentação:
    # tensão externa cortada + veículo andando com ignição DESLIGADA
    MIN_REG_MOV_SEM_IGN=3,     # nº mínimo de registros andando com ignição off e energia cortada
    # Rótulo: "jammer" | "energia" | "ambos"  (fraudes_confirmadas.csv tem prioridade)
    ROTULO_MODO="ambos",
)

# Famílias de Reason Code: cada firmware escreve o mesmo evento de um jeito
# (ex.: "ENTERING GSM JAMMER AREA" x "GMS JAMMING ACTIVATED").
FAMILIAS_EVENTO = {
    "evt_jammer_inicio": r"ENTERING.*JAMM|JAMMING ACTIVATED|JAMMER ON",
    "evt_jammer_fim":    r"LEAVING.*JAMM|JAMMING DISABLED|JAMMER OFF",
    # Lista "REASON CODES (FRAUDE)": CALAMP, CONCOX, MXT, NEWTEC, PV_MXT,
    # SERTEC_RST, ST3XX, ST4XX, QUECLINK (54 códigos, todos cobertos abaixo)
    "evt_falha_energia": (r"EXTERNAL POWER FAIL|(EXTERNAL|MAIN) POWER( SUPPLY)? DISCONNECT"
                          r"|DISCONNECTED WITH THE MAIN POWER|(^|_)POWER DISCONNECT|POWER SUPPLY DISCONNECTED"
                          r"|REMOVE MAIN POWER|EXTERNAL BATTERY REMOVED|POWER SUPPLY ERROR|MAIN POWER (LOST|OFF)"),
    "evt_energia_ok":    (r"EXTERNAL POWER OK|(EXTERNAL|MAIN) POWER( SUPPLY)? CONNECT"
                          r"|(?<!DIS)CONNECTED WITH THE MAIN POWER|(^|_)POWER CONNECT"
                          r"|EXTERNAL BATTERY CONNECTED|POWER RECONNECT|MAIN POWER (ON|OK)"),
    "evt_tensao_baixa":  (r"LOW BATTERY|BATTERY LOW|LOW BACKUP BATTERY|BATTERY VOLTAGE BELOW"
                          r"|BELOW PWR THRES|DUE TO LOW POWER|BATTERY ERROR"),
    "evt_bateria_backup_removida": (r"BACKUP BATTERY UNPLUGGED|INTERNAL BATTERY REMOVED"
                                    r"|DISCONNECTED WITH THE BACK-?UP BATTERY|(^|_)BATTERY DISCONNECT"),
    "evt_bateria_backup_ok": (r"BACKUP BATTERY PLUGGED|INTERNAL BATTERY CONNECTED"
                              r"|(?<!DIS)CONNECTED WITH THE BACK-?UP BATTERY|(^|_)BATTERY CONNECT"),
    "evt_antena_gps_removida": r"GPS ANTENNA (UNPLUGGED|DISCONNECT)|(^|_)DISCONNECTED GPS ANTENNA|GPS SIGNAL FAILURE",
    "evt_antena_gps_ok": r"GPS ANTENNA (PLUGGED|CONNECT)|RECONNECTED GPS ANTENNA",
    "evt_desligamento":  r"POWER OFF|SHUT ?DOWN|(^|_)DISCONNECT$",
    "evt_religamento":   r"POWER ON|WAKE UP|GPRS FIRST ATTACHED|REBOOT|RESTART",
    "evt_seguranca":     r"UNAUTHORIZED DRIVER|PASSWORD WRONG|SUSPICIOUS|TRUCK (DIS)?CONNECTED",
    "evt_reconstrucao":  r"RECONSTRUCTION|STORED|BUFFER|OFFLINE",
}

# Famílias que, sozinhas, já indicam adulteração física do equipamento
EVENTOS_CRITICOS = ["evt_falha_energia", "evt_bateria_backup_removida",
                    "evt_antena_gps_removida", "evt_seguranca"]


def relatorio_cobertura(colunas_motivo):
    """Mostra quais Reason Codes da base caem em cada família e quais ficaram de fora."""
    linhas = []
    for c in colunas_motivo:
        fams = [f for f, p in FAMILIAS_EVENTO.items() if re.search(p, c.upper())]
        linhas.append({"reason_code": c.replace("motivo_", ""), "familias": ", ".join(fams) or "-"})
    return pd.DataFrame(linhas)


def familias_evento(df):
    """Soma as colunas one-hot de Reason Code em famílias comparáveis entre firmwares."""
    motivos = [c for c in df.columns if c.startswith("motivo_")]
    out = {}
    for fam, padrao in FAMILIAS_EVENTO.items():
        cols = [c for c in motivos if re.search(padrao, c.upper())]
        out[fam] = (df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
                    if cols else pd.Series(0, index=df.index)).astype("int16")
    return pd.DataFrame(out, index=df.index)


def gsm_em_dbm(gsm):
    """Alguns firmwares mandam GSM em dBm (-113 a -51), outros em CSQ (0 a 31; 99 = desconhecido)."""
    g = gsm.where(gsm != 99)
    if (g.dropna() >= 0).mean() > 0.8:          # escala CSQ
        return -113 + 2 * g.clip(0, 31)
    return g


def _num(df, c):
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def haversine_vec(lat1, lon1, lat2, lon2, R=6371.0):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def duracao_continua(cond, ts):
    """Minutos desde o início da sequência ininterrupta em que cond é verdadeira."""
    cond = pd.Series(cond, index=ts.index).fillna(False).astype(bool)
    run = (cond != cond.shift()).cumsum()
    inicio = ts.groupby(run).transform("min")
    return ((ts - inicio).dt.total_seconds() / 60).where(cond, 0.0).astype("float32")


def rotulo_persistente(fraude, ts, dur_min):
    """Fraude = estado de jammer que dura pelo menos dur_min minutos."""
    fr = fraude.fillna(0).astype(int)
    run = (fr != fr.shift()).cumsum()
    fim = ts.shift(-1).fillna(ts)
    ini = ts.groupby(run).transform("min")
    fim_run = fim.groupby(run).transform("max")
    dur = ((fim_run - ini).dt.total_seconds() / 60).where(fr == 1)
    inicio_ep = (fr == 1) & (fr.shift(fill_value=0) != 1)
    return ((fr == 1) & (dur >= dur_min)).astype("int8"), dur, inicio_ep


def features_registro(df, P=PARAMS):
    """Features por registro, calculadas DENTRO de um único rastreador, em ordem temporal."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    ts = df["timestamp"]
    D = P["DURACAO_MIN_SUSPEITA"]
    delay, hdop, sat = _num(df, "Reception_delay_seg"), _num(df, "HDOP"), _num(df, "GPS Satellites")
    gsm, vext, bat = _num(df, "GSM Signal"), _num(df, "External Voltage"), _num(df, "Internal Battery")
    lat, lng, vel, ign = _num(df, "Lat"), _num(df, "Lng"), _num(df, "Speed"), _num(df, "Ignition")
    lat, lng = lat.where(lat != 0), lng.where(lng != 0)
    gsm = gsm_em_dbm(gsm)
    if bat.fillna(0).eq(0).all():               # rastreador sem bateria interna medida
        bat = pd.Series(np.nan, index=df.index)
    if sat.max() > 24:                          # alguns firmwares somam satélites visíveis de várias constelações
        sat = sat.where(sat > 0)
    fam = familias_evento(df)

    out = pd.DataFrame({"timestamp": ts})
    dt_min = ts.diff().dt.total_seconds() / 60
    out["dt_min"] = dt_min

    # Sinal: só a DURAÇÃO conta (um pico curto = interferência comum, não fraude)
    out["delay_alto"] = (delay > P["DELAY_ALTO_SEG"]).astype("int8")
    out["dur_delay_alto"] = duracao_continua(delay > P["DELAY_ALTO_SEG"], ts)
    out["dur_hdop_alto"] = duracao_continua(hdop >= P["HDOP_ALTO"], ts)
    out["dur_sem_gps"] = duracao_continua(sat < P["SATELITES_MIN"], ts)
    out["dur_gsm_fraco"] = duracao_continua(gsm <= P["GSM_FRACO_DBM"], ts)

    # Energia: tensão externa e bateria interna, relativas ao próprio rastreador
    # Estado da alimentação pelos EVENTOS (falha -> cortada, ok -> normal).
    # Usado quando o firmware não envia tensão externa naquele registro.
    est = pd.Series(np.nan, index=df.index)
    est[fam["evt_falha_energia"] > 0] = 1
    est[fam["evt_energia_ok"] > 0] = 0
    est.iloc[0] = 0 if pd.isna(est.iloc[0]) else est.iloc[0]
    est = est.ffill()
    # o estado por evento expira em 12 h sem confirmação (evita "corte eterno" se faltar o evento de volta)
    ult_evt = ts.where((fam["evt_falha_energia"] > 0) | (fam["evt_energia_ok"] > 0)).ffill()
    est = est.where((ts - ult_evt) <= pd.Timedelta(hours=12), 0)
    ext_cortada = (vext < P["TENSAO_EXT_MIN"]).where(vext.notna(), est == 1).astype(bool)
    out["ext_cortada"] = ext_cortada.astype("int8")
    out["dur_ext_cortada"] = duracao_continua(ext_cortada, ts)
    out["vext"] = vext
    out["tensao_rel"] = vext - vext.median()
    out["queda_tensao"] = (-vext.diff()).clip(lower=0)
    out["bat"] = bat
    out["bateria_rel"] = bat - bat.median()
    out["queda_bateria"] = (-bat.diff()).clip(lower=0)

    # Posição: deslocamento, salto após apagão e área habitual do próprio rastreador
    d = haversine_vec(lat.shift(), lng.shift(), lat, lng)
    out["dist_km"] = d
    out["salto_pos_gap_km"] = np.where(dt_min > P["GAP_TRANSMISSAO_MIN"], d, 0.0)
    g = P["GRADE_GRAUS"]
    ok = lat.notna() & lng.notna()
    celula = pd.Series("", index=df.index)
    celula[ok] = ((lat[ok] / g).round().astype(int).astype(str) + "_"
                  + (lng[ok] / g).round().astype(int).astype(str))
    out["freq_local"] = celula.map(celula[ok].value_counts(normalize=True)).where(ok).astype("float32")
    out["dist_mediana_km"] = haversine_vec(lat, lng, lat.median(), lng.median())
    out["parado"] = (vel < P["PARADO_KMH"]).astype("int8")
    out["speed"], out["ignicao"] = vel, ign
    out["hora_sin"], out["hora_cos"] = _num(df, "hora_sin"), _num(df, "hora_cos")

    # Indicadores compostos
    degradado = pd.concat([out["dur_delay_alto"], out["dur_hdop_alto"],
                           out["dur_gsm_fraco"], out["dur_sem_gps"]], axis=1).max(axis=1)
    local_suspeito = (out["freq_local"] < P["FREQ_LOCAL_RARO"]) | (out["salto_pos_gap_km"] > P["SALTO_KM"])
    out["sinal_degradado_persistente"] = (degradado >= D).astype("int8")
    out["regra_combinada"] = ((degradado >= D) & local_suspeito).astype("int8")
    out["regra_energia"] = (out["dur_ext_cortada"] >= D).astype("int8")

    # Violação de alimentação: energia externa cortada E veículo andando com ignição desligada.
    # Fisicamente incoerente: o carro roda 100 km/h "desligado" enquanto o rastreador
    # está na bateria de backup e só depois envia a rota (ROUTE'S RECONSTRUCTION).
    andando = vel > P["KMH_MOVIMENTO"]
    out["mov_sem_ignicao"] = (andando & (ign == 0)).astype("int8")
    out["mov_sem_ign_cortada"] = (andando & (ign == 0) & ext_cortada).astype("int8")
    for c in fam.columns:
        if not c.startswith("evt_jammer"):
            out[c] = fam[c].values

    # Episódios contínuos de energia cortada
    run_ext = (ext_cortada != ext_cortada.shift()).cumsum()
    n_mov_ep = out["mov_sem_ign_cortada"].groupby(run_ext).transform("sum")
    dur_ep_ext = out["dur_ext_cortada"].groupby(run_ext).transform("max")
    violacao = ext_cortada & ((n_mov_ep >= P["MIN_REG_MOV_SEM_IGN"]) | (dur_ep_ext >= D))
    out["regra_violacao_energia"] = violacao.astype("int8")

    # Adulteração física: antena GPS ou bateria de backup removida, evento de segurança,
    # ou falha de energia — só vira alarme se houver CONFIRMAÇÃO no comportamento
    # (energia cortada ou veículo andando com ignição desligada nas 2 h seguintes).
    critico = fam[[c for c in EVENTOS_CRITICOS if c in fam]].sum(axis=1) > 0
    confirma = (ext_cortada | (out["mov_sem_ignicao"] == 1)).astype(int)
    # olha 2 h para a frente: para cada registro, existe confirmação em [t, t + 2 h]?
    # (ts já está em ordem crescente; soma acumulada + busca binária)
    t_ns = ts.to_numpy("datetime64[ns]")
    fim = np.searchsorted(t_ns, t_ns + np.timedelta64(2, "h"), side="right")
    acum = np.concatenate([[0], np.cumsum(confirma.to_numpy())])
    confirma_2h = acum[fim] - acum[np.arange(len(t_ns))]
    out["regra_adulteracao"] = (critico & (confirma_2h > 0)).astype("int8")

    for c in df.columns:   # contagem de eventos brutos (sem os que definem o rótulo de jammer)
        if c.startswith("motivo_") and not re.search("JAMM", c.upper()):
            out["n_" + c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")

    # Rótulo de jammer recalculado aqui a partir das famílias (vale para todos os firmwares)
    estado = pd.Series(np.nan, index=df.index)
    estado[fam["evt_jammer_inicio"] > 0] = 1
    estado[fam["evt_jammer_fim"] > 0] = 0
    estado.iloc[0] = 0 if pd.isna(estado.iloc[0]) else estado.iloc[0]
    jam_estado = estado.ffill().astype(int)
    jam_estado[fam["evt_jammer_fim"] > 0] = 1       # o registro de saída ainda está no jammer
    rot_jam, out["_dur_jammer"], out["_ini_jammer"] = rotulo_persistente(
        jam_estado, ts, P["ROTULO_DURACAO_MIN"])

    modo = P.get("ROTULO_MODO", "ambos")
    rot_en = out["regra_violacao_energia"]
    out[TARGET] = {"jammer": rot_jam, "energia": rot_en}.get(
        modo, ((rot_jam == 1) | (rot_en == 1)).astype("int8"))
    out["_rot_jammer"], out["_rot_energia"] = rot_jam, rot_en
    out["rastreador"] = str(df["rastreador"].iloc[0]) if "rastreador" in df.columns else "desconhecido"
    out["lat"], out["lng"] = lat, lng
    return out


def agregar_janelas(r, P=PARAMS):
    """Resume os registros em janelas de P['JANELA_MIN'] minutos: 1 linha = 1 janela."""
    r = r.copy()
    r["janela"] = r["timestamp"].dt.floor(f"{P['JANELA_MIN']}min")
    g = r.groupby("janela")
    J = pd.DataFrame({
        "n_registros": g.size(),
        "maior_gap_min": g["dt_min"].max(),
        "frac_delay_alto": g["delay_alto"].mean(),
        "dur_max_delay_alto": g["dur_delay_alto"].max(),
        "dur_max_hdop_alto": g["dur_hdop_alto"].max(),
        "dur_max_sem_gps": g["dur_sem_gps"].max(),
        "dur_max_gsm_fraco": g["dur_gsm_fraco"].max(),
        "frac_ext_cortada": g["ext_cortada"].mean(),
        "dur_max_ext_cortada": g["dur_ext_cortada"].max(),
        "tensao_rel_min": g["tensao_rel"].min(),
        "tensao_rel_media": g["tensao_rel"].mean(),
        "tensao_std": g["vext"].std(),
        "queda_tensao_max": g["queda_tensao"].max(),
        "bateria_rel_min": g["bateria_rel"].min(),
        "bateria_std": g["bat"].std(),
        "queda_bateria_max": g["queda_bateria"].max(),
        "dist_total_km": g["dist_km"].sum(),
        "salto_max_km": g["salto_pos_gap_km"].max(),
        "freq_local_min": g["freq_local"].min(),
        "dist_mediana_max_km": g["dist_mediana_km"].max(),
        "frac_parado": g["parado"].mean(),
        "frac_ignicao": g["ignicao"].mean(),
        "vel_media": g["speed"].mean(),
        "vel_max": g["speed"].max(),
        "hora_sin": g["hora_sin"].mean(),
        "hora_cos": g["hora_cos"].mean(),
        "sinal_degradado_persistente": g["sinal_degradado_persistente"].max(),
        "regra_combinada": g["regra_combinada"].max(),
        "regra_energia": g["regra_energia"].max(),
        "regra_violacao_energia": g["regra_violacao_energia"].max(),
        "n_mov_sem_ignicao": g["mov_sem_ignicao"].sum(),
        "n_mov_sem_ign_cortada": g["mov_sem_ign_cortada"].sum(),
        "n_evt_falha_energia": g["evt_falha_energia"].sum(),
        "n_evt_energia_ok": g["evt_energia_ok"].sum(),
        "n_evt_desligamento": g["evt_desligamento"].sum(),
        "n_evt_religamento": g["evt_religamento"].sum(),
        "n_evt_reconstrucao": g["evt_reconstrucao"].sum(),
        "n_evt_tensao_baixa": g["evt_tensao_baixa"].sum(),
        "n_evt_bateria_backup_removida": g["evt_bateria_backup_removida"].sum(),
        "n_evt_bateria_backup_ok": g["evt_bateria_backup_ok"].sum(),
        "n_evt_antena_gps_removida": g["evt_antena_gps_removida"].sum(),
        "n_evt_antena_gps_ok": g["evt_antena_gps_ok"].sum(),
        "n_evt_seguranca": g["evt_seguranca"].sum(),
        "regra_adulteracao": g["regra_adulteracao"].max(),
        "_rot_jammer": g["_rot_jammer"].max(),
        "_rot_energia": g["_rot_energia"].max(),
        "lat": g["lat"].median(),
        "lng": g["lng"].median(),
        TARGET: g[TARGET].max(),
        "_dur_jammer_max": g["_dur_jammer"].max(),
    })
    n_cols = [c for c in r.columns if c.startswith("n_motivo_")]
    if n_cols:
        J = J.join(g[n_cols].sum())
    J["n_registros_rel"] = J["n_registros"] / J["n_registros"].median()
    J = J.reset_index().rename(columns={"janela": "inicio_janela"})
    J["rastreador"] = r["rastreador"].iloc[0]
    return J


NAO_FEATURES = {"inicio_janela", "rastreador", "arquivo", "lat", "lng", TARGET}

# Variáveis que definem o rótulo de energia ficam fora do treino para o modelo
# não apenas reaprender a regra de rotulação.
DEFINEM_ROTULO_ENERGIA = {"regra_violacao_energia", "n_mov_sem_ign_cortada"}

def lista_features(J, P=PARAMS):
    excl = set(NAO_FEATURES)
    if P.get("ROTULO_MODO", "ambos") in ("energia", "ambos"):
        excl |= DEFINEM_ROTULO_ENERGIA
    return [c for c in J.columns
            if c not in excl and not c.startswith("_")
            and not any(p in c.upper() for p in VAZAMENTO_PATTERNS)]


# Decisão combinada (veredito do arquivo)
# Cada indício isolado NUNCA basta para declarar fraude. Um episódio só é fraude se
# reunir pelo menos MIN_EVIDENCIAS categorias diferentes de indício E a confiança
# combinada (modelo + indícios) atingir RISCO_MIN.
DECISAO = dict(
    RISCO_MIN=0.50,              # confiança mínima para declarar FRAUDE
    MIN_EVIDENCIAS=2,            # nº mínimo de categorias diferentes no mesmo episódio
    TOLERANCIA_JANELAS=1,        # janelas sem indício permitidas dentro de um episódio
    GAP_APAGAO_MIN=180,          # apagão de transmissão (min)
    # peso dos indícios combinados (nº de categorias -> probabilidade "de regra")
    PESO_EVIDENCIAS={2: 0.30, 3: 0.45, 4: 0.80},   # 2–3 indícios precisam do modelo; 4+ bastam
    # os indícios precisam ocorrer JUNTOS: na mesma janela ou nas vizinhas (±1 janela).
    # Indícios espalhados em dias diferentes NÃO se somam.
)

NOMES_EVIDENCIA = {
    "ev_energia": "energia externa cortada / falha ou tensão baixa",
    "ev_movimento_incoerente": "veículo andando com ignição desligada",
    "ev_desligamento": "equipamento desligado / religado",
    "ev_adulteracao": "antena GPS ou bateria de backup removida / evento de segurança",
    "ev_sinal_persistente": "sinal degradado por 30 min ou mais",
    "ev_jammer_persistente": "jammer por 30 min ou mais",
    "ev_posicao": "local incomum ou salto de posição após apagão",
    "ev_apagao": "apagão longo de transmissão",
}


def _col(J, c, padrao=0):
    return J[c].fillna(padrao) if c in J.columns else pd.Series(padrao, index=J.index)


def evidencias_janela(J, P=PARAMS, D=DECISAO):
    """Uma coluna booleana por CATEGORIA de indício, para cada janela."""
    return pd.DataFrame({
        # >= 2 leituras de tensão cortada: 1 leitura zerada isolada costuma ser falha de leitura
        "ev_energia": (_col(J, "frac_ext_cortada") * _col(J, "n_registros") >= 2)
                      | (_col(J, "n_evt_falha_energia") > 0)
                      | (_col(J, "n_evt_tensao_baixa") > 0),
        "ev_movimento_incoerente": _col(J, "n_mov_sem_ignicao") >= P["MIN_REG_MOV_SEM_IGN"],
        "ev_desligamento": (_col(J, "n_evt_desligamento") > 0) | (_col(J, "n_evt_religamento") > 0),
        "ev_adulteracao": (_col(J, "n_evt_antena_gps_removida") + _col(J, "n_evt_bateria_backup_removida")
                           + _col(J, "n_evt_seguranca")) > 0,
        # parado na garagem sem GPS é normal: sinal só conta com o veículo em movimento
        "ev_sinal_persistente": (_col(J, "sinal_degradado_persistente") > 0)
                                & (_col(J, "vel_max") > P["KMH_MOVIMENTO"]),
        "ev_jammer_persistente": _col(J, "_rot_jammer") > 0,
        "ev_posicao": (_col(J, "freq_local_min", 1.0) < P["FREQ_LOCAL_RARO"])
                      | (_col(J, "salto_max_km") > P["SALTO_KM"]),
        "ev_apagao": _col(J, "maior_gap_min") > D["GAP_APAGAO_MIN"],
    }, index=J.index)


def decisao_combinada(J, prob, P=PARAMS, D=DECISAO):
    """
    Agrupa janelas com indícios em episódios e decide cada episódio.
    confiança = 1 - (1 - p_modelo) * (1 - p_indicios)   se n_categorias >= MIN_EVIDENCIAS
              = no máximo RISCO_MIN - 1%                 se houver só 1 categoria (nunca é fraude)
    Retorna (veredito, confiança_fraude, tabela_de_episódios, evidências por janela).
    """
    J = J.reset_index(drop=True)
    prob = np.nan_to_num(np.asarray(prob, dtype=float))
    ev = evidencias_janela(J, P, D)
    tem = ev.any(axis=1).to_numpy()
    passo = pd.Timedelta(minutes=P["JANELA_MIN"])
    episodio = np.full(len(J), -1)
    idx = np.where(tem)[0]
    if len(idx):
        t = J["inicio_janela"].iloc[idx]
        episodio[idx] = (t.diff() > passo * (1 + D["TOLERANCIA_JANELAS"])).cumsum().to_numpy()
    # co-ocorrência local: categorias da janela + vizinhas imediatas
    t_all = J["inicio_janela"]
    ant_ok = ((t_all - t_all.shift(1)) == passo).to_numpy()
    prox_ok = ((t_all.shift(-1) - t_all) == passo).to_numpy()
    E = ev.to_numpy(dtype=bool)
    loc = E.copy()
    loc[1:] |= E[:-1] & ant_ok[1:, None]
    loc[:-1] |= E[1:] & prox_ok[:-1, None]
    n_loc = loc.sum(axis=1)
    teto = D["RISCO_MIN"] - 0.01
    linhas = []
    for e in np.unique(episodio[episodio >= 0]):
        m = episodio == e
        i_best = np.where(m)[0][np.argmax(n_loc[m])]
        cats = [c for j, c in enumerate(ev.columns) if loc[i_best, j]]
        n = len(cats)
        p_mod = float(prob[m].max())
        p_ind = D["PESO_EVIDENCIAS"].get(min(n, max(D["PESO_EVIDENCIAS"])), 0.0) if n >= 2 else 0.0
        conf = 1 - (1 - p_mod) * (1 - p_ind) if n >= D["MIN_EVIDENCIAS"] else min(p_mod, teto)
        ini = J.loc[m, "inicio_janela"].min()
        fim = J.loc[m, "inicio_janela"].max() + passo
        linhas.append(dict(inicio=ini, fim=fim, duracao_h=(fim - ini).total_seconds() / 3600,
                           n_indicios=n, indicios="; ".join(NOMES_EVIDENCIA[c] for c in cats),
                           prob_modelo=round(p_mod, 3), confianca=round(conf, 3),
                           fraude=conf >= D["RISCO_MIN"],
                           lat=J.loc[m, "lat"].median(), lng=J.loc[m, "lng"].median()))
    ep = pd.DataFrame(linhas, columns=["inicio", "fim", "duracao_h", "n_indicios", "indicios",
                                       "prob_modelo", "confianca", "fraude", "lat", "lng"])
    if len(ep):
        conf_arquivo = float(ep["confianca"].max())
    else:
        conf_arquivo = float(min(prob.max() if len(prob) else 0.0, teto))
    veredito = "FRAUDE" if conf_arquivo >= D["RISCO_MIN"] else "NÃO FRAUDE"
    ep = ep.sort_values("confianca", ascending=False).reset_index(drop=True)
    return veredito, conf_arquivo, ep, ev
