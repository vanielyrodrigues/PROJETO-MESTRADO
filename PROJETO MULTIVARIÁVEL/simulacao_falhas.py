# simulacao_falhas.py

import numpy as np
import pandas as pd

LABEL_NORMAL = "normal"
LABEL_STUCK = "stuck"
LABEL_STUCK_ZERO = "stuck_at_zero"
LABEL_LACUNA = "lacuna"
LABEL_QUEDA = "queda"
LABEL_OSC = "oscilacao"


def _to_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _garantir_colunas_controle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = LABEL_NORMAL
    if "evento_id" not in df.columns:
        df["evento_id"] = ""
    return df


def _marcar_intervalo(df: pd.DataFrame, inicio, fim, label: str, evento_id: str | None = None) -> None:
    m = (df["Datetime"] >= inicio) & (df["Datetime"] <= fim)
    df.loc[m, "label"] = label
    if evento_id is not None:
        df.loc[m, "evento_id"] = evento_id


def preparar_base(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", dayfirst=True)

    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    if col in df.columns:
        df[col] = _to_numeric_series(df[col])

    df["label"] = LABEL_NORMAL
    df["evento_id"] = ""
    return df


def _janela_livre(df: pd.DataFrame, i0: int, i1: int) -> bool:
    if i0 < 0 or i1 > len(df) or i0 >= i1:
        return False
    return (df.loc[i0:i1 - 1, "label"] == LABEL_NORMAL).all()


def _trecho_tem_dados(df: pd.DataFrame, col: str, i0: int, i1: int, frac_min_valida: float = 0.6) -> bool:
    trecho = _to_numeric_series(df.loc[i0:i1 - 1, col])
    validos = trecho.notna().sum()
    minimo = max(3, int((i1 - i0) * frac_min_valida))
    return validos >= minimo


def _sortear_int_param(valor, rng: np.random.Generator) -> int:
    """Aceita inteiro fixo ou intervalo (min, max) e retorna valor aleatório."""
    if isinstance(valor, (list, tuple)) and len(valor) == 2:
        a, b = int(valor[0]), int(valor[1])
        return int(rng.integers(min(a, b), max(a, b) + 1))
    return int(valor)


def _sortear_float_param(valor, rng: np.random.Generator) -> float:
    """Aceita float fixo ou intervalo (min, max) e retorna valor aleatório."""
    if isinstance(valor, (list, tuple)) and len(valor) == 2:
        a, b = float(valor[0]), float(valor[1])
        return float(rng.uniform(min(a, b), max(a, b)))
    return float(valor)


def _gerar_inicios_aleatorios(
    n_total: int,
    duracao_max: int,
    n_eventos: int,
    margem: int = 0,
    rng: np.random.Generator | None = None
) -> list[int]:
    """Gera posições iniciais aleatórias, evitando concentração sempre nos mesmos pontos."""
    if rng is None:
        rng = np.random.default_rng()

    if n_total <= duracao_max + 2 or n_eventos <= 0:
        return []

    ini_min = max(0, margem)
    ini_max = max(ini_min + 1, n_total - duracao_max - margem)
    candidatos = np.arange(ini_min, ini_max)
    rng.shuffle(candidatos)

    return [int(x) for x in candidatos[: max(1, min(n_eventos * 8, len(candidatos)))]]


def _ajustar_inicio_para_janela_livre(
    df: pd.DataFrame,
    col: str,
    i0_base: int,
    duracao_pts: int,
    max_desloc: int = 20
):
    candidatos = [i0_base]

    for d in range(1, max_desloc + 1):
        candidatos.append(i0_base - d)
        candidatos.append(i0_base + d)

    for i0 in candidatos:
        i1 = i0 + duracao_pts

        if i0 < 0 or i1 > len(df):
            continue

        if not _janela_livre(df, i0, i1):
            continue

        if not _trecho_tem_dados(df, col, i0, i1):
            continue

        return i0, i1

    return None


def _inj_stuck_local(
    df: pd.DataFrame,
    col: str,
    i0: int,
    i1: int,
    label: str,
    evento_id: str | None = None
) -> pd.DataFrame:
    df = _garantir_colunas_controle(df)
    base = _to_numeric_series(df.loc[i0:i1 - 1, col]).dropna()

    if len(base) == 0:
        return df

    valor = float(base.iloc[0])
    df.loc[i0:i1 - 1, col] = valor

    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label, evento_id)
    return df


def _inj_stuck_zero_local(
    df: pd.DataFrame,
    col: str,
    i0: int,
    i1: int,
    label: str,
    evento_id: str | None = None
) -> pd.DataFrame:
    df = _garantir_colunas_controle(df)
    df.loc[i0:i1 - 1, col] = 0.0

    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label, evento_id)
    return df


def _inj_queda_local(
    df: pd.DataFrame,
    col: str,
    i0: int,
    i1: int,
    label: str,
    delta: float | tuple = (-12.0, -4.0),
    evento_id: str | None = None,
    rng: np.random.Generator | None = None
) -> pd.DataFrame:
    df = _garantir_colunas_controle(df)
    base = _to_numeric_series(df.loc[i0:i1 - 1, col]).to_numpy(dtype=float)

    if np.isnan(base).all():
        return df

    if rng is None:
        rng = np.random.default_rng()

    med = np.nanmedian(base)
    base = np.where(np.isnan(base), med, base)

    delta_evento = _sortear_float_param(delta, rng)

    # Queda sempre deve reduzir a leitura.
    # Se algum valor positivo for enviado por engano, ele é convertido para negativo.
    delta_evento = -abs(delta_evento)

    df.loc[i0:i1 - 1, col] = np.maximum(base + delta_evento, 0.0)

    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label, evento_id)
    return df


def _inj_oscilacao_local(
    df: pd.DataFrame,
    col: str,
    i0: int,
    i1: int,
    label: str,
    amp: float | tuple = (2.0, 4.0),
    evento_id: str | None = None,
    rng: np.random.Generator | None = None
) -> pd.DataFrame:
    df = _garantir_colunas_controle(df)
    duracao_pts = i1 - i0

    base = _to_numeric_series(df.loc[i0:i1 - 1, col]).to_numpy(dtype=float)

    if np.isnan(base).all():
        return df

    if rng is None:
        rng = np.random.default_rng()

    med = np.nanmedian(base)
    base = np.where(np.isnan(base), med, base)

    # Oscilação reduzida e aleatória para evitar padrão artificial.
    amp_evento = _sortear_float_param(amp, rng)

    periodo = int(rng.integers(5, 9))
    fase = float(rng.uniform(0, 2 * np.pi))

    t = np.arange(duracao_pts)

    sinal_senoidal = amp_evento * np.sin(2 * np.pi * t / periodo + fase)
    ruido = rng.normal(0, amp_evento * 0.20, size=duracao_pts)

    sinal = sinal_senoidal + ruido

    df.loc[i0:i1 - 1, col] = np.maximum(base + sinal, 0.0)

    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label, evento_id)
    return df


def _inj_lacuna_local(
    df: pd.DataFrame,
    col: str,
    i0: int,
    i1: int,
    label: str,
    evento_id: str | None = None
) -> pd.DataFrame:
    df = _garantir_colunas_controle(df)

    # A lacuna permanece como NaN para aparecer visualmente como ausência de leitura.
    df.loc[i0:i1 - 1, col] = np.nan

    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label, evento_id)
    return df


def _injetar_eventos_espalhados(
    df: pd.DataFrame,
    col: str,
    label: str,
    duracao_pts: int | tuple,
    n_eventos: int,
    func_injecao,
    rng: np.random.Generator | None = None,
    margem: int = 30
):
    df = _garantir_colunas_controle(df).sort_values("Datetime").reset_index(drop=True)

    if rng is None:
        rng = np.random.default_rng()

    duracao_max = max(duracao_pts) if isinstance(duracao_pts, (list, tuple)) else int(duracao_pts)

    inicios = _gerar_inicios_aleatorios(
        len(df),
        duracao_max,
        n_eventos,
        margem=margem,
        rng=rng
    )

    eventos = []
    ordem = 0

    for i0_base in inicios:
        if ordem >= n_eventos:
            break

        dur_evento = _sortear_int_param(duracao_pts, rng)

        janela = _ajustar_inicio_para_janela_livre(
            df,
            col,
            i0_base,
            dur_evento,
            max_desloc=40
        )

        if janela is None:
            continue

        i0, i1 = janela
        ordem += 1
        evento_id = f"{label}_{ordem:03d}"

        df = func_injecao(df, col, i0, i1, label, evento_id, rng)

        eventos.append({
            "evento_id": evento_id,
            "falha": label,
            "inicio": df.loc[i0, "Datetime"],
            "fim": df.loc[i1 - 1, "Datetime"],
            "duracao_pts": int(i1 - i0)
        })

    return df, eventos


def injetar_stuck(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 25,
    valor=None,
    seed: int = 42,
    label: str = LABEL_STUCK
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _garantir_colunas_controle(df).sort_values("Datetime").reset_index(drop=True)

    if len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts

    base = _to_numeric_series(df.loc[i0:i1 - 1, col]).dropna()

    if len(base) == 0:
        return df

    if valor is None:
        valor = float(base.iloc[0])

    evento_id = f"{label}_manual"
    df.loc[i0:i1 - 1, col] = float(valor)

    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label, evento_id)
    return df


def injetar_stuck_zero(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 20,
    seed: int = 43
) -> pd.DataFrame:
    return injetar_stuck(
        df=df,
        col=col,
        duracao_pts=duracao_pts,
        valor=0.0,
        seed=seed,
        label=LABEL_STUCK_ZERO
    )


def injetar_queda(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 20,
    delta: float | tuple = (-12.0, -4.0),
    seed: int = 44
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _garantir_colunas_controle(df).sort_values("Datetime").reset_index(drop=True)

    if len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts

    return _inj_queda_local(
        df,
        col,
        i0,
        i1,
        LABEL_QUEDA,
        delta=delta,
        evento_id="queda_manual",
        rng=rng
    )


def injetar_oscilacao(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 30,
    amp: float | tuple = (2.0, 4.0),
    seed: int = 45
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _garantir_colunas_controle(df).sort_values("Datetime").reset_index(drop=True)

    if len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts

    return _inj_oscilacao_local(
        df,
        col,
        i0,
        i1,
        LABEL_OSC,
        amp=amp,
        evento_id="oscilacao_manual",
        rng=rng
    )


def injetar_lacuna(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 36,
    seed: int = 46
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _garantir_colunas_controle(df).sort_values("Datetime").reset_index(drop=True)

    if len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts

    return _inj_lacuna_local(
        df,
        col,
        i0,
        i1,
        LABEL_LACUNA,
        evento_id="lacuna_manual"
    )


def injetar_intervalo_por_tempo(
    df: pd.DataFrame,
    col: str,
    inicio,
    fim,
    modo: str,
    amp: float | tuple = (2.0, 4.0),
    delta: float | tuple = (-12.0, -4.0),
    evento_id: str | None = None
) -> pd.DataFrame:
    df = _garantir_colunas_controle(df).sort_values("Datetime").reset_index(drop=True)

    m = (df["Datetime"] >= inicio) & (df["Datetime"] <= fim)

    if m.sum() == 0:
        return df

    idx = df.index[m]
    i0, i1 = int(idx[0]), int(idx[-1]) + 1

    if evento_id is None:
        evento_id = f"{modo}_manual_{pd.Timestamp(inicio).strftime('%Y%m%d%H%M')}"

    if modo == LABEL_STUCK:
        return _inj_stuck_local(df, col, i0, i1, LABEL_STUCK, evento_id)

    if modo == LABEL_STUCK_ZERO:
        return _inj_stuck_zero_local(df, col, i0, i1, LABEL_STUCK_ZERO, evento_id)

    if modo == LABEL_OSC:
        return _inj_oscilacao_local(
            df,
            col,
            i0,
            i1,
            LABEL_OSC,
            amp=amp,
            evento_id=evento_id
        )

    if modo == LABEL_QUEDA:
        return _inj_queda_local(
            df,
            col,
            i0,
            i1,
            LABEL_QUEDA,
            delta=delta,
            evento_id=evento_id
        )

    if modo == LABEL_LACUNA:
        return _inj_lacuna_local(df, col, i0, i1, LABEL_LACUNA, evento_id)

    return df


def balancear_falhas(
    df: pd.DataFrame,
    col: str,
    config: dict | None = None,
    return_log: bool = False,
    seed: int | None = None
):
    df = _garantir_colunas_controle(df).sort_values("Datetime").reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if config is None:
        config = {
            LABEL_OSC: {
                "duracao_pts": (30, 60),
                "n_eventos": 10,
                "amp": (2.0, 4.0)
            },

            LABEL_STUCK: {
                "duracao_pts": 25,
                "n_eventos": 10
            },

            LABEL_STUCK_ZERO: {
                "duracao_pts": 20,
                "n_eventos": 10
            },

            LABEL_LACUNA: {
                "duracao_pts": (36, 72),
                "n_eventos": 10
            },

            LABEL_QUEDA: {
                "duracao_pts": (18, 36),
                "n_eventos": 10,
                "delta": (-12.0, -4.0)
            },
        }

    eventos_gerados = []

    if LABEL_OSC in config:
        p = config[LABEL_OSC]
        df, eventos = _injetar_eventos_espalhados(
            df,
            col,
            LABEL_OSC,
            p["duracao_pts"],
            p["n_eventos"],
            lambda d, c, i0, i1, lab, eid, r: _inj_oscilacao_local(
                d,
                c,
                i0,
                i1,
                lab,
                amp=p.get("amp", (2.0, 4.0)),
                evento_id=eid,
                rng=r
            ),
            rng=rng
        )
        eventos_gerados.extend(eventos)

    if LABEL_STUCK in config:
        p = config[LABEL_STUCK]
        df, eventos = _injetar_eventos_espalhados(
            df,
            col,
            LABEL_STUCK,
            p["duracao_pts"],
            p["n_eventos"],
            lambda d, c, i0, i1, lab, eid, r: _inj_stuck_local(
                d,
                c,
                i0,
                i1,
                lab,
                evento_id=eid
            ),
            rng=rng
        )
        eventos_gerados.extend(eventos)

    if LABEL_STUCK_ZERO in config:
        p = config[LABEL_STUCK_ZERO]
        df, eventos = _injetar_eventos_espalhados(
            df,
            col,
            LABEL_STUCK_ZERO,
            p["duracao_pts"],
            p["n_eventos"],
            lambda d, c, i0, i1, lab, eid, r: _inj_stuck_zero_local(
                d,
                c,
                i0,
                i1,
                lab,
                evento_id=eid
            ),
            rng=rng
        )
        eventos_gerados.extend(eventos)

    if LABEL_LACUNA in config:
        p = config[LABEL_LACUNA]
        df, eventos = _injetar_eventos_espalhados(
            df,
            col,
            LABEL_LACUNA,
            p["duracao_pts"],
            p["n_eventos"],
            lambda d, c, i0, i1, lab, eid, r: _inj_lacuna_local(
                d,
                c,
                i0,
                i1,
                lab,
                evento_id=eid
            ),
            rng=rng
        )
        eventos_gerados.extend(eventos)

    if LABEL_QUEDA in config:
        p = config[LABEL_QUEDA]
        df, eventos = _injetar_eventos_espalhados(
            df,
            col,
            LABEL_QUEDA,
            p["duracao_pts"],
            p["n_eventos"],
            lambda d, c, i0, i1, lab, eid, r: _inj_queda_local(
                d,
                c,
                i0,
                i1,
                lab,
                delta=p.get("delta", (-12.0, -4.0)),
                evento_id=eid,
                rng=r
            ),
            rng=rng
        )
        eventos_gerados.extend(eventos)

    if return_log:
        return df, pd.DataFrame(eventos_gerados)

    return df


def contar_eventos_rotulados(df: pd.DataFrame, label: str) -> int:
    """
    Conta eventos contínuos no sinal original.
    Exemplo: várias linhas seguidas como 'queda' contam como apenas 1 evento.
    """
    labels = df["label"].astype(str).tolist()
    normal = True
    cont = 0

    for lab in labels:
        if lab == label:
            if normal:
                cont += 1
                normal = False
        else:
            normal = True

    return cont


def resumo_eventos_injetados(df: pd.DataFrame, falhas=None) -> pd.DataFrame:
    if falhas is None:
        falhas = [
            LABEL_OSC,
            LABEL_LACUNA,
            LABEL_QUEDA,
            LABEL_STUCK,
            LABEL_STUCK_ZERO
        ]

    rows = []

    for falha in falhas:
        pontos = int((df["label"].astype(str) == falha).sum())
        eventos = contar_eventos_rotulados(df, falha)

        rows.append({
            "falha": falha,
            "eventos_continuos": eventos,
            "pontos_com_erro": pontos
        })

    return pd.DataFrame(rows)