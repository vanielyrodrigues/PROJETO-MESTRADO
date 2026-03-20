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


def _marcar_intervalo(df: pd.DataFrame, inicio, fim, label: str) -> None:
    m = (df["Datetime"] >= inicio) & (df["Datetime"] <= fim)
    df.loc[m, "label"] = label


def preparar_base(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", dayfirst=True)

    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    if col in df.columns:
        df[col] = _to_numeric_series(df[col])

    df["label"] = LABEL_NORMAL
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


def _gerar_inicios_espalhados(
    n_total: int,
    duracao_pts: int,
    n_eventos: int,
    margem: int = 0
) -> list[int]:
    if n_total <= duracao_pts + 2 or n_eventos <= 0:
        return []

    ini_min = margem
    ini_max = n_total - duracao_pts - margem
    if ini_max <= ini_min:
        ini_min = 0
        ini_max = n_total - duracao_pts

    if ini_max <= ini_min:
        return []

    posicoes = np.linspace(ini_min, ini_max, n_eventos, dtype=int)
    return sorted(set(int(p) for p in posicoes))


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


def _injetar_eventos_espalhados(
    df: pd.DataFrame,
    col: str,
    label: str,
    duracao_pts: int,
    n_eventos: int,
    func_injecao
) -> pd.DataFrame:
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    inicios = _gerar_inicios_espalhados(len(df), duracao_pts, n_eventos, margem=duracao_pts)

    for i0_base in inicios:
        janela = _ajustar_inicio_para_janela_livre(df, col, i0_base, duracao_pts, max_desloc=30)
        if janela is None:
            continue
        i0, i1 = janela
        df = func_injecao(df, col, i0, i1, label)

    return df


def _inj_stuck_local(df: pd.DataFrame, col: str, i0: int, i1: int, label: str) -> pd.DataFrame:
    df = df.copy()
    base = _to_numeric_series(df.loc[i0:i1 - 1, col]).dropna()
    if len(base) == 0:
        return df

    valor = float(base.iloc[0])
    df.loc[i0:i1 - 1, col] = valor
    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label)
    return df


def _inj_stuck_zero_local(df: pd.DataFrame, col: str, i0: int, i1: int, label: str) -> pd.DataFrame:
    df = df.copy()
    df.loc[i0:i1 - 1, col] = 0.0
    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label)
    return df


def _inj_queda_local(
    df: pd.DataFrame,
    col: str,
    i0: int,
    i1: int,
    label: str,
    delta: float = -15.0
) -> pd.DataFrame:
    df = df.copy()
    base = _to_numeric_series(df.loc[i0:i1 - 1, col]).to_numpy(dtype=float)
    if np.isnan(base).all():
        return df

    med = np.nanmedian(base)
    base = np.where(np.isnan(base), med, base)
    df.loc[i0:i1 - 1, col] = base + float(delta)
    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label)
    return df


def _inj_oscilacao_local(
    df: pd.DataFrame,
    col: str,
    i0: int,
    i1: int,
    label: str,
    amp: float = 10.0
) -> pd.DataFrame:
    df = df.copy()
    duracao_pts = i1 - i0
    base = _to_numeric_series(df.loc[i0:i1 - 1, col]).to_numpy(dtype=float)
    if np.isnan(base).all():
        return df

    med = np.nanmedian(base)
    base = np.where(np.isnan(base), med, base)

    t = np.arange(duracao_pts)
    sinal = amp * np.sin(2 * np.pi * t / 4.0)

    df.loc[i0:i1 - 1, col] = base + sinal
    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label)
    return df


def _inj_lacuna_local(df: pd.DataFrame, col: str, i0: int, i1: int, label: str) -> pd.DataFrame:
    df = df.copy()
    df.loc[i0:i1 - 1, col] = np.nan
    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label)
    return df


def injetar_stuck(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 25,
    valor=None,
    seed: int = 42,
    label: str = LABEL_STUCK
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    if len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts

    base = _to_numeric_series(df.loc[i0:i1 - 1, col]).dropna()
    if len(base) == 0:
        return df

    if valor is None:
        valor = float(base.iloc[0])

    df.loc[i0:i1 - 1, col] = float(valor)
    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label)
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
    duracao_pts: int = 10,
    delta: float = -15.0,
    seed: int = 44
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    if len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts
    return _inj_queda_local(df, col, i0, i1, LABEL_QUEDA, delta=delta)


def injetar_oscilacao(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 30,
    amp: float = 10.0,
    seed: int = 45
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    if len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts
    return _inj_oscilacao_local(df, col, i0, i1, LABEL_OSC, amp=amp)


def injetar_lacuna(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 15,
    seed: int = 46
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    if len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts
    return _inj_lacuna_local(df, col, i0, i1, LABEL_LACUNA)


def injetar_intervalo_por_tempo(
    df: pd.DataFrame,
    col: str,
    inicio,
    fim,
    modo: str,
    amp: float = 8.0,
    delta: float = -12.0
) -> pd.DataFrame:
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    m = (df["Datetime"] >= inicio) & (df["Datetime"] <= fim)
    if m.sum() == 0:
        return df

    idx = df.index[m]
    i0, i1 = int(idx[0]), int(idx[-1]) + 1

    if modo == LABEL_STUCK:
        return _inj_stuck_local(df, col, i0, i1, LABEL_STUCK)

    if modo == LABEL_STUCK_ZERO:
        return _inj_stuck_zero_local(df, col, i0, i1, LABEL_STUCK_ZERO)

    if modo == LABEL_OSC:
        return _inj_oscilacao_local(df, col, i0, i1, LABEL_OSC, amp=amp)

    if modo == LABEL_QUEDA:
        return _inj_queda_local(df, col, i0, i1, LABEL_QUEDA, delta=delta)

    if modo == LABEL_LACUNA:
        return _inj_lacuna_local(df, col, i0, i1, LABEL_LACUNA)

    return df


def balancear_falhas(
    df: pd.DataFrame,
    col: str,
    config: dict | None = None
) -> pd.DataFrame:
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    if config is None:
        config = {
            LABEL_OSC: {"duracao_pts": 30, "n_eventos": 10, "amp": 10.0},
            LABEL_STUCK: {"duracao_pts": 25, "n_eventos": 10},
            LABEL_STUCK_ZERO: {"duracao_pts": 20, "n_eventos": 10},
            LABEL_LACUNA: {"duracao_pts": 15, "n_eventos": 10},
            LABEL_QUEDA: {"duracao_pts": 12, "n_eventos": 10, "delta": -15.0},
        }

    if LABEL_OSC in config:
        p = config[LABEL_OSC]
        df = _injetar_eventos_espalhados(
            df, col, LABEL_OSC, p["duracao_pts"], p["n_eventos"],
            lambda d, c, i0, i1, lab: _inj_oscilacao_local(
                d, c, i0, i1, lab, amp=float(p.get("amp", 10.0))
            )
        )

    if LABEL_STUCK in config:
        p = config[LABEL_STUCK]
        df = _injetar_eventos_espalhados(
            df, col, LABEL_STUCK, p["duracao_pts"], p["n_eventos"],
            lambda d, c, i0, i1, lab: _inj_stuck_local(d, c, i0, i1, lab)
        )

    if LABEL_STUCK_ZERO in config:
        p = config[LABEL_STUCK_ZERO]
        df = _injetar_eventos_espalhados(
            df, col, LABEL_STUCK_ZERO, p["duracao_pts"], p["n_eventos"],
            lambda d, c, i0, i1, lab: _inj_stuck_zero_local(d, c, i0, i1, lab)
        )

    if LABEL_LACUNA in config:
        p = config[LABEL_LACUNA]
        df = _injetar_eventos_espalhados(
            df, col, LABEL_LACUNA, p["duracao_pts"], p["n_eventos"],
            lambda d, c, i0, i1, lab: _inj_lacuna_local(d, c, i0, i1, lab)
        )

    if LABEL_QUEDA in config:
        p = config[LABEL_QUEDA]
        df = _injetar_eventos_espalhados(
            df, col, LABEL_QUEDA, p["duracao_pts"], p["n_eventos"],
            lambda d, c, i0, i1, lab: _inj_queda_local(
                d, c, i0, i1, lab, delta=float(p.get("delta", -15.0))
            )
        )

    return df