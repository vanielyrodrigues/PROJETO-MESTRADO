# simulacao_falhas.py
import numpy as np
import pandas as pd

LABEL_NORMAL = "normal"
LABEL_STUCK = "stuck"
LABEL_STUCK_ZERO = "stuck_at_zero"
LABEL_LACUNA = "lacuna"
LABEL_QUEDA = "queda"
LABEL_OSC = "oscilacao"


def _marcar_intervalo(df: pd.DataFrame, inicio, fim, label: str) -> None:
    """
    Marca como 'label' todas as linhas cujo Datetime esteja no intervalo [inicio, fim].
    """
    m = (df["Datetime"] >= inicio) & (df["Datetime"] <= fim)
    df.loc[m, "label"] = label


def preparar_base(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Garante:
      - ordenação por tempo
      - coluna alvo numérica (quando possível)
      - label padrão = normal
    """
    df = df.copy()

    # garante coluna Datetime em datetime
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", dayfirst=True)

    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    # tenta converter a coluna alvo para float (sem forçar erro)
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label"] = LABEL_NORMAL
    return df


def injetar_stuck(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 30,
    valor=None,
    seed: int = 42,
    label: str = LABEL_STUCK
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)

    if col not in df.columns or len(df) <= duracao_pts + 2:
        return df

    # escolhe janela
    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts

    # valor padrão = valor no início da janela
    if valor is None:
        valor = float(df.loc[i0, col])

    df.loc[i0:i1 - 1, col] = float(valor)
    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], label)
    return df


def injetar_stuck_zero(df: pd.DataFrame, col: str, duracao_pts: int = 30, seed: int = 43) -> pd.DataFrame:
    return injetar_stuck(df, col, duracao_pts=duracao_pts, valor=0.0, seed=seed, label=LABEL_STUCK_ZERO)


def injetar_queda(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 10,
    delta: float = -15.0,
    seed: int = 44
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)

    if col not in df.columns or len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts

    base = pd.to_numeric(df.loc[i0:i1 - 1, col], errors="coerce").to_numpy(dtype=float)
    df.loc[i0:i1 - 1, col] = base + float(delta)

    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], LABEL_QUEDA)
    return df


def injetar_oscilacao(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 40,
    amp: float = 10.0,
    seed: int = 45
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)

    if col not in df.columns or len(df) <= duracao_pts + 2:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts

    # alternância rápida (+amp, -amp, +amp, -amp...)
    t = np.arange(duracao_pts)
    sinal = amp * np.sign(np.sin(2 * np.pi * t / 4.0))

    base = pd.to_numeric(df.loc[i0:i1 - 1, col], errors="coerce").to_numpy(dtype=float)
    df.loc[i0:i1 - 1, col] = base + sinal

    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], LABEL_OSC)
    return df


def injetar_lacuna(
    df: pd.DataFrame,
    col: str,
    duracao_pts: int = 20,
    seed: int = 46
) -> pd.DataFrame:
    """
    Cria lacuna "realista" para o pipeline:
      - coloca NaN no valor do sensor (não remove linhas)
      - marca o intervalo como lacuna
    """
    rng = np.random.default_rng(seed)
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    if col not in df.columns or len(df) <= duracao_pts + 3:
        return df

    i0 = int(rng.integers(0, len(df) - duracao_pts))
    i1 = i0 + duracao_pts

    df.loc[i0:i1 - 1, col] = np.nan
    _marcar_intervalo(df, df.loc[i0, "Datetime"], df.loc[i1 - 1, "Datetime"], LABEL_LACUNA)
    return df
def injetar_intervalo_por_tempo(df: pd.DataFrame, col: str, inicio, fim, modo: str, amp=8.0, delta=-12.0):
    """
    Injeta falha em um intervalo de tempo específico.
    modo: "stuck", "stuck_at_zero", "oscilacao", "queda", "lacuna"
    """
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    m = (df["Datetime"] >= inicio) & (df["Datetime"] <= fim)
    if m.sum() == 0:
        return df

    idx = df.index[m]

    if modo == LABEL_STUCK:
        v = float(df.loc[idx[0], col])
        df.loc[idx, col] = v
        df.loc[idx, "label"] = LABEL_STUCK

    elif modo == LABEL_STUCK_ZERO:
        df.loc[idx, col] = 0.0
        df.loc[idx, "label"] = LABEL_STUCK_ZERO

    elif modo == LABEL_OSC:
        t = np.arange(len(idx))
        sinal = amp * np.sign(np.sin(2 * np.pi * t / 4.0))
        base = pd.to_numeric(df.loc[idx, col], errors="coerce").to_numpy(dtype=float)
        df.loc[idx, col] = base + sinal
        df.loc[idx, "label"] = LABEL_OSC

    elif modo == LABEL_QUEDA:
        base = pd.to_numeric(df.loc[idx, col], errors="coerce").to_numpy(dtype=float)
        df.loc[idx, col] = base + float(delta)
        df.loc[idx, "label"] = LABEL_QUEDA

    elif modo == LABEL_LACUNA:
        df.loc[idx, col] = np.nan
        df.loc[idx, "label"] = LABEL_LACUNA

    return df
