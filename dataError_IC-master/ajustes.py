#AJUSTES

import os
import numpy as np
import pandas as pd

COLUNAS_RAW_9 = [
    "Data", "Hora", "Num", "Temperatura", "Umidade",
    "MP2,5_1", "MP10_1", "MP2,5_2", "MP10_2"
]

COLUNAS_BASE = [
    "Datetime", "Data", "Hora", "Num", "Temperatura",
    "Umidade", "MP2,5_1", "MP10_1", "MP2,5_2", "MP10_2"
]


def _to_numeric_safe(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _parse_datetime_flex(data_str: pd.Series, hora_str: pd.Series) -> pd.Series:
    combinado = data_str.astype(str).str.strip() + " " + hora_str.astype(str).str.strip()

    # tenta formatos comuns primeiro
    dt = pd.to_datetime(combinado, format="%d/%m/%Y %H:%M:%S", errors="coerce")

    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            combinado.loc[mask],
            format="%d/%m/%y %H:%M:%S",
            errors="coerce"
        )

    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            combinado.loc[mask],
            format="%d/%m/%Y %H:%M",
            errors="coerce"
        )

    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            combinado.loc[mask],
            format="%d/%m/%y %H:%M",
            errors="coerce"
        )

    # fallback flexível
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            combinado.loc[mask],
            errors="coerce",
            dayfirst=True
        )

    return dt


def _normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    novas = []
    for c in df.columns:
        c2 = str(c).strip().replace("\ufeff", "")
        novas.append(c2)
    df.columns = novas

    return df


def _read_csv_com_cabecalho(caminho: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(caminho)
        df = _normalizar_colunas(df)

        if "Datetime" in df.columns or ("Data" in df.columns and "Hora" in df.columns):
            return df
        return None
    except Exception:
        return None


def _read_csv_raw(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(
        caminho,
        sep=";",
        header=None,
        engine="python"
    )

    # remove colunas finais totalmente vazias
    while df.shape[1] > 0 and df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]

    # caso bruto padrão
    if df.shape[1] == 9:
        df.columns = COLUNAS_RAW_9
        return df

    # às vezes vem com coluna extra no final
    if df.shape[1] > 9:
        df = df.iloc[:, :9].copy()
        df.columns = COLUNAS_RAW_9
        return df

    raise ValueError(
        f"Formato inesperado em {os.path.basename(caminho)}: "
        f"{df.shape[1]} colunas."
    )


def _read_any_csv(caminho: str) -> pd.DataFrame:
    # 1) tenta com cabeçalho
    df = _read_csv_com_cabecalho(caminho)
    if df is not None:
        return df

    # 2) tenta raw
    return _read_csv_raw(caminho)


def _normalizar_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _normalizar_colunas(df)

    if "Datetime" in df.columns:
        dt = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

        mask = dt.isna()
        if mask.any():
            dt.loc[mask] = pd.to_datetime(
                df.loc[mask, "Datetime"],
                format="%d/%m/%Y %H:%M:%S",
                errors="coerce"
            )

        mask = dt.isna()
        if mask.any():
            dt.loc[mask] = pd.to_datetime(
                df.loc[mask, "Datetime"],
                format="%d/%m/%y %H:%M:%S",
                errors="coerce"
            )

        mask = dt.isna()
        if mask.any():
            dt.loc[mask] = pd.to_datetime(
                df.loc[mask, "Datetime"],
                errors="coerce",
                dayfirst=True
            )

        df["Datetime"] = dt

    elif "Data" in df.columns and "Hora" in df.columns:
        df["Datetime"] = _parse_datetime_flex(df["Data"], df["Hora"])

    else:
        raise ValueError("Arquivo sem colunas Datetime nem Data/Hora.")

    df = df.dropna(subset=["Datetime"]).copy()
    return df


def _padronizar_estrutura(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _normalizar_colunas(df)

    # garante colunas mínimas
    for col in COLUNAS_BASE:
        if col not in df.columns:
            df[col] = np.nan

    # mantém só o que interessa
    df = df[COLUNAS_BASE].copy()

    # preenche Data/Hora a partir de Datetime, se necessário
    if df["Data"].isna().all():
        df["Data"] = df["Datetime"].dt.strftime("%d/%m/%Y")
    if df["Hora"].isna().all():
        df["Hora"] = df["Datetime"].dt.strftime("%H:%M:%S")

    # converte numéricos
    for col in ["Num", "Temperatura", "Umidade", "MP2,5_1", "MP10_1", "MP2,5_2", "MP10_2"]:
        df[col] = _to_numeric_safe(df[col])

    return df


def carregar_dados(pasta="dados", salvar_csv=True) -> pd.DataFrame:
    arquivos_csv = sorted(
        [arq for arq in os.listdir(pasta) if arq.lower().endswith(".csv")]
    )

    todos = []

    for nome in arquivos_csv:
        caminho = os.path.join(pasta, nome)
        print(f"Lendo arquivo: {nome}")

        try:
            df = _read_any_csv(caminho)
            df = _normalizar_datetime(df)
            df = _padronizar_estrutura(df)
            todos.append(df)

        except Exception as e:
            print(f"⚠️ Pulando {nome} (erro): {e}")

    if not todos:
        print("Nenhum dado válido encontrado.")
        return pd.DataFrame()

    df_all = pd.concat(todos, ignore_index=True)
    df_all = df_all.sort_values("Datetime").reset_index(drop=True)

    if salvar_csv:
        df_all.to_csv("dados_ordenados.csv", index=False)
        print("Arquivo 'dados_ordenados.csv' criado com sucesso!")

    return df_all


def filtrar_periodo(df: pd.DataFrame, inicio, fim) -> pd.DataFrame:
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    return df[(df["Datetime"] >= inicio) & (df["Datetime"] <= fim)].copy()


def detectar_stuck(df: pd.DataFrame, coluna: str, janela=20, tol=1e-9) -> pd.DataFrame:
    resultados = []
    valores = df[coluna].astype(float).values
    datas = df["Datetime"].values

    for i in range(len(valores) - janela + 1):
        seg = valores[i:i + janela]
        if np.nanstd(seg) <= tol:
            resultados.append((datas[i + janela // 2], valores[i + janela // 2]))

    return pd.DataFrame(resultados, columns=["Datetime", coluna])


# ============================================================
# MUDANÇA NOVA:
# detector por algoritmo para stuck e stuck_at_zero
# - NÃO usa modelo de ML
# - marca diretamente a coluna "label"
# - permite definir quantos pontos consecutivos caracterizam travamento
# ============================================================
def detectar_stuck_e_stuck_zero(
    df: pd.DataFrame,
    coluna: str,
    min_pontos_stuck: int = 6,
    min_pontos_zero: int = 6,
    tol: float = 1e-9
) -> pd.DataFrame:
    """
    Marca no próprio DataFrame:
    - stuck: sequência de valores constantes por min_pontos_stuck ou mais
    - stuck_at_zero: sequência de zeros por min_pontos_zero ou mais

    Exemplo com frequência de 10 minutos:
    6 pontos  = 1 hora
    24 pontos = 4 horas
    """
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    if "label" not in df.columns:
        df["label"] = "normal"

    valores = pd.to_numeric(df[coluna], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    i = 0

    while i < n:
        vi = valores[i]

        if np.isnan(vi):
            i += 1
            continue

        # ------------------------------------
        # caso 1: sequência de zeros
        # ------------------------------------
        if np.isclose(vi, 0.0, atol=tol):
            j = i + 1
            while j < n:
                vj = valores[j]
                if np.isnan(vj) or (not np.isclose(vj, 0.0, atol=tol)):
                    break
                j += 1

            tamanho = j - i
            if tamanho >= min_pontos_zero:
                df.loc[i:j - 1, "label"] = "stuck_at_zero"

            i = j
            continue

        # ------------------------------------
        # caso 2: sequência constante não-zero
        # ------------------------------------
        j = i + 1
        while j < n:
            vj = valores[j]
            if np.isnan(vj):
                break
            if np.isclose(vj, vi, atol=tol):
                j += 1
            else:
                break

        tamanho = j - i
        if tamanho >= min_pontos_stuck:
            # só marca stuck onde ainda não houver outro rótulo
            mask_normal = df.loc[i:j - 1, "label"].astype(str) == "normal"
            idxs = df.loc[i:j - 1].index[mask_normal]
            df.loc[idxs, "label"] = "stuck"

        i = j

    return df


def detectar_oscilacoes(df: pd.DataFrame, coluna: str, limite=10) -> pd.DataFrame:
    df = df.copy()
    df["Diferenca"] = df[coluna].diff().abs()
    return df[df["Diferenca"] > limite][["Datetime", coluna]]


def detectar_lacunas(df: pd.DataFrame, limite_minutos=60) -> pd.DataFrame:
    df = df.copy()
    df["Diferenca"] = df["Datetime"].diff().dt.total_seconds() / 60.0
    return df[df["Diferenca"] > limite_minutos][["Datetime", "Diferenca"]]


def reamostrar_e_imputar(df: pd.DataFrame, freq="10min") -> pd.DataFrame:
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime")

    # só colunas numéricas no reamostramento
    cols_num = ["Temperatura", "Umidade", "MP2,5_1", "MP10_1", "MP2,5_2", "MP10_2"]
    cols_existentes = [c for c in cols_num if c in df.columns]

    # Reamostra para frequência regular, mas preserva quais pontos eram observados.
    # Isso permite que os gráficos gerais não liguem visualmente grandes intervalos sem leituras.
    df_reamostrado = df[cols_existentes].resample(freq).mean()

    for col in cols_existentes:
        df_reamostrado[f"{col}_observado"] = df_reamostrado[col].notna()

    df_imputado = df_reamostrado.copy()
    df_imputado[cols_existentes] = df_reamostrado[cols_existentes].interpolate(method="linear")

    df_imputado.reset_index(inplace=True)
    return df_imputado