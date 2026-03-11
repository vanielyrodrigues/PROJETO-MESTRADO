# ajustes.py
import os
import numpy as np
import pandas as pd

COLUNAS_RAW_9 = [
    "Data", "Hora", "Num", "Temperatura", "Umidade",
    "MP2,5_1", "MP10_1", "MP2,5_2", "MP10_2"
]

def _to_numeric_safe(s: pd.Series) -> pd.Series:
    # troca vírgula por ponto se vier como string
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _read_any_csv(caminho: str) -> pd.DataFrame:
    """
    Lê CSV em dois formatos:
    (A) Com cabeçalho e separador ',' (ex.: 29102023.csv)
    (B) Sem cabeçalho e separador ';' (ex.: 291223.csv)
    """
    # tenta leitura padrão
    try:
        df = pd.read_csv(caminho)
        if "Datetime" in df.columns or ("Data" in df.columns and "Hora" in df.columns):
            return df
    except Exception:
        pass

    # tenta leitura raw com ';' sem header
    df = pd.read_csv(
        caminho,
        sep=";",
        header=None,
        engine="python"
    )

    # remove última coluna vazia se existir (por causa do ';' final)
    if df.shape[1] >= 10 and df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]

    if df.shape[1] != 9:
        raise ValueError(f"Formato inesperado em {os.path.basename(caminho)}: {df.shape[1]} colunas (esperado 9).")

    df.columns = COLUNAS_RAW_9
    return df

def _normalizar_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Datetime" not in df.columns:
        if "Data" in df.columns and "Hora" in df.columns:
            df["Datetime"] = pd.to_datetime(
                df["Data"].astype(str) + " " + df["Hora"].astype(str),
                errors="coerce",
                dayfirst=True
            )
        else:
            raise ValueError("Não foi possível criar Datetime: faltam colunas Data/Hora e Datetime.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    return df

def carregar_dados(pasta="dados", salvar_csv=True) -> pd.DataFrame:
    arquivos_csv = sorted([arq for arq in os.listdir(pasta) if arq.lower().endswith(".csv")])
    todos = []

    for nome in arquivos_csv:
        caminho = os.path.join(pasta, nome)
        print(f"Lendo arquivo: {nome}")

        try:
            df = _read_any_csv(caminho)
            df = _normalizar_datetime(df)

            # padroniza colunas numéricas mais comuns, se existirem
            for col in ["Temperatura", "Umidade", "MP2,5_1", "MP10_1", "MP2,5_2", "MP10_2"]:
                if col in df.columns:
                    df[col] = _to_numeric_safe(df[col])

            todos.append(df)

        except Exception as e:
            print(f"⚠️ Pulando {nome} (erro): {e}")

    if not todos:
        print("Nenhum dado válido encontrado.")
        return pd.DataFrame()

    df_all = pd.concat(todos, ignore_index=True)
    df_all.sort_values("Datetime", inplace=True)

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
        seg = valores[i:i+janela]
        if np.nanstd(seg) <= tol:
            resultados.append((datas[i + janela//2], valores[i + janela//2]))

    return pd.DataFrame(resultados, columns=["Datetime", coluna])

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
    df_reamostrado = df.resample(freq).mean(numeric_only=True)
    df_imputado = df_reamostrado.interpolate(method="linear")
    df_imputado.reset_index(inplace=True)
    return df_imputado
