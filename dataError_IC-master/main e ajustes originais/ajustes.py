import pandas as pd
import os
import numpy as np

def carregar_dados(pasta='dados'):
    arquivos_csv = sorted([arq for arq in os.listdir(pasta) if arq.endswith('.csv')])
    todos_dados = []

    for nome_arquivo in arquivos_csv:
        caminho = os.path.join(pasta, nome_arquivo)
        print(f"Lendo arquivo: {nome_arquivo}")
        df = pd.read_csv(caminho)

        if 'Datetime' not in df.columns and ('Data' in df.columns and 'Hora' in df.columns):
            df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], errors='coerce', dayfirst=True)
        elif 'Datetime' not in df.columns:
            print(f"Erro: Arquivo {nome_arquivo} não contém 'Datetime' nem 'Data' e 'Hora'. Pulando...")
            continue

        todos_dados.append(df)

    if todos_dados:
        df_concatenado = pd.concat(todos_dados, ignore_index=True)
        df_concatenado.sort_values(by='Datetime', inplace=True)
        df_concatenado.to_csv('dados_ordenados.csv', index=False)
        print("Arquivo 'dados_ordenados.csv' criado com sucesso!")
        return df_concatenado
    else:
        print("Nenhum dado válido encontrado.")
        return pd.DataFrame()

def filtrar_periodo(df, inicio, fim):
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    return df[(df['Datetime'] >= inicio) & (df['Datetime'] <= fim)].copy()

def detectar_stuck(df, coluna, janela=20):
    resultados = []
    valores = df[coluna].round(3).values
    datas = df['Datetime'].values

    for i in range(len(valores) - janela + 1):
        segmento = valores[i:i+janela]
        if np.all(segmento == segmento[0]):
            resultados.append((datas[i+janela//2], valores[i+janela//2]))

    return pd.DataFrame(resultados, columns=['Datetime', coluna])

def detectar_oscilacoes(df, coluna, limite=10):
    df = df.copy()
    df['Diferenca'] = df[coluna].diff().abs()
    return df[df['Diferenca'] > limite][['Datetime', coluna]]

def detectar_lacunas(df, limite_minutos=60):
    df = df.copy()
    df['Diferenca'] = df['Datetime'].diff().dt.total_seconds() / 60.0
    return df[df['Diferenca'] > limite_minutos][['Datetime', 'Diferenca']]

def reamostrar_e_imputar(df, freq='10min'):
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)
    df_reamostrado = df.resample(freq).mean(numeric_only=True)
    df_imputado = df_reamostrado.interpolate(method='linear')
    df_imputado.reset_index(inplace=True)
    return df_imputado
