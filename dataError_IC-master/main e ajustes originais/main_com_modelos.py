
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    auc, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

try:
    from catboost import CatBoostClassifier
    CATBOOST_DISPONIVEL = True
except ImportError:
    CATBOOST_DISPONIVEL = False

def detectar_stuck(df, coluna, janela=48):
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

def gerar_dataset_supervisionado(df, coluna):
    df = df.copy()
    df['falha'] = 0
    stuck_df = detectar_stuck(df, coluna)
    oscilacao_df = detectar_oscilacoes(df, coluna)
    lacuna_df = detectar_lacunas(df)
    df.loc[df['Datetime'].isin(stuck_df['Datetime']), 'falha'] = 1
    df.loc[df['Datetime'].isin(oscilacao_df['Datetime']), 'falha'] = 1
    df.loc[df['Datetime'].isin(lacuna_df['Datetime']), 'falha'] = 1
    df = df.dropna()
    return df

def carregar_dados(pasta='dados'):
    arquivos_csv = sorted([arq for arq in os.listdir(pasta) if arq.endswith('.csv')])
    todos_dados = []
    for nome_arquivo in arquivos_csv:
        caminho = os.path.join(pasta, nome_arquivo)
        df = pd.read_csv(caminho)
        if 'Datetime' not in df.columns and {'Data', 'Hora'}.issubset(df.columns):
            df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], errors='coerce', dayfirst=True)
        elif 'Datetime' not in df.columns:
            continue
        todos_dados.append(df)
    if todos_dados:
        df_concatenado = pd.concat(todos_dados, ignore_index=True)
        df_concatenado.sort_values(by='Datetime', inplace=True)
        return df_concatenado
    else:
        return pd.DataFrame()

def salvar_grafico(fig, nome_arquivo):
    os.makedirs("graficos", exist_ok=True)
    fig.savefig(f"graficos/{nome_arquivo}", dpi=300, bbox_inches='tight')
    plt.close(fig)

def treinar_modelos(df, coluna):
    X = df[[coluna]].values
    y = df['falha'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    modelos = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    if CATBOOST_DISPONIVEL:
        modelos['CatBoost'] = CatBoostClassifier(verbose=0)

    resultados = {}

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else y_pred
        resultados[nome] = (y_test, y_pred, y_prob)

    ann = Sequential()
    ann.add(Dense(16, input_dim=1, activation='relu'))
    ann.add(Dense(8, activation='relu'))
    ann.add(Dense(1, activation='sigmoid'))
    ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ann.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    y_prob_ann = ann.predict(X_test).flatten()
    y_pred_ann = (y_prob_ann > 0.5).astype("int32")
    resultados['Rede Neural (ANN)'] = (y_test, y_pred_ann, y_prob_ann)

    for nome, (y_true, y_pred, y_prob) in resultados.items():
        print(f"\nModelo: {nome}")
        print(classification_report(y_true, y_pred))

        fig1 = plt.figure(figsize=(5, 4))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusão - {nome}")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        salvar_grafico(fig1, f"matriz_confusao_{nome}.png")

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig2 = plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Curva ROC - {nome}")
        plt.legend()
        salvar_grafico(fig2, f"roc_{nome}.png")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig3 = plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precisão vs Recall - {nome}")
        salvar_grafico(fig3, f"precisao_recall_{nome}.png")

# Execução com menu
df = carregar_dados('dados')
if df.empty:
    print("Nenhum dado carregado.")
    exit()

variaveis = ['Temperatura', 'Umidade', 'MP2,5_1', 'MP10_1', 'MP2,5_2', 'MP10_2']
print("\nEscolha a variável para análise:")
for idx, var in enumerate(variaveis, 1):
    print(f"({idx}) {var}")
op = input("Digite o número correspondente: ")
try:
    coluna = variaveis[int(op) - 1]
except:
    print("Opção inválida.")
    exit()

# Coleta período
print("\nDigite o período desejado:")
inicio = input("Data de início (dd/mm/aaaa hh:mm): ")
fim = input("Data de fim (dd/mm/aaaa hh:mm): ")
try:
    data_inicio = pd.to_datetime(inicio, dayfirst=True)
    data_fim = pd.to_datetime(fim, dayfirst=True)
except Exception as e:
    print("Erro nas datas:", e)
    exit()

df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.dropna(subset=[coluna])
df_filtrado = df[(df['Datetime'] >= data_inicio) & (df['Datetime'] <= data_fim)]

if df_filtrado.empty:
    print("Nenhum dado encontrado no período informado.")
    exit()

df_supervisionado = gerar_dataset_supervisionado(df_filtrado, coluna)
treinar_modelos(df_supervisionado, coluna)
