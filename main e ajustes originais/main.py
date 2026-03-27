# main.py
from ajustes import (
   carregar_dados,
   filtrar_periodo,
   detectar_stuck,
   detectar_oscilacoes,
   detectar_lacunas,
   reamostrar_e_imputar
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# Carregar e organizar os dados
dados_ordenados = carregar_dados()

if dados_ordenados.empty:
   print("Erro: Nenhum dado carregado.")
   exit()

# Menu de seleção de variável
variaveis = {
   "1": "Temperatura",
   "2": "Umidade",
   "3": "MP2,5_1",
   "4": "MP10_1",
   "5": "MP2,5_2",
   "6": "MP10_2"
}

print("\nEscolha a opção desejada para análise:")
for k, v in variaveis.items():
   print(f"({k}) {v}")
print("(7) Sair")

opcao = input("Digite a opção desejada: ")
if opcao == "7" or opcao not in variaveis:
   print("Encerrando...")
   exit()

coluna = variaveis[opcao]

# Entrada do período
data_inicio = input("Data de início (dd/mm/aaaa hh:mm): ")
data_fim = input("Data de fim (dd/mm/aaaa hh:mm): ")

try:
   data_inicio = pd.to_datetime(data_inicio, dayfirst=True)
   data_fim = pd.to_datetime(data_fim, dayfirst=True)
except Exception as e:
   print(f"Erro ao converter datas: {e}")
   exit()

# Filtragem
df_filtrado = filtrar_periodo(dados_ordenados, data_inicio, data_fim)
print(f"→ Total de pontos para plotagem: {len(df_filtrado)}")
print(df_filtrado[[coluna, "Datetime"]].head())

# Detecção de falhas
df_stuck = detectar_stuck(df_filtrado, coluna)
df_oscilacoes = detectar_oscilacoes(df_filtrado, coluna)
df_lacunas = detectar_lacunas(df_filtrado)

# Imputação
df_imputado = reamostrar_e_imputar(df_filtrado)

# Gráfico
plt.figure(figsize=(15, 6))
plt.plot(df_imputado['Datetime'], df_imputado[coluna], label='Original + Imputado', color='blue')

if not df_stuck.empty:
   plt.scatter(df_stuck['Datetime'], df_stuck[coluna], color='red', label='STUCK')
if not df_oscilacoes.empty:
   plt.scatter(df_oscilacoes['Datetime'], df_oscilacoes[coluna], color='orange', label='OSCILAÇÃO')
if not df_lacunas.empty:
    y_lacuna = df_imputado[coluna].mean()
    plt.scatter(df_lacunas['Datetime'], [y_lacuna] * len(df_lacunas), color='purple', label='LACUNA')

    # Adiciona uma anotação explicativa ao lado de cada ponto
    for idx, row in df_lacunas.iterrows():
        lacuna_time = row['Datetime']
        plt.text(
            lacuna_time,
            y_lacuna + 1,  # ligeiramente acima da média
            f"Lacuna > {int(row['Diferenca'])} min",
            fontsize=8,
            rotation=45,
            ha='left',
            va='bottom',
            color='purple'
        )

# Formatar eixo X com data e hora
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
plt.xticks(rotation=45)

plt.xlabel("Data e Hora")
plt.ylabel(coluna)
plt.title(f"Análise: {coluna}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
