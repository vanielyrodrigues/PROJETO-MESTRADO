import os
import numpy as np
import pandas as pd

# ==============================
# MUDANÇA:
# força backend não interativo para evitar erro do tkinter
# ==============================
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ajustes import (
    carregar_dados,
    filtrar_periodo,
    reamostrar_e_imputar,
    detectar_stuck_e_stuck_zero
)

from simulacao_falhas import (
    preparar_base,
    injetar_intervalo_por_tempo,
    balancear_falhas,
    LABEL_OSC,
    LABEL_LACUNA
)

from ml_pipeline import (
    avaliar_xgb_por_falha_e_janela
)

# ==============================
# CONFIG
# MUDANÇA:
# uso de caminhos absolutos para garantir
# que resultados e relatórios sejam salvos
# na pasta correta do projeto
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA_DADOS = os.path.join(BASE_DIR, "dados")
FREQ_PADRAO = "10min"
PASTA_RESULTADOS = os.path.join(BASE_DIR, "resultados")
PASTA_RELATORIOS = os.path.join(BASE_DIR, "relatorios")
COLUNA_ALVO = "MP2,5_1"


# AUXILIARES

def ler_periodo():
    ini = pd.to_datetime(input("Data início (dd/mm/aaaa hh:mm): "), dayfirst=True)
    fim = pd.to_datetime(input("Data fim (dd/mm/aaaa hh:mm): "), dayfirst=True)
    return ini, fim


def gerar_nome_base(coluna, ini, fim):
    col_safe = coluna.replace(",", "").replace(".", "").replace(" ", "_")
    return f"{col_safe}_{ini.strftime('%Y%m%d_%H%M')}_to_{fim.strftime('%Y%m%d_%H%M')}"


# PLOT PADRÃO PAPER

def _paper_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def _savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)


def _primeira_janela_da_classe(df, label, pad_before=25, pad_after=25):
    idx = df.index[df["label"].astype(str) == str(label)].to_list()
    if not idx:
        return None
    i0 = idx[0]
    pos0 = df.index.get_loc(i0)
    a = max(0, pos0 - pad_before)
    b = min(len(df) - 1, pos0 + pad_after)
    return df.iloc[a:b + 1].copy()


def plot_e_salvar_falha_individual(df, coluna, label, nome_arquivo, titulo):
    if "label" not in df.columns:
        print("⚠️ DF não tem coluna 'label'.")
        return

    trecho = _primeira_janela_da_classe(df, label)
    if trecho is None:
        print(f"⚠️ Não encontrou classe '{label}' para plotar.")
        return

    _paper_style()
    fig = plt.figure(figsize=(10, 4))
    plt.plot(trecho["Datetime"], trecho[coluna], linewidth=1.8)
    plt.title(titulo)
    plt.xlabel("Tempo")
    plt.ylabel(coluna.replace(",", "."))
    plt.grid(True)
    caminho = os.path.join(PASTA_RELATORIOS, nome_arquivo)
    _savefig(caminho)
    plt.close(fig)

    # MUDANÇA:
    # mostra caminho absoluto para não haver dúvida
    print(f"✅ Figura salva: {os.path.abspath(caminho)}")


def gerar_figuras_falhas(df, coluna):
    plot_e_salvar_falha_individual(
        df, coluna, "queda", "falha_queda.png",
        "Falha: Queda (redução abrupta do valor)"
    )

    plot_e_salvar_falha_individual(
        df, coluna, "oscilacao", "falha_oscilacao.png",
        "Falha: Oscilação (variação rápida e instável)"
    )

    plot_e_salvar_falha_individual(
        df, coluna, "lacuna", "falha_lacuna.png",
        "Falha: Lacuna (ausência de leituras no intervalo)"
    )

    # ==============================
    # MANTIDO:
    # continua gerando figura de stuck
    # agora detectado por algoritmo
    # ==============================
    plot_e_salvar_falha_individual(
        df, coluna, "stuck", "falha_stuck.png",
        "Falha: Stuck (sinal constante por um período)"
    )

    # ==============================
    # MANTIDO:
    # continua gerando figura de stuck_at_zero
    # agora detectado por algoritmo
    # ==============================
    plot_e_salvar_falha_individual(
        df, coluna, "stuck_at_zero", "falha_stuck_at_zero.png",
        "Falha: Stuck-at-zero (valores nulos persistentes)"
    )


# OPÇÃO 2 – FIGURAS 1 e 2

def modo_simular_plot(dados):
    coluna = COLUNA_ALVO
    ini, fim = ler_periodo()

    df = filtrar_periodo(dados, ini, fim)
    df = reamostrar_e_imputar(df, FREQ_PADRAO)
    df = preparar_base(df, coluna)

    # ==============================
    # MUDANÇA:
    # detecta stuck e stuck_at_zero por algoritmo
    # antes da simulação das outras falhas
    # ==============================
    df = detectar_stuck_e_stuck_zero(
        df,
        coluna=coluna,
        min_pontos_stuck=6,   # 6 pontos = 1 hora em dados de 10 min
        min_pontos_zero=6,
        tol=1e-9
    )

    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    _paper_style()
    fig = plt.figure(figsize=(12, 4))
    plt.plot(df["Datetime"], df[coluna], linewidth=1.2)
    plt.title(f"Figura 1 – Série temporal original ({coluna})")
    plt.xlabel("Tempo")
    plt.ylabel(coluna.replace(",", "."))
    plt.grid(True)
    _savefig(os.path.join(PASTA_RESULTADOS, "Figura_1_original.png"))
    plt.close(fig)

    # ==============================
    # MUDANÇA:
    # removido stuck e stuck_at_zero da simulação
    # porque agora são tratados por algoritmo
    # ==============================
    config_balanco = {
        "oscilacao": {"duracao_pts": 30, "n_eventos": 18, "amp": 10.0},
        "lacuna": {"duracao_pts": 15, "n_eventos": 18},
        "queda": {"duracao_pts": 12, "n_eventos": 18, "delta": -15.0},
    }

    df2 = balancear_falhas(df.copy(), coluna, config=config_balanco)

    _paper_style()
    fig = plt.figure(figsize=(12, 4))
    plt.plot(df2["Datetime"], df2[coluna], linewidth=1.2)
    plt.title("Figura 2 – Série temporal com falhas simuladas")
    plt.xlabel("Tempo")
    plt.ylabel(coluna.replace(",", "."))
    plt.grid(True)
    _savefig(os.path.join(PASTA_RESULTADOS, "Figura_2_com_falhas.png"))
    plt.close(fig)
    plt.close("all")

    print(f"✅ Figuras 1 e 2 salvas em: {os.path.abspath(PASTA_RESULTADOS)}")


# OPÇÃO 3 – SIMULA + TREINA

def modo_simular_e_treinar(dados):
    coluna = COLUNA_ALVO
    ini, fim = ler_periodo()

    df = filtrar_periodo(dados, ini, fim)
    df = reamostrar_e_imputar(df, FREQ_PADRAO)
    df = preparar_base(df, coluna)

    if df.empty:
        print("⚠️ Não há dados no período selecionado.")
        return

    # ==============================
    # MUDANÇA:
    # stuck e stuck_at_zero detectados por algoritmo
    # não passam pelos modelos
    # ==============================
    df = detectar_stuck_e_stuck_zero(
        df,
        coluna=coluna,
        min_pontos_stuck=6,   # 1 hora com frequência de 10 min
        min_pontos_zero=6,
        tol=1e-9
    )

    # 1) Injeta falhas balanceadas ao longo da série
    # ==============================
    # MUDANÇA:
    # removido stuck e stuck_at_zero da simulação
    # ==============================
    config_balanco = {
        "oscilacao": {"duracao_pts": 30, "n_eventos": 18, "amp": 10.0},
        "lacuna": {"duracao_pts": 15, "n_eventos": 18},
        "queda": {"duracao_pts": 12, "n_eventos": 18, "delta": -15.0},
    }
    df = balancear_falhas(df, coluna, config=config_balanco)

    # 2) Garante falhas no final da série para aparecerem no teste
    fim_serie = df["Datetime"].max()

    df = injetar_intervalo_por_tempo(
        df, coluna,
        fim_serie - pd.Timedelta(minutes=180),
        fim_serie - pd.Timedelta(minutes=150),
        modo=LABEL_OSC
    )

    # ==============================
    # MUDANÇA:
    # removidas as injeções finais de stuck e stuck_at_zero
    # ==============================

    df = injetar_intervalo_por_tempo(
        df, coluna,
        fim_serie - pd.Timedelta(minutes=60),
        fim_serie - pd.Timedelta(minutes=45),
        modo=LABEL_LACUNA
    )

    df = injetar_intervalo_por_tempo(
        df, coluna,
        fim_serie - pd.Timedelta(minutes=40),
        fim_serie - pd.Timedelta(minutes=20),
        modo="queda"
    )

    # 3) Gera figuras das falhas
    os.makedirs(PASTA_RELATORIOS, exist_ok=True)
    gerar_figuras_falhas(df, coluna)

    print("\nDistribuição de classes no sinal analisado:")
    print(df["label"].value_counts(dropna=False))

    nome_base = gerar_nome_base(coluna, ini, fim)

    # 4) Treino separado por falha + teste de várias janelas/passos
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    resultados = avaliar_xgb_por_falha_e_janela(
        df_base=df,
        coluna=coluna,

        # ANTES:
        # janelas=list(range(10, 101, 10))
        # passos_cfg=[1, 5, "n/3"]

        # AGORA:
        # deixei a lógica mais próxima do pseudocódigo do professor:
        # testa vários tamanhos de janela e vários avanços
        janelas=list(range(20, 101, 10)),          # 20, 30, ..., 100
        passos_cfg=list(range(5, 101, 5)),         # depois o pipeline limita aos válidos por janela
        n_splits=5,
        pasta_out=PASTA_RESULTADOS,
        nome_base=nome_base,

        # ANTES estava "qualquer"
        # mantive para detectar se a falha apareceu em qualquer ponto da janela
        rotulo_modo="qualquer"
    )

    print(f"\n✅ Resultados salvos em: {os.path.abspath(PASTA_RESULTADOS)}")
    print(f"✅ Figuras de falhas salvas em: {os.path.abspath(PASTA_RELATORIOS)}")

    print("\nMelhores configurações por falha e por modelo:")
    if resultados["melhores_configuracoes"].empty:
        print("⚠️ Nenhuma configuração válida foi encontrada.")
    else:
        print(resultados["melhores_configuracoes"])

    # Arquivos principais gerados
    print("\nArquivos principais gerados:")
    for k, v in resultados["paths"].items():
        print(f" - {k}: {v}")

    plt.close("all")


# MENU PRINCIPAL

def main():
    print("\n=== Sistema de Simulação + Treinamento ===")
    print("(1) Sair")
    print("(2) Gerar Figuras 1 e 2 (série temporal)")
    print("(3) Simular + Treinar Modelos por falha (janelas e passos) + Figuras falhas")

    dados = carregar_dados(PASTA_DADOS)

    if dados.empty:
        print("⚠️ Nenhum dado carregado. Verifique a pasta de dados.")
        return

    while True:
        op = input("\nEscolha: ").strip()

        if op == "1":
            break
        elif op == "2":
            modo_simular_plot(dados)
        elif op == "3":
            modo_simular_e_treinar(dados)
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()