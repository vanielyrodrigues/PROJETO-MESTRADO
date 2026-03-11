# main.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ajustes import (
    carregar_dados,
    filtrar_periodo,
    reamostrar_e_imputar
)

from simulacao_falhas import (
    preparar_base,
    injetar_stuck,
    injetar_stuck_zero,
    injetar_queda,
    injetar_oscilacao,
    injetar_lacuna,
    injetar_intervalo_por_tempo,
    LABEL_OSC,
    LABEL_STUCK_ZERO,
    LABEL_LACUNA
)

from ml_pipeline import (
    criar_features_temporais,
    avaliar_modelos
)

# =============================
# CONFIG
# =============================
PASTA_DADOS = "dados"
FREQ_PADRAO = "10min"
PASTA_RESULTADOS = "resultados"
PASTA_RELATORIOS = "relatorios"


# =============================
# AUXILIARES
# =============================
def ler_periodo():
    ini = pd.to_datetime(input("Data início (dd/mm/aaaa hh:mm): "), dayfirst=True)
    fim = pd.to_datetime(input("Data fim (dd/mm/aaaa hh:mm): "), dayfirst=True)
    return ini, fim


def gerar_nome_base(coluna, ini, fim):
    # padroniza nome seguro
    col_safe = coluna.replace(",", "").replace(".", "").replace(" ", "_")
    return f"{col_safe}_{ini.strftime('%Y%m%d_%H%M')}_to_{fim.strftime('%Y%m%d_%H%M')}"



# PLOT PADRÃO PAPER + FIGURAS DE FALHAS

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
        print("⚠️ DF não tem coluna 'label'. Gere com preparar_base + injeções.")
        return

    trecho = _primeira_janela_da_classe(df, label)
    if trecho is None:
        print(f"⚠️ Não encontrou classe '{label}' para plotar.")
        return

    _paper_style()
    fig = plt.figure(figsize=(8.5, 3.2))
    ax = plt.gca()

    ax.plot(trecho["Datetime"], trecho[coluna], linewidth=1.2, label="Série")

    mask = (trecho["label"].astype(str) == str(label))
    if mask.any():
        t_ini = trecho.loc[mask, "Datetime"].iloc[0]
        t_fim = trecho.loc[mask, "Datetime"].iloc[-1]
        ax.axvspan(t_ini, t_fim, alpha=0.20, label=f"Falha: {label}")

    ax.set_title(titulo)
    ax.set_xlabel("Tempo")
    ax.set_ylabel(coluna.replace(",", "."))
    ax.legend(loc="best")

    out = os.path.join(PASTA_RELATORIOS, nome_arquivo)
    _savefig(out)
    plt.close(fig)
    print(f"✅ Salvo: {out}")


def plot_e_salvar_falhas_todas(df, coluna, nome_arquivo="falhas_todas.png",
                              titulo="Série temporal com falhas simuladas (visão geral)"):
    if "label" not in df.columns:
        print("⚠️ DF não tem coluna 'label'. Gere com preparar_base + injeções.")
        return

    _paper_style()
    fig = plt.figure(figsize=(12, 4.2))
    ax = plt.gca()

    ax.plot(df["Datetime"], df[coluna], linewidth=1.0, alpha=0.85, label="Série")

    classes = [c for c in df["label"].astype(str).unique().tolist() if c != "normal"]
    classes = sorted(classes)

    for c in classes:
        m = (df["label"].astype(str) == c)
        if m.sum() == 0:
            continue
        ax.scatter(df.loc[m, "Datetime"], df.loc[m, coluna], s=18, label=c)

    ax.set_title(titulo)
    ax.set_xlabel("Tempo")
    ax.set_ylabel(coluna.replace(",", "."))

    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)

    out = os.path.join(PASTA_RELATORIOS, nome_arquivo)
    _savefig(out)
    plt.close(fig)
    print(f"✅ Salvo: {out}")


def gerar_figuras_falhas(df, coluna):
    # geral
    plot_e_salvar_falhas_todas(df, coluna, nome_arquivo="falhas_todas.png")

    # individuais (labels precisam existir no DF)
    plot_e_salvar_falha_individual(df, coluna, "queda", "falha_queda.png",
                                   "Falha: Queda (redução abrupta do sinal)")

    plot_e_salvar_falha_individual(df, coluna, "oscilacao", "falha_oscilacao.png",
                                   "Falha: Oscilação (variação rápida e instável)")

    plot_e_salvar_falha_individual(df, coluna, "lacuna", "falha_lacuna.png",
                                   "Falha: Lacuna (ausência de leituras no intervalo)")

    plot_e_salvar_falha_individual(df, coluna, "stuck", "falha_stuck.png",
                                   "Falha: Stuck (sinal constante por um período)")

    plot_e_salvar_falha_individual(df, coluna, "stuck_at_zero", "falha_stuck_at_zero.png",
                                   "Falha: Stuck-at-zero (valores nulos persistentes)")



# OPÇÃO 2 – FIGURAS 1 e 2

def modo_simular_plot(dados):
    coluna = "MP2,5_1"
    ini, fim = ler_periodo()

    df = filtrar_periodo(dados, ini, fim)
    df = reamostrar_e_imputar(df, FREQ_PADRAO)
    df = preparar_base(df, coluna)

    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    # Figura 1: Original
    _paper_style()
    fig = plt.figure(figsize=(12, 4))
    plt.plot(df["Datetime"], df[coluna], linewidth=1.2)
    plt.title("Figura 1 – Série temporal original (MP2,5_1)")
    plt.xlabel("Tempo")
    plt.ylabel("MP2,5")
    plt.grid(True)
    _savefig(os.path.join(PASTA_RESULTADOS, "Figura_1_original.png"))
    plt.close(fig)

    # Injeta falhas leves
    df2 = df.copy()
    df2 = injetar_stuck(df2, coluna, duracao_pts=25, seed=1)
    df2 = injetar_oscilacao(df2, coluna, duracao_pts=30, seed=2)
    df2 = injetar_queda(df2, coluna, duracao_pts=10, seed=3)
    df2 = injetar_lacuna(df2, coluna, duracao_pts=15, seed=4)

    # Figura 2: Com falhas
    _paper_style()
    fig = plt.figure(figsize=(12, 4))
    plt.plot(df2["Datetime"], df2[coluna], linewidth=1.2)
    plt.title("Figura 2 – Série temporal com falhas simuladas")
    plt.xlabel("Tempo")
    plt.ylabel("MP2,5")
    plt.grid(True)
    _savefig(os.path.join(PASTA_RESULTADOS, "Figura_2_com_falhas.png"))
    plt.close(fig)

    print(f"✅ Figuras 1 e 2 salvas em: {PASTA_RESULTADOS}/")



# OPÇÃO 3 – SIMULA + TREINA

def modo_simular_e_treinar(dados):
    coluna = "MP2,5_1"
    ini, fim = ler_periodo()

    df = filtrar_periodo(dados, ini, fim)
    df = reamostrar_e_imputar(df, FREQ_PADRAO)
    df = preparar_base(df, coluna)


    # 1) Injeta falhas (treino)

    df = injetar_stuck(df, coluna, 25, seed=10)
    df = injetar_queda(df, coluna, 10, seed=11)
    df = injetar_oscilacao(df, coluna, 25, seed=12)
    df = injetar_stuck_zero(df, coluna, 20, seed=13)
    df = injetar_lacuna(df, coluna, 15, seed=14)


    # 2) Garante falhas no FINAL (para aparecer no teste)

    fim_serie = df["Datetime"].max()
    janela_inicio = fim_serie - pd.Timedelta(minutes=90)

    df = injetar_intervalo_por_tempo(
        df, coluna,
        janela_inicio, fim_serie,
        modo=LABEL_OSC
    )

    df = injetar_intervalo_por_tempo(
        df, coluna,
        fim_serie - pd.Timedelta(minutes=60),
        fim_serie - pd.Timedelta(minutes=40),
        modo=LABEL_STUCK_ZERO
    )

    df = injetar_intervalo_por_tempo(
        df, coluna,
        fim_serie - pd.Timedelta(minutes=30),
        fim_serie - pd.Timedelta(minutes=20),
        modo=LABEL_LACUNA
    )


    # 3) Gera figuras das falhas (geral + individuais)

    os.makedirs(PASTA_RELATORIOS, exist_ok=True)
    gerar_figuras_falhas(df, coluna)


    # 4) Features temporais  - AQUI está o criar_features_temporais

    df_feat = criar_features_temporais(df, coluna, lags=12)

    print("\nDistribuição de classes:")
    print(df_feat["label"].value_counts())

    # base nome para salvar
    nome_base = gerar_nome_base(coluna, ini, fim)


    # 5) Avalia modelos (salva matrizes, métricas, etc.)

    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    resultados = avaliar_modelos(
        df_feat=df_feat,
        col_alvo=coluna,
        lags=12,
        n_splits=5,
        salvar_saidas=True,
        pasta_out=PASTA_RESULTADOS,
        nome_base=nome_base
    )

    print(f" Resultados salvos em: {PASTA_RESULTADOS}/")
    print(f" Figuras de falhas salvas em: {PASTA_RELATORIOS}/")



# MENU PRINCIPAL - TEMPORÁRIO

def main():
    print("\n=== Sistema de Simulação + Treinamento ===")
    print("(1) Sair")
    print("(2) Gerar Figuras 1 e 2 (série temporal)")
    print("(3) Simular + Treinar Modelos (matrizes, F1, tabelas) + Figuras falhas")

    dados = carregar_dados(PASTA_DADOS)

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
