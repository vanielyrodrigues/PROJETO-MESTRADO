#Main

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ajustes import (
    carregar_dados,
    filtrar_periodo,
    reamostrar_e_imputar,
)

from simulacao_falhas import (
    preparar_base,
    injetar_intervalo_por_tempo,
    balancear_falhas,
    resumo_eventos_injetados,
    LABEL_OSC,
    LABEL_LACUNA,
    LABEL_QUEDA,
)

from ml_pipeline import (
    avaliar_modelos_treino_teste_eventos
)


# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA_DADOS = os.path.join(BASE_DIR, "dados")
FREQ_PADRAO = "10min"
PASTA_RESULTADOS = os.path.join(BASE_DIR, "resultados")
PASTA_RELATORIOS = os.path.join(BASE_DIR, "relatorios")
COLUNA_ALVO = "MP2,5_1"

PASTA_RESULTADOS_SEM_ERRO = os.path.join(PASTA_RESULTADOS, "sem_erro")
PASTA_RESULTADOS_COM_ERRO = os.path.join(PASTA_RESULTADOS, "com_erro")

PASTA_RELATORIOS_SEM_ERRO = os.path.join(PASTA_RELATORIOS, "sem_erro")
PASTA_RELATORIOS_COM_ERRO = os.path.join(PASTA_RELATORIOS, "com_erro")


# ==============================
# AUXILIARES
# ==============================
def criar_pastas():
    for pasta in [
        PASTA_RESULTADOS,
        PASTA_RELATORIOS,
        PASTA_RESULTADOS_SEM_ERRO,
        PASTA_RESULTADOS_COM_ERRO,
        PASTA_RELATORIOS_SEM_ERRO,
        PASTA_RELATORIOS_COM_ERRO,
    ]:
        os.makedirs(pasta, exist_ok=True)


def ler_periodo():
    ini = pd.to_datetime(input("Data início (dd/mm/aaaa hh:mm): "), dayfirst=True)
    fim = pd.to_datetime(input("Data fim (dd/mm/aaaa hh:mm): "), dayfirst=True)
    return ini, fim


def gerar_nome_base(coluna, ini, fim):
    col_safe = coluna.replace(",", "").replace(".", "").replace(" ", "_")
    return f"{col_safe}_{ini.strftime('%Y%m%d_%H%M')}_to_{fim.strftime('%Y%m%d_%H%M')}"


def preparar_dados_periodo(dados, coluna, ini, fim):
    df = filtrar_periodo(dados, ini, fim)
    df = reamostrar_e_imputar(df, FREQ_PADRAO)
    df = preparar_base(df, coluna)
    return df


def config_falhas_padrao():
    return {
        "oscilacao": {"duracao_pts": 30, "n_eventos": 18, "amp": 10.0},
        "lacuna": {"duracao_pts": 15, "n_eventos": 18},
        "queda": {"duracao_pts": 12, "n_eventos": 18, "delta": -15.0},
    }


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


def salvar_distribuicao(df, pasta, nome_arquivo):
    os.makedirs(pasta, exist_ok=True)

    dist = df["label"].value_counts(dropna=False).reset_index()
    dist.columns = ["classe", "pontos"]

    resumo = resumo_eventos_injetados(df, falhas=[LABEL_OSC, LABEL_LACUNA, LABEL_QUEDA])

    dist.to_csv(os.path.join(pasta, f"{nome_arquivo}_distribuicao_pontos.csv"), index=False)
    resumo.to_csv(os.path.join(pasta, f"{nome_arquivo}_resumo_eventos.csv"), index=False)

    print("\nDistribuição de pontos por classe:")
    print(dist)

    print("\nResumo por eventos contínuos:")
    print(resumo)


# ==============================
# FIGURAS
# ==============================
def plot_serie_com_labels(df, coluna, pasta, nome_arquivo, titulo):
    _paper_style()
    fig = plt.figure(figsize=(12, 4))

    plt.plot(df["Datetime"], df[coluna], linewidth=1.2, label="Série")

    falhas = [LABEL_OSC, LABEL_LACUNA, LABEL_QUEDA]
    for falha in falhas:
        sub = df[df["label"].astype(str) == falha]
        if not sub.empty:
            plt.scatter(sub["Datetime"], sub[coluna], s=18, label=falha)

    plt.title(titulo)
    plt.xlabel("Tempo")
    plt.ylabel(coluna.replace(",", "."))
    plt.legend()
    plt.grid(True)

    caminho = os.path.join(pasta, nome_arquivo)
    _savefig(caminho)
    plt.close(fig)
    print(f"✅ Figura salva: {os.path.abspath(caminho)}")


def _primeira_janela_da_classe(df, label, pad_before=25, pad_after=25):
    idx = df.index[df["label"].astype(str) == str(label)].to_list()
    if not idx:
        return None
    i0 = idx[0]
    pos0 = df.index.get_loc(i0)
    a = max(0, pos0 - pad_before)
    b = min(len(df) - 1, pos0 + pad_after)
    return df.iloc[a:b + 1].copy()


def plot_e_salvar_falha_individual(df, coluna, label, pasta, nome_arquivo, titulo):
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

    caminho = os.path.join(pasta, nome_arquivo)
    _savefig(caminho)
    plt.close(fig)

    print(f"✅ Figura salva: {os.path.abspath(caminho)}")


def gerar_figuras_falhas(df, coluna, pasta):
    plot_e_salvar_falha_individual(
        df, coluna, LABEL_QUEDA, pasta, "falha_queda.png",
        "Falha: Queda (redução abrupta do valor)"
    )

    plot_e_salvar_falha_individual(
        df, coluna, LABEL_OSC, pasta, "falha_oscilacao.png",
        "Falha: Oscilação (variação rápida e instável)"
    )

    plot_e_salvar_falha_individual(
        df, coluna, LABEL_LACUNA, pasta, "falha_lacuna.png",
        "Falha: Lacuna (ausência de leituras no intervalo)"
    )


# ==============================
# CRIA BASE COM FALHAS PARA TREINO OU TESTE
# ==============================
def criar_base_com_falhas(df_original, coluna, pasta_resultados, nome_base, incluir_falhas_finais=True):
    config_balanco = config_falhas_padrao()

    df_com, log_eventos = balancear_falhas(
        df_original.copy(),
        coluna,
        config=config_balanco,
        return_log=True
    )

    # Falhas finais para garantir que também existam eventos próximos ao final da série.
    # Elas entram na contagem como eventos adicionais.
    if incluir_falhas_finais and not df_com.empty:
        fim_serie = df_com["Datetime"].max()

        df_com = injetar_intervalo_por_tempo(
            df_com, coluna,
            fim_serie - pd.Timedelta(minutes=180),
            fim_serie - pd.Timedelta(minutes=150),
            modo=LABEL_OSC,
            evento_id="oscilacao_final"
        )

        df_com = injetar_intervalo_por_tempo(
            df_com, coluna,
            fim_serie - pd.Timedelta(minutes=60),
            fim_serie - pd.Timedelta(minutes=45),
            modo=LABEL_LACUNA,
            evento_id="lacuna_final"
        )

        df_com = injetar_intervalo_por_tempo(
            df_com, coluna,
            fim_serie - pd.Timedelta(minutes=40),
            fim_serie - pd.Timedelta(minutes=20),
            modo=LABEL_QUEDA,
            evento_id="queda_final"
        )

        extras = pd.DataFrame([
            {"evento_id": "oscilacao_final", "falha": LABEL_OSC},
            {"evento_id": "lacuna_final", "falha": LABEL_LACUNA},
            {"evento_id": "queda_final", "falha": LABEL_QUEDA},
        ])
        log_eventos = pd.concat([log_eventos, extras], ignore_index=True)

    path_log = os.path.join(pasta_resultados, f"{nome_base}_eventos_injetados.csv")
    log_eventos.to_csv(path_log, index=False)

    print(f"✅ Log de eventos injetados salvo em: {os.path.abspath(path_log)}")
    return df_com, log_eventos


# ==============================
# OPÇÃO 2 – RODAR SEM FALHAS INJETADAS
# ==============================
def modo_sem_erro(dados):
    coluna = COLUNA_ALVO
    ini, fim = ler_periodo()
    nome_base = gerar_nome_base(coluna, ini, fim)

    criar_pastas()

    df_sem = preparar_dados_periodo(dados, coluna, ini, fim)

    if df_sem.empty:
        print("⚠️ Não há dados no período selecionado.")
        return

    # Mantém tudo como normal para validar falso positivo.
    df_sem["label"] = "normal"
    df_sem["evento_id"] = ""

    # Cria uma base com falhas APENAS para treinar o modelo.
    df_treino_com_falhas, _ = criar_base_com_falhas(
        df_sem.copy(),
        coluna,
        PASTA_RESULTADOS_SEM_ERRO,
        nome_base=f"{nome_base}_treino",
        incluir_falhas_finais=True
    )

    plot_serie_com_labels(
        df_sem,
        coluna,
        PASTA_RELATORIOS_SEM_ERRO,
        "serie_sem_erro.png",
        "Série temporal sem falhas injetadas"
    )

    salvar_distribuicao(df_sem, PASTA_RESULTADOS_SEM_ERRO, f"{nome_base}_sem_erro")

    resultados = avaliar_modelos_treino_teste_eventos(
        df_treino=df_treino_com_falhas,
        df_teste=df_sem,
        coluna=coluna,
        janelas=list(range(20, 101, 10)),
        passos_cfg=list(range(5, 101, 5)),
        pasta_out=PASTA_RESULTADOS_SEM_ERRO,
        nome_base=nome_base,
        rotulo_modo="qualquer",
        nome_teste="sem_erro"
    )

    print(f"\n✅ Resultados SEM ERRO salvos em: {os.path.abspath(PASTA_RESULTADOS_SEM_ERRO)}")
    print(f"✅ Figuras SEM ERRO salvas em: {os.path.abspath(PASTA_RELATORIOS_SEM_ERRO)}")

    if resultados["melhores_configuracoes"].empty:
        print("⚠️ Nenhuma configuração válida foi encontrada.")
    else:
        print("\nMelhores configurações no teste SEM ERRO:")
        print(resultados["melhores_configuracoes"])


# ==============================
# OPÇÃO 3 – RODAR COM FALHAS INJETADAS
# ==============================
def modo_com_erro(dados):
    coluna = COLUNA_ALVO
    ini, fim = ler_periodo()
    nome_base = gerar_nome_base(coluna, ini, fim)

    criar_pastas()

    df_original = preparar_dados_periodo(dados, coluna, ini, fim)

    if df_original.empty:
        print("⚠️ Não há dados no período selecionado.")
        return

    df_original["label"] = "normal"
    df_original["evento_id"] = ""

    df_com, log_eventos = criar_base_com_falhas(
        df_original.copy(),
        coluna,
        PASTA_RESULTADOS_COM_ERRO,
        nome_base=nome_base,
        incluir_falhas_finais=True
    )

    plot_serie_com_labels(
        df_original,
        coluna,
        PASTA_RELATORIOS_COM_ERRO,
        "serie_original_sem_falhas.png",
        "Série temporal original antes da injeção das falhas"
    )

    plot_serie_com_labels(
        df_com,
        coluna,
        PASTA_RELATORIOS_COM_ERRO,
        "serie_com_falhas.png",
        "Série temporal com falhas injetadas"
    )

    gerar_figuras_falhas(df_com, coluna, PASTA_RELATORIOS_COM_ERRO)

    salvar_distribuicao(df_com, PASTA_RESULTADOS_COM_ERRO, f"{nome_base}_com_erro")

    resultados = avaliar_modelos_treino_teste_eventos(
        df_treino=df_com,
        df_teste=df_com,
        coluna=coluna,
        janelas=list(range(20, 101, 10)),
        passos_cfg=list(range(5, 101, 5)),
        pasta_out=PASTA_RESULTADOS_COM_ERRO,
        nome_base=nome_base,
        rotulo_modo="qualquer",
        nome_teste="com_erro"
    )

    print(f"\n✅ Resultados COM ERRO salvos em: {os.path.abspath(PASTA_RESULTADOS_COM_ERRO)}")
    print(f"✅ Figuras COM ERRO salvas em: {os.path.abspath(PASTA_RELATORIOS_COM_ERRO)}")

    if resultados["melhores_configuracoes"].empty:
        print("⚠️ Nenhuma configuração válida foi encontrada.")
    else:
        print("\nMelhores configurações no teste COM ERRO:")
        print(resultados["melhores_configuracoes"])

    print("\nEventos injetados por tipo:")
    print(log_eventos["falha"].value_counts())


# ==============================
# OPÇÃO 4 – GERAR FIGURAS 1 E 2
# ==============================
def modo_figuras_serie(dados):
    coluna = COLUNA_ALVO
    ini, fim = ler_periodo()
    nome_base = gerar_nome_base(coluna, ini, fim)

    criar_pastas()

    df_original = preparar_dados_periodo(dados, coluna, ini, fim)

    if df_original.empty:
        print("⚠️ Não há dados no período selecionado.")
        return

    df_original["label"] = "normal"
    df_original["evento_id"] = ""

    df_com, _ = criar_base_com_falhas(
        df_original.copy(),
        coluna,
        PASTA_RESULTADOS_COM_ERRO,
        nome_base=nome_base,
        incluir_falhas_finais=True
    )

    plot_serie_com_labels(
        df_original,
        coluna,
        PASTA_RESULTADOS,
        "Figura_1_original.png",
        f"Figura 1 – Série temporal original ({coluna})"
    )

    plot_serie_com_labels(
        df_com,
        coluna,
        PASTA_RESULTADOS,
        "Figura_2_com_falhas.png",
        "Figura 2 – Série temporal com falhas simuladas"
    )

    print(f"\n✅ Figuras 1 e 2 salvas em: {os.path.abspath(PASTA_RESULTADOS)}")


# ==============================
# MENU PRINCIPAL
# ==============================
def main():
    criar_pastas()

    print("\n=== Sistema de Simulação + Treinamento ===")
    print("(1) Sair")
    print("(2) Rodar SEM falhas injetadas")
    print("(3) Rodar COM falhas injetadas")
    print("(4) Gerar Figuras 1 e 2")

    dados = carregar_dados(PASTA_DADOS)

    if dados.empty:
        print("⚠️ Nenhum dado carregado. Verifique a pasta de dados.")
        return

    while True:
        op = input("\nEscolha: ").strip()

        if op == "1":
            break
        elif op == "2":
            modo_sem_erro(dados)
        elif op == "3":
            modo_com_erro(dados)
        elif op == "4":
            modo_figuras_serie(dados)
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()

