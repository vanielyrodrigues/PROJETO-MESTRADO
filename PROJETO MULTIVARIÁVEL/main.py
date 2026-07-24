# main.py

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
COLUNA_ALVO = "MP2,5_1"  # mantido para compatibilidade com funções antigas

# Variáveis que serão processadas automaticamente nas 5 rodadas.
COLUNAS_ANALISADAS = ["MP2,5_1", "MP10_1", "Temperatura", "Umidade"]

# Nome seguro para criar pastas e arquivos.
NOMES_VARIAVEIS = {
    "MP2,5_1": "MP25_1",
    "MP10_1": "MP10_1",
    "Temperatura": "Temperatura",
    "Umidade": "Umidade",
}

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
        # Configuração usada nos experimentos gerais.
        "oscilacao": {
            "duracao_pts": (36, 72),
            "n_eventos": 25,
            "amp": (2.0, 4.0)
        },

        "lacuna": {
            "duracao_pts": (24, 48),
            "n_eventos": 25
        },

        "queda": {
            "duracao_pts": (18, 36),
            "n_eventos": 25,
            "delta": (-18.0, -6.0)
        },
    }


def config_falhas_figuras_professor():
    return {
        # Oscilação menor e mais realista para atender ao pedido do professor.
        "oscilacao": {
            "duracao_pts": (30, 50),
            "n_eventos": 12,
            "amp": (1.5, 3.0)
        },

        # Lacuna maior para aparecer claramente no gráfico geral.
        "lacuna": {
            "duracao_pts": (70, 110),
            "n_eventos": 10
        },

        # Queda sempre negativa, com variação aleatória mais suave.
        "queda": {
            "duracao_pts": (18, 36),
            "n_eventos": 12,
            "delta": (-10.0, -4.0)
        },
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

    resumo = resumo_eventos_injetados(
        df,
        falhas=[LABEL_OSC, LABEL_LACUNA, LABEL_QUEDA]
    )

    dist.to_csv(
        os.path.join(pasta, f"{nome_arquivo}_distribuicao_pontos.csv"),
        index=False
    )

    resumo.to_csv(
        os.path.join(pasta, f"{nome_arquivo}_resumo_eventos.csv"),
        index=False
    )

    print("\nDistribuição de pontos por classe:")
    print(dist)

    print("\nResumo por eventos contínuos:")
    print(resumo)


# ==============================
# FIGURAS
# ==============================
def _serie_para_plotar_sem_ligar_lacunas(df, coluna):
    """
    Usa a série imputada para o processamento, mas no gráfico geral interrompe
    a linha onde não havia leitura observada originalmente.
    """
    y = pd.to_numeric(df[coluna], errors="coerce").copy()
    col_obs = f"{coluna}_observado"

    if col_obs in df.columns:
        y = y.where(df[col_obs].astype(bool))

    return y


def _destacar_lacunas_no_grafico(df, coluna):
    """Retorna pontos das lacunas em uma altura visível no gráfico."""
    sub = df[df["label"].astype(str) == LABEL_LACUNA].copy()

    if sub.empty:
        return sub, None

    y_ref = pd.to_numeric(df[coluna], errors="coerce")
    y_lacuna = y_ref.quantile(0.05)

    if pd.isna(y_lacuna):
        y_lacuna = 0

    sub["_y_lacuna_plot"] = y_lacuna
    return sub, y_lacuna


def _destacar_intervalos_lacuna(df):
    """
    Identifica intervalos contínuos rotulados como lacuna para sombrear no gráfico.
    Isso ajuda a lacuna aparecer claramente na figura geral.
    """
    if "label" not in df.columns or df.empty:
        return []

    mask = df["label"].astype(str) == LABEL_LACUNA

    if not mask.any():
        return []

    intervalos = []
    em_lacuna = False
    inicio = None
    ultimo = None

    for _, row in df.iterrows():
        if row["label"] == LABEL_LACUNA and not em_lacuna:
            em_lacuna = True
            inicio = row["Datetime"]
            ultimo = row["Datetime"]
        elif row["label"] == LABEL_LACUNA and em_lacuna:
            ultimo = row["Datetime"]
        elif row["label"] != LABEL_LACUNA and em_lacuna:
            intervalos.append((inicio, ultimo))
            em_lacuna = False
            inicio = None
            ultimo = None

    if em_lacuna:
        intervalos.append((inicio, ultimo))

    return intervalos


def plot_serie_com_labels(
    df,
    coluna,
    pasta,
    nome_arquivo,
    titulo,
    destacar_lacunas=False
):
    _paper_style()
    fig = plt.figure(figsize=(12, 4))

    y_plot = _serie_para_plotar_sem_ligar_lacunas(df, coluna)

    plt.plot(
        df["Datetime"],
        y_plot,
        linewidth=1.2,
        label="Série"
    )

    # Nos gráficos gerais, marca as falhas, mas a lacuna precisa aparecer mesmo sendo NaN.
    for falha in [LABEL_OSC, LABEL_QUEDA]:
        sub = df[df["label"].astype(str) == falha]

        if not sub.empty:
            plt.scatter(
                sub["Datetime"],
                sub[coluna],
                s=18,
                label=falha
            )

    sub_lac, _ = _destacar_lacunas_no_grafico(df, coluna)

    if not sub_lac.empty:
        plt.scatter(
            sub_lac["Datetime"],
            sub_lac["_y_lacuna_plot"],
            s=35 if destacar_lacunas else 28,
            marker="x",
            label=LABEL_LACUNA
        )

    # Sombra vertical nos intervalos de lacuna apenas nas figuras ajustadas.
    if destacar_lacunas:
        for ini_lac, fim_lac in _destacar_intervalos_lacuna(df):
            plt.axvspan(
                ini_lac,
                fim_lac,
                alpha=0.15
            )

    plt.title(titulo)
    plt.xlabel("Tempo")
    plt.ylabel(coluna.replace(",", "."))
    plt.legend()
    plt.grid(True)

    caminho = os.path.join(pasta, nome_arquivo)
    _savefig(caminho)
    plt.close(fig)

    print(f"✅ Figura salva: {os.path.abspath(caminho)}")


def _primeira_janela_da_classe(df, label, pad_before=80, pad_after=80):
    idx = df.index[df["label"].astype(str) == str(label)].to_list()

    if not idx:
        return None

    i0 = idx[0]
    pos0 = df.index.get_loc(i0)
    a = max(0, pos0 - pad_before)
    b = min(len(df) - 1, pos0 + pad_after)

    return df.iloc[a:b + 1].copy()


def plot_e_salvar_falha_individual(
    df,
    coluna,
    label,
    pasta,
    nome_arquivo,
    titulo,
    destacar_lacunas=False
):
    if "label" not in df.columns:
        print("⚠️ DF não tem coluna 'label'.")
        return

    trecho = _primeira_janela_da_classe(
        df,
        label,
        pad_before=80,
        pad_after=80
    )

    if trecho is None:
        print(f"⚠️ Não encontrou classe '{label}' para plotar.")
        return

    _paper_style()
    fig = plt.figure(figsize=(11, 4))

    y_plot = _serie_para_plotar_sem_ligar_lacunas(trecho, coluna)

    plt.plot(
        trecho["Datetime"],
        y_plot,
        linewidth=1.6,
        label="Série"
    )

    sub = trecho[trecho["label"].astype(str) == label].copy()

    if label == LABEL_LACUNA:
        sub_lac, _ = _destacar_lacunas_no_grafico(trecho, coluna)

        if not sub_lac.empty:
            plt.scatter(
                sub_lac["Datetime"],
                sub_lac["_y_lacuna_plot"],
                s=45 if destacar_lacunas else 35,
                marker="x",
                label="lacuna"
            )

        if destacar_lacunas:
            for ini_lac, fim_lac in _destacar_intervalos_lacuna(trecho):
                plt.axvspan(
                    ini_lac,
                    fim_lac,
                    alpha=0.15
                )

    elif not sub.empty:
        plt.scatter(
            sub["Datetime"],
            sub[coluna],
            s=24,
            label=label
        )

    plt.title(titulo)
    plt.xlabel("Tempo")
    plt.ylabel(coluna.replace(",", "."))
    plt.legend()
    plt.grid(True)

    caminho = os.path.join(pasta, nome_arquivo)
    _savefig(caminho)
    plt.close(fig)

    print(f"✅ Figura salva: {os.path.abspath(caminho)}")


def gerar_figuras_falhas(df, coluna, pasta, destacar_lacunas=False):
    plot_e_salvar_falha_individual(
        df,
        coluna,
        LABEL_QUEDA,
        pasta,
        "falha_queda.png",
        "Falha: Queda (redução abrupta do valor)",
        destacar_lacunas=destacar_lacunas
    )

    plot_e_salvar_falha_individual(
        df,
        coluna,
        LABEL_OSC,
        pasta,
        "falha_oscilacao.png",
        "Falha: Oscilação (variação rápida e instável)",
        destacar_lacunas=destacar_lacunas
    )

    plot_e_salvar_falha_individual(
        df,
        coluna,
        LABEL_LACUNA,
        pasta,
        "falha_lacuna.png",
        "Falha: Lacuna (ausência de leituras no intervalo)",
        destacar_lacunas=destacar_lacunas
    )


# ==============================
# CRIA BASE COM FALHAS PARA TREINO OU TESTE
# ==============================
def criar_base_com_falhas(
    df_original,
    coluna,
    pasta_resultados,
    nome_base,
    incluir_falhas_finais=True,
    seed=42,
    config_custom=None
):
    config_balanco = (
        config_custom
        if config_custom is not None
        else config_falhas_padrao()
    )

    df_com, log_eventos = balancear_falhas(
        df_original.copy(),
        coluna,
        config=config_balanco,
        return_log=True,
        seed=seed
    )

    # Falhas finais para garantir que também existam eventos próximos ao final da série.
    # Usa os parâmetros da própria variável para evitar simulações irreais
    # em Temperatura e Umidade.
    if incluir_falhas_finais and not df_com.empty:
        fim_serie = df_com["Datetime"].max()

        amp_final = config_balanco.get("oscilacao", {}).get("amp", (1.5, 3.0))
        delta_final = config_balanco.get("queda", {}).get("delta", (-10.0, -4.0))

        df_com = injetar_intervalo_por_tempo(
            df_com,
            coluna,
            fim_serie - pd.Timedelta(minutes=180),
            fim_serie - pd.Timedelta(minutes=150),
            modo=LABEL_OSC,
            amp=amp_final,
            evento_id="oscilacao_final"
        )

        df_com = injetar_intervalo_por_tempo(
            df_com,
            coluna,
            fim_serie - pd.Timedelta(minutes=90),
            fim_serie - pd.Timedelta(minutes=30),
            modo=LABEL_LACUNA,
            evento_id="lacuna_final"
        )

        df_com = injetar_intervalo_por_tempo(
            df_com,
            coluna,
            fim_serie - pd.Timedelta(minutes=25),
            fim_serie - pd.Timedelta(minutes=10),
            modo=LABEL_QUEDA,
            delta=delta_final,
            evento_id="queda_final"
        )

        extras = pd.DataFrame([
            {"evento_id": "oscilacao_final", "falha": LABEL_OSC},
            {"evento_id": "lacuna_final", "falha": LABEL_LACUNA},
            {"evento_id": "queda_final", "falha": LABEL_QUEDA},
        ])

        log_eventos = pd.concat([log_eventos, extras], ignore_index=True)

    path_log = os.path.join(
        pasta_resultados,
        f"{nome_base}_eventos_injetados.csv"
    )

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
        incluir_falhas_finais=True,
        seed=42
    )

    plot_serie_com_labels(
        df_sem,
        coluna,
        PASTA_RELATORIOS_SEM_ERRO,
        "serie_sem_erro.png",
        "Série temporal sem falhas injetadas"
    )

    salvar_distribuicao(
        df_sem,
        PASTA_RESULTADOS_SEM_ERRO,
        f"{nome_base}_sem_erro"
    )

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

    # Cria duas bases com falhas independentes: uma para treino e outra para teste.
    # Isso evita treinar e testar nos mesmos eventos, o que gerava métricas 1.00 artificiais.
    df_treino_com, _ = criar_base_com_falhas(
        df_original.copy(),
        coluna,
        PASTA_RESULTADOS_COM_ERRO,
        nome_base=f"{nome_base}_treino",
        incluir_falhas_finais=False,
        seed=42
    )

    df_com, log_eventos = criar_base_com_falhas(
        df_original.copy(),
        coluna,
        PASTA_RESULTADOS_COM_ERRO,
        nome_base=f"{nome_base}_teste",
        incluir_falhas_finais=True,
        seed=2026
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

    gerar_figuras_falhas(
        df_com,
        coluna,
        PASTA_RELATORIOS_COM_ERRO
    )

    salvar_distribuicao(
        df_com,
        PASTA_RESULTADOS_COM_ERRO,
        f"{nome_base}_com_erro"
    )

    resultados = avaliar_modelos_treino_teste_eventos(
        df_treino=df_treino_com,
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
        incluir_falhas_finais=True,
        seed=2026
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
# OPÇÃO 5 – FIGURAS AJUSTADAS PROFESSOR
# ==============================
def modo_figuras_professor(dados):
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
        nome_base=f"{nome_base}_figuras_professor",
        incluir_falhas_finais=True,
        seed=2026,
        config_custom=config_falhas_figuras_professor()
    )

    pasta_figuras_professor = os.path.join(PASTA_RESULTADOS, "figuras_professor")
    os.makedirs(pasta_figuras_professor, exist_ok=True)

    plot_serie_com_labels(
        df_original,
        coluna,
        pasta_figuras_professor,
        "Figura_1_original_professor.png",
        f"Figura 1 – Série temporal original ({coluna})"
    )

    plot_serie_com_labels(
        df_com,
        coluna,
        pasta_figuras_professor,
        "Figura_2_com_falhas_professor.png",
        "Figura 2 – Série temporal com falhas simuladas",
        destacar_lacunas=True
    )

    gerar_figuras_falhas(
        df_com,
        coluna,
        pasta_figuras_professor,
        destacar_lacunas=True
    )

    print(f"\n✅ Figuras ajustadas para o professor salvas em: {os.path.abspath(pasta_figuras_professor)}")

# ==============================
# EXECUÇÃO AUTOMÁTICA – MULTIVARIÁVEL, 5 RODADAS
# ==============================
def config_base_variavel(coluna: str) -> dict:
    """
    Configura amplitudes e quedas conforme a escala de cada variável.
    As durações permanecem em pontos de 10 minutos.
    """
    configs = {
        "MP2,5_1": {
            "oscilacao": {"duracao_pts": (36, 72), "amp": (2.0, 4.0)},
            "lacuna": {"duracao_pts": (24, 48)},
            "queda": {"duracao_pts": (18, 36), "delta": (-18.0, -6.0)},
        },
        "MP10_1": {
            "oscilacao": {"duracao_pts": (36, 72), "amp": (4.0, 8.0)},
            "lacuna": {"duracao_pts": (24, 48)},
            "queda": {"duracao_pts": (18, 36), "delta": (-35.0, -10.0)},
        },
        "Temperatura": {
            "oscilacao": {"duracao_pts": (36, 72), "amp": (0.4, 1.2)},
            "lacuna": {"duracao_pts": (24, 48)},
            "queda": {"duracao_pts": (18, 36), "delta": (-3.0, -1.0)},
        },
        "Umidade": {
            "oscilacao": {"duracao_pts": (36, 72), "amp": (3.0, 8.0)},
            "lacuna": {"duracao_pts": (24, 48)},
            "queda": {"duracao_pts": (18, 36), "delta": (-15.0, -5.0)},
        },
    }

    if coluna not in configs:
        raise ValueError(f"Variável sem configuração de falhas: {coluna}")

    return configs[coluna]


def config_falhas_aleatoria(seed, coluna: str):
    """
    Gera uma configuração diferente de falhas para cada rodada,
    mantendo amplitudes e quedas adequadas à escala da variável.
    """
    rng = np.random.default_rng(seed)
    base = config_base_variavel(coluna)

    return {
        "oscilacao": {
            "duracao_pts": base["oscilacao"]["duracao_pts"],
            "n_eventos": int(rng.integers(20, 31)),
            "amp": base["oscilacao"]["amp"],
        },
        "lacuna": {
            "duracao_pts": base["lacuna"]["duracao_pts"],
            "n_eventos": int(rng.integers(20, 31)),
        },
        "queda": {
            "duracao_pts": base["queda"]["duracao_pts"],
            "n_eventos": int(rng.integers(20, 31)),
            "delta": base["queda"]["delta"],
        },
    }


def salvar_config_falhas(config, pasta, nome_arquivo):
    """Salva a configuração de falhas usada em cada rodada."""
    os.makedirs(pasta, exist_ok=True)

    linhas = []
    for falha, params in config.items():
        linha = {"falha": falha}
        linha.update(params)
        linhas.append(linha)

    pd.DataFrame(linhas).to_csv(
        os.path.join(pasta, nome_arquivo),
        index=False
    )


def _criar_pastas_variavel(nome_var_safe):
    pasta_resultados_var = os.path.join(PASTA_RESULTADOS, nome_var_safe)
    pasta_relatorios_var = os.path.join(PASTA_RELATORIOS, nome_var_safe)
    os.makedirs(pasta_resultados_var, exist_ok=True)
    os.makedirs(pasta_relatorios_var, exist_ok=True)
    return pasta_resultados_var, pasta_relatorios_var


def executar_rodadas_variavel(dados, coluna: str, ini, fim, n_rodadas=5):
    """
    Executa o experimento completo para uma variável:
    - 5 rodadas;
    - cenário sem erro;
    - cenário com erro;
    - janelas deslizantes;
    - métricas, médias, gráficos e logs.
    """
    nome_var_safe = NOMES_VARIAVEIS.get(coluna, gerar_nome_base(coluna, ini, fim).split("_")[0])
    pasta_resultados_var, pasta_relatorios_var = _criar_pastas_variavel(nome_var_safe)

    nome_base_periodo = gerar_nome_base(coluna, ini, fim)

    print("\n============================================================")
    print(f"🔎 PROCESSANDO VARIÁVEL: {coluna}")
    print("============================================================")

    df_original = preparar_dados_periodo(dados, coluna, ini, fim)

    if df_original.empty:
        print(f"⚠️ Não há dados no período selecionado para {coluna}.")
        return None, None

    df_original["label"] = "normal"
    df_original["evento_id"] = ""

    resumo_melhores = []
    resumo_eventos = []

    for rodada in range(1, n_rodadas + 1):
        print("\n==============================")
        print(f"🚀 {coluna} | RODADA {rodada}/{n_rodadas}")
        print("==============================")

        pasta_resultados_rodada = os.path.join(pasta_resultados_var, f"rodada_{rodada}")
        pasta_relatorios_rodada = os.path.join(pasta_relatorios_var, f"rodada_{rodada}")

        pasta_resultados_sem = os.path.join(pasta_resultados_rodada, "sem_erro")
        pasta_resultados_com = os.path.join(pasta_resultados_rodada, "com_erro")

        pasta_relatorios_sem = os.path.join(pasta_relatorios_rodada, "sem_erro")
        pasta_relatorios_com = os.path.join(pasta_relatorios_rodada, "com_erro")

        for pasta in [
            pasta_resultados_rodada,
            pasta_relatorios_rodada,
            pasta_resultados_sem,
            pasta_resultados_com,
            pasta_relatorios_sem,
            pasta_relatorios_com,
        ]:
            os.makedirs(pasta, exist_ok=True)

        # Seeds diferentes por variável e rodada, mas reproduzíveis.
        offset_var = abs(hash(coluna)) % 10000
        seed_treino = offset_var + 1000 + rodada
        seed_teste = offset_var + 2000 + rodada

        config_treino = config_falhas_aleatoria(seed_treino, coluna)
        config_teste = config_falhas_aleatoria(seed_teste, coluna)

        salvar_config_falhas(
            config_treino,
            pasta_resultados_rodada,
            f"rodada_{rodada}_config_falhas_treino.csv"
        )
        salvar_config_falhas(
            config_teste,
            pasta_resultados_rodada,
            f"rodada_{rodada}_config_falhas_teste.csv"
        )

        nome_base = f"{nome_base_periodo}_rodada_{rodada}"

        # ==========================
        # CENÁRIO SEM ERRO
        # ==========================
        print("\n📌 Rodando cenário SEM ERRO...")

        df_sem = df_original.copy()
        df_sem["label"] = "normal"
        df_sem["evento_id"] = ""

        df_treino_sem, log_treino_sem = criar_base_com_falhas(
            df_sem.copy(),
            coluna,
            pasta_resultados_sem,
            nome_base=f"{nome_base}_treino_sem_erro",
            incluir_falhas_finais=True,
            seed=seed_treino,
            config_custom=config_treino
        )

        plot_serie_com_labels(
            df_sem,
            coluna,
            pasta_relatorios_sem,
            "serie_sem_erro.png",
            f"Série temporal sem falhas injetadas – {coluna}"
        )

        salvar_distribuicao(
            df_sem,
            pasta_resultados_sem,
            f"{nome_base}_sem_erro"
        )

        resultados_sem = avaliar_modelos_treino_teste_eventos(
            df_treino=df_treino_sem,
            df_teste=df_sem,
            coluna=coluna,
            janelas=list(range(20, 101, 10)),
            passos_cfg=list(range(5, 101, 5)),
            pasta_out=pasta_resultados_sem,
            nome_base=nome_base,
            rotulo_modo="qualquer",
            nome_teste="sem_erro"
        )

        if not resultados_sem["melhores_configuracoes"].empty:
            df_aux = resultados_sem["melhores_configuracoes"].copy()
            df_aux["variavel"] = coluna
            df_aux["rodada"] = rodada
            df_aux["cenario"] = "sem_erro"
            resumo_melhores.append(df_aux)

        eventos_treino_sem = log_treino_sem.copy()
        eventos_treino_sem["variavel"] = coluna
        eventos_treino_sem["rodada"] = rodada
        eventos_treino_sem["cenario"] = "sem_erro"
        eventos_treino_sem["base"] = "treino"
        resumo_eventos.append(eventos_treino_sem)

        # ==========================
        # CENÁRIO COM ERRO
        # ==========================
        print("\n📌 Rodando cenário COM ERRO...")

        df_treino_com, log_treino_com = criar_base_com_falhas(
            df_original.copy(),
            coluna,
            pasta_resultados_com,
            nome_base=f"{nome_base}_treino_com_erro",
            incluir_falhas_finais=False,
            seed=seed_treino,
            config_custom=config_treino
        )

        df_com, log_teste_com = criar_base_com_falhas(
            df_original.copy(),
            coluna,
            pasta_resultados_com,
            nome_base=f"{nome_base}_teste_com_erro",
            incluir_falhas_finais=True,
            seed=seed_teste,
            config_custom=config_teste
        )

        plot_serie_com_labels(
            df_original,
            coluna,
            pasta_relatorios_com,
            "serie_original_sem_falhas.png",
            f"Série temporal original antes da injeção das falhas – {coluna}"
        )

        plot_serie_com_labels(
            df_com,
            coluna,
            pasta_relatorios_com,
            "serie_com_falhas.png",
            f"Série temporal com falhas injetadas – {coluna}"
        )

        gerar_figuras_falhas(
            df_com,
            coluna,
            pasta_relatorios_com
        )

        salvar_distribuicao(
            df_com,
            pasta_resultados_com,
            f"{nome_base}_com_erro"
        )

        resultados_com = avaliar_modelos_treino_teste_eventos(
            df_treino=df_treino_com,
            df_teste=df_com,
            coluna=coluna,
            janelas=list(range(20, 101, 10)),
            passos_cfg=list(range(5, 101, 5)),
            pasta_out=pasta_resultados_com,
            nome_base=nome_base,
            rotulo_modo="qualquer",
            nome_teste="com_erro"
        )

        if not resultados_com["melhores_configuracoes"].empty:
            df_aux = resultados_com["melhores_configuracoes"].copy()
            df_aux["variavel"] = coluna
            df_aux["rodada"] = rodada
            df_aux["cenario"] = "com_erro"
            resumo_melhores.append(df_aux)

        eventos_treino_com = log_treino_com.copy()
        eventos_treino_com["variavel"] = coluna
        eventos_treino_com["rodada"] = rodada
        eventos_treino_com["cenario"] = "com_erro"
        eventos_treino_com["base"] = "treino"
        resumo_eventos.append(eventos_treino_com)

        eventos_teste_com = log_teste_com.copy()
        eventos_teste_com["variavel"] = coluna
        eventos_teste_com["rodada"] = rodada
        eventos_teste_com["cenario"] = "com_erro"
        eventos_teste_com["base"] = "teste"
        resumo_eventos.append(eventos_teste_com)

        log_teste_com.to_csv(
            os.path.join(pasta_resultados_com, f"{nome_base}_quantidade_erros_simulados.csv"),
            index=False
        )

        print("\nEventos injetados no teste COM ERRO desta rodada:")
        print(log_teste_com["falha"].value_counts())

    # ==========================
    # RESUMOS FINAIS DA VARIÁVEL
    # ==========================
    df_medias = pd.DataFrame()
    df_final = pd.DataFrame()

    if resumo_eventos:
        df_eventos = pd.concat(resumo_eventos, ignore_index=True)

        path_eventos = os.path.join(
            pasta_resultados_var,
            f"{nome_base_periodo}_eventos_simulados_todas_rodadas.csv"
        )
        df_eventos.to_csv(path_eventos, index=False)

        df_qtd_eventos = (
            df_eventos
            .groupby(["variavel", "rodada", "cenario", "base", "falha"], as_index=False)
            .size()
            .rename(columns={"size": "quantidade_eventos"})
        )

        path_qtd_eventos = os.path.join(
            pasta_resultados_var,
            f"{nome_base_periodo}_quantidade_eventos_por_rodada.csv"
        )
        df_qtd_eventos.to_csv(path_qtd_eventos, index=False)

        print("\n✅ Quantidade de eventos simulados por rodada salva em:")
        print(os.path.abspath(path_qtd_eventos))

    if resumo_melhores:
        df_final = pd.concat(resumo_melhores, ignore_index=True)

        path_todos = os.path.join(
            pasta_resultados_var,
            f"{nome_base_periodo}_resumo_todas_rodadas.csv"
        )
        df_final.to_csv(path_todos, index=False)

        colunas_media = [
            "accuracy",
            "balanced_accuracy",
            "precision_pos",
            "recall_pos",
            "f1_pos",
            "specificity",
            "eventos_reais_teste",
            "eventos_detectados",
            "diferenca_eventos",
            "fp",
            "fn",
            "tp",
            "tn",
        ]

        colunas_existentes = [c for c in colunas_media if c in df_final.columns]

        df_medias = (
            df_final
            .groupby(["variavel", "cenario", "modelo", "falha"], as_index=False)[colunas_existentes]
            .agg(["mean", "std"])
        )

        # Achata colunas MultiIndex geradas pelo agg.
        df_medias.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
            if isinstance(col, tuple) else str(col)
            for col in df_medias.columns
        ]
        df_medias = df_medias.reset_index(drop=True)

        path_medias = os.path.join(
            pasta_resultados_var,
            f"{nome_base_periodo}_medias_{n_rodadas}_rodadas.csv"
        )
        df_medias.to_csv(path_medias, index=False)

        # Melhor modelo por falha no cenário com erro, útil para escrever a dissertação.
        df_com_erro = df_final[df_final["cenario"] == "com_erro"].copy()
        if not df_com_erro.empty and "f1_pos" in df_com_erro.columns:
            df_melhor = (
                df_com_erro
                .groupby(["variavel", "falha", "modelo"], as_index=False)
                .agg(
                    f1_medio=("f1_pos", "mean"),
                    f1_desvio=("f1_pos", "std"),
                    precisao_media=("precision_pos", "mean"),
                    recall_medio=("recall_pos", "mean"),
                    eventos_reais_medios=("eventos_reais_teste", "mean"),
                    eventos_detectados_medios=("eventos_detectados", "mean"),
                )
                .sort_values(["variavel", "falha", "f1_medio", "recall_medio", "precisao_media"], ascending=[True, True, False, False, False])
                .groupby(["variavel", "falha"], as_index=False)
                .first()
            )

            path_melhores = os.path.join(
                pasta_resultados_var,
                f"{nome_base_periodo}_melhores_modelos_por_falha.csv"
            )
            df_melhor.to_csv(path_melhores, index=False)

            print("\n✅ Melhores modelos por falha salvos em:")
            print(os.path.abspath(path_melhores))

        print("\n✅ Resumo geral de todas as rodadas salvo em:")
        print(os.path.abspath(path_todos))

        print(f"\n✅ Médias das {n_rodadas} rodadas salvas em:")
        print(os.path.abspath(path_medias))
    else:
        print(f"⚠️ Não foi possível gerar o resumo final das métricas para {coluna}.")

    return df_final, df_medias


def executar_rodadas_automaticas(dados, n_rodadas=5):
    """
    Solicita apenas um período e executa automaticamente o experimento
    para todas as variáveis listadas em COLUNAS_ANALISADAS.
    """
    ini, fim = ler_periodo()

    resumo_global = []
    medias_global = []

    for coluna in COLUNAS_ANALISADAS:
        if coluna not in dados.columns:
            print(f"⚠️ Coluna não encontrada nos dados: {coluna}. Pulando...")
            continue

        df_final, df_medias = executar_rodadas_variavel(
            dados=dados,
            coluna=coluna,
            ini=ini,
            fim=fim,
            n_rodadas=n_rodadas
        )

        if df_final is not None and not df_final.empty:
            resumo_global.append(df_final)
        if df_medias is not None and not df_medias.empty:
            medias_global.append(df_medias)

    # ==========================
    # RESUMO GLOBAL MULTIVARIÁVEL
    # ==========================
    nome_periodo = f"{ini.strftime('%Y%m%d_%H%M')}_to_{fim.strftime('%Y%m%d_%H%M')}"

    if resumo_global:
        df_resumo_global = pd.concat(resumo_global, ignore_index=True)
        path_resumo_global = os.path.join(
            PASTA_RESULTADOS,
            f"MULTIVARIAVEL_{nome_periodo}_resumo_todas_rodadas.csv"
        )
        df_resumo_global.to_csv(path_resumo_global, index=False)
        print("\n✅ Resumo global multivariável salvo em:")
        print(os.path.abspath(path_resumo_global))

    if medias_global:
        df_medias_global = pd.concat(medias_global, ignore_index=True)
        path_medias_global = os.path.join(
            PASTA_RESULTADOS,
            f"MULTIVARIAVEL_{nome_periodo}_medias_{n_rodadas}_rodadas.csv"
        )
        df_medias_global.to_csv(path_medias_global, index=False)
        print("\n✅ Médias globais multivariáveis salvas em:")
        print(os.path.abspath(path_medias_global))

    print("\n✅ Execução automática multivariável finalizada.")


# ==============================
# PROGRAMA PRINCIPAL
# ==============================
def main():
    criar_pastas()

    print("\n=== Sistema Automático Multivariável de Simulação + Treinamento ===")
    print("Informe apenas o período. O sistema executará 5 rodadas para:")
    print(", ".join(COLUNAS_ANALISADAS))

    dados = carregar_dados(PASTA_DADOS)

    if dados.empty:
        print("⚠️ Nenhum dado carregado. Verifique a pasta de dados.")
        return

    executar_rodadas_automaticas(dados, n_rodadas=5)


if __name__ == "__main__":
    main()
