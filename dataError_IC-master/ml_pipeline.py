import os
import json
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score
)

# ==============================
# MUDANÇA: inclusão dos outros modelos
# ==============================
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False


# ==============================
# MUDANÇA:
# removido stuck e stuck_at_zero dos modelos
# Agora os modelos trabalham apenas com:
# oscilacao, lacuna e queda
# ==============================
FALHAS_ALVO = ["oscilacao", "lacuna", "queda"]


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
        "savefig.dpi": 300,
    })


def _savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)


def _to_numeric_safe(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _maior_sequencia_true(mask_bool) -> int:
    maior = 0
    atual = 0
    for v in mask_bool:
        if bool(v):
            atual += 1
            if atual > maior:
                maior = atual
        else:
            atual = 0
    return maior


def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return 0.0
    x = x[ok]
    y = y[ok]
    x_mean = x.mean()
    y_mean = y.mean()
    den = np.sum((x - x_mean) ** 2)
    if den == 0:
        return 0.0
    num = np.sum((x - x_mean) * (y - y_mean))
    return float(num / den)


def extrair_atributos_janela(sub: pd.DataFrame, coluna: str) -> dict:
    serie = _to_numeric_safe(sub[coluna]).astype(float)
    vals = serie.to_numpy(dtype=float)

    feats = {}

    feats["n"] = len(vals)
    feats["nan_count"] = int(np.isnan(vals).sum())
    feats["nan_prop"] = float(np.isnan(vals).mean())

    if np.isnan(vals).all():
        feats.update({
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "amplitude": 0.0,
            "last": 0.0,
            "first": 0.0,
            "last_first_diff": 0.0,
            "slope": 0.0,
            "abs_diff_mean": 0.0,
            "abs_diff_max": 0.0,
            "prop_zero": 0.0,
            "prop_quase_zero": 0.0,
            "prop_repetido": 0.0,
            "mudanca_sinal": 0.0,
            "energia_variacao": 0.0,
            "max_seq_zero": 0.0,
            "max_seq_constante": 0.0,
            "fim_is_nan": 1.0,
            "fim_is_zero": 0.0,
        })

        for i in range(len(vals)):
            feats[f"lag_{i+1}"] = 0.0

        return feats

    # preenche NaN localmente apenas para cálculo estatístico
    s_fill = pd.Series(vals).interpolate(limit_direction="both")
    s_fill = s_fill.ffill().bfill()
    vals_fill = s_fill.to_numpy(dtype=float)

    diffs = np.diff(vals_fill)
    abs_diffs = np.abs(diffs)

    zero_mask = np.isclose(vals_fill, 0.0, atol=1e-9)
    quase_zero_mask = np.abs(vals_fill) <= 0.1

    repetido_mask = np.zeros(len(vals_fill), dtype=float)
    if len(vals_fill) > 1:
        repetido_mask[1:] = np.isclose(vals_fill[1:], vals_fill[:-1], atol=1e-9).astype(float)

    sinal = np.sign(diffs)
    mudou_sinal = 0.0
    if len(sinal) > 1:
        mudou_sinal = float(np.sum(sinal[1:] * sinal[:-1] < 0))

    const_mask = np.zeros(len(vals_fill), dtype=bool)
    if len(vals_fill) > 1:
        const_mask[1:] = np.isclose(vals_fill[1:], vals_fill[:-1], atol=1e-9)

    feats.update({
        "mean": float(np.mean(vals_fill)),
        "std": float(np.std(vals_fill)),
        "min": float(np.min(vals_fill)),
        "max": float(np.max(vals_fill)),
        "median": float(np.median(vals_fill)),
        "amplitude": float(np.max(vals_fill) - np.min(vals_fill)),
        "last": float(vals_fill[-1]),
        "first": float(vals_fill[0]),
        "last_first_diff": float(vals_fill[-1] - vals_fill[0]),
        "slope": _slope(vals_fill),
        "abs_diff_mean": float(abs_diffs.mean()) if len(abs_diffs) else 0.0,
        "abs_diff_max": float(abs_diffs.max()) if len(abs_diffs) else 0.0,
        "prop_zero": float(zero_mask.mean()),
        "prop_quase_zero": float(quase_zero_mask.mean()),
        "prop_repetido": float(repetido_mask.mean()),
        "mudanca_sinal": mudou_sinal,
        "energia_variacao": float(abs_diffs.sum()) if len(abs_diffs) else 0.0,
        "max_seq_zero": float(_maior_sequencia_true(zero_mask)),
        "max_seq_constante": float(_maior_sequencia_true(const_mask)),
        "fim_is_nan": float(np.isnan(vals[-1])),
        "fim_is_zero": float(np.isclose(vals_fill[-1], 0.0, atol=1e-9)),
    })

    # usa os valores da própria janela como lags explícitos
    for i, v in enumerate(vals_fill):
        feats[f"lag_{i+1}"] = float(v)

    return feats


def montar_dataset_janelas(
    df: pd.DataFrame,
    coluna: str,
    falha_alvo: str,
    janela: int,
    passo: int,
    rotulo_modo: str = "fim"
) -> pd.DataFrame:
    """    rotulo_modo:
      - 'fim'      -> y=1 se a última leitura da janela for da falha alvo
      - 'qualquer' -> y=1 se qualquer ponto da janela contiver a falha alvo
      - 'maioria'  -> y=1 se mais de 50% da janela for da falha alvo    """

    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    rows = []

    if len(df) < janela:
        return pd.DataFrame()

    # ==============================
    # IMPLEMENTAÇÃO NOVA
    # ==============================
    # Agora a janela móvel está explícita:
    # INI, FIM e avanço manual
    ini = 0
    fim = ini + janela

    while fim <= len(df):
        sub = df.iloc[ini:fim].copy()
        labels = sub["label"].astype(str).tolist()

        if rotulo_modo == "fim":
            y = int(labels[-1] == falha_alvo)
        elif rotulo_modo == "qualquer":
            y = int(falha_alvo in labels)
        elif rotulo_modo == "maioria":
            y = int(sum(l == falha_alvo for l in labels) > len(labels) / 2.0)
        else:
            raise ValueError("rotulo_modo inválido = 'fim', 'qualquer' ou 'maioria'.")

        feats = extrair_atributos_janela(sub, coluna)
        feats["y"] = y
        feats["falha_alvo"] = falha_alvo
        feats["janela"] = int(janela)
        feats["passo"] = int(passo)
        feats["Datetime_ini"] = sub["Datetime"].iloc[0]
        feats["Datetime_fim"] = sub["Datetime"].iloc[-1]

        rows.append(feats)

        ini += passo
        fim = ini + janela

    # IMPLEMENTAÇÃO ANTIGA

    # for fim in range(janela - 1, len(df), passo):
    #     ini = fim - janela + 1
    #     sub = df.iloc[ini:fim + 1].copy()

    ds = pd.DataFrame(rows)
    if ds.empty:
        return ds

    ds = ds.sort_values("Datetime_fim").reset_index(drop=True)
    return ds


def _plot_confusion_binaria(cm, titulo, path_png):
    """
    MELHORIA:
    - antes a matriz mostrava apenas valores absolutos
    - agora mostra também proporção normalizada por linha
    """
    _paper_style()
    fig = plt.figure(figsize=(5.5, 4.6))
    ax = plt.gca()

    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm.astype(float) / row_sums

    im = ax.imshow(cm_norm, vmin=0, vmax=1)

    ax.set_title(titulo)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")

    classes = ["Normal", "Falha"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center", va="center"
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _savefig(path_png)
    plt.close(fig)


def _resolver_passo(passo_cfg, janela):
    """
    Permite usar:
    - inteiro fixo (ex.: 5)
    - string 'n/3'
    """
    if isinstance(passo_cfg, str):
        if passo_cfg.lower() == "n/3":
            return max(1, janela // 3)
        raise ValueError(f"Passo string não suportado: {passo_cfg}")
    return int(passo_cfg)


def _plot_comparacao_janelas(df_resultados_falha: pd.DataFrame, falha: str, pasta_out: str, nome_base: str):
    """
    Gera gráfico comparando janelas para uma falha.
    Usa o melhor resultado de cada janela.
    """
    if df_resultados_falha.empty:
        return None

    df_plot = (
        df_resultados_falha
        .sort_values(["janela", "f1_pos", "recall_pos", "precision_pos"], ascending=[True, False, False, False])
        .groupby("janela", as_index=False)
        .first()
        .copy()
    )

    if df_plot.empty:
        return None

    _paper_style()
    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(df_plot["janela"], df_plot["f1_pos"], marker="o", label="F1 falha")
    plt.plot(df_plot["janela"], df_plot["recall_pos"], marker="s", label="Recall falha")
    plt.plot(df_plot["janela"], df_plot["precision_pos"], marker="^", label="Precisão falha")
    plt.plot(df_plot["janela"], df_plot["balanced_accuracy"], marker="d", label="Balanced Acc.")

    plt.title(f"Comparação de janelas – {falha}")
    plt.xlabel("Tamanho da janela")
    plt.ylabel("Métrica")
    plt.ylim(0, 1.05)
    plt.legend()

    path_png = os.path.join(pasta_out, f"{nome_base}_{falha}_comparacao_janelas.png")
    _savefig(path_png)
    plt.close(fig)
    return path_png


# ==============================
# MUDANÇA:
# função auxiliar para instanciar cada modelo
# ==============================
def _criar_modelo(nome_modelo: str, ratio: float):
    nome_modelo = nome_modelo.lower()

    if nome_modelo == "xgboost":
        if not _HAS_XGB:
            return None
        return XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=max(1.0, float(ratio))
        )

    if nome_modelo == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )

    if nome_modelo == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate="adaptive",
            max_iter=600,
            random_state=42
        )

    if nome_modelo == "catboost":
        if not _HAS_CAT:
            return None
        return CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=42,
            verbose=False
        )

    raise ValueError(f"Modelo não suportado: {nome_modelo}")


def avaliar_modelos_por_falha_e_janela(
    df_base: pd.DataFrame,
    coluna: str,
    janelas=None,
    passos_cfg=None,
    modelos=None,
    n_splits: int = 5,
    pasta_out: str = "resultados",
    nome_base: str = "saida",
    rotulo_modo: str = "fim"
):
    os.makedirs(pasta_out, exist_ok=True)

    if janelas is None:
        janelas = list(range(20, 101, 10))

    if passos_cfg is None:
        passos_cfg = [5, 10, "n/3"]

    # ==============================
    # MUDANÇA:
    # inclusão dos vários modelos
    # ==============================
    if modelos is None:
        modelos = ["xgboost", "random_forest", "mlp", "catboost"]

    resultados = []
    melhores_rows = []
    paths_comparacao = {}

    for falha in FALHAS_ALVO:
        for nome_modelo in modelos:
            melhor_score = -1.0
            melhor_row = None

            for janela in janelas:
                for passo_cfg in passos_cfg:
                    passo = _resolver_passo(passo_cfg, janela)

                    # Evita passos maiores que a janela
                    if passo > janela:
                        continue

                    ds = montar_dataset_janelas(
                        df=df_base,
                        coluna=coluna,
                        falha_alvo=falha,
                        janela=janela,
                        passo=passo,
                        rotulo_modo=rotulo_modo
                    )

                    if ds.empty:
                        continue

                    y = ds["y"].astype(int).to_numpy()

                    X = ds.drop(
                        columns=[
                            "y", "falha_alvo", "janela", "passo",
                            "Datetime_ini", "Datetime_fim"
                        ],
                        errors="ignore"
                    ).copy()

                    for c in X.columns:
                        X[c] = pd.to_numeric(X[c], errors="coerce")
                    X = X.fillna(0)

                    positivos = int(y.sum())
                    negativos = int((y == 0).sum())

                    if positivos < 2 or negativos < 2:
                        continue

                    # Se tiver poucas amostras, reduz automaticamente os splits
                    n_splits_ajustado = min(n_splits, max(2, len(X) // 20))
                    if n_splits_ajustado < 2:
                        continue

                    tscv = TimeSeriesSplit(n_splits=n_splits_ajustado)

                    y_true_all = []
                    y_pred_all = []
                    splits_validos = 0

                    for split_id, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
                        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
                        ytr, yte = y[tr_idx], y[te_idx]

                        if len(np.unique(ytr)) < 2:
                            continue
                        if len(np.unique(yte)) < 2:
                            continue

                        pos = int((ytr == 1).sum())
                        neg = int((ytr == 0).sum())
                        ratio = neg / max(1, pos)
                        ratio = ratio * 1.5

                        modelo = _criar_modelo(nome_modelo, ratio)
                        if modelo is None:
                            continue

                        modelo.fit(Xtr, ytr)

                        # ==============================
                        # MUDANÇA:
                        # limiar especial só para XGBoost
                        # nos demais modelos usa predict normal
                        # ==============================
                        if nome_modelo.lower() == "xgboost":
                            probs = modelo.predict_proba(Xte)[:, 1]
                            ypred = (probs >= 0.30).astype(int)
                        elif nome_modelo.lower() == "catboost":
                            probs = modelo.predict_proba(Xte)[:, 1]
                            ypred = (probs >= 0.50).astype(int)
                        else:
                            ypred = modelo.predict(Xte).astype(int)

                        y_true_all.extend(yte.tolist())
                        y_pred_all.extend(ypred.tolist())
                        splits_validos += 1

                    if splits_validos == 0:
                        continue

                    y_true_all = np.array(y_true_all, dtype=int)
                    y_pred_all = np.array(y_pred_all, dtype=int)

                    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp + 1e-9)

                    row = {
                        "modelo": nome_modelo,
                        "falha": falha,
                        "janela": int(janela),
                        "passo_cfg": str(passo_cfg),
                        "passo_real": int(passo),
                        "amostras": int(len(ds)),
                        "positivos": int(positivos),
                        "negativos": int(negativos),
                        "splits_validos": int(splits_validos),
                        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
                        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
                        "precision_pos": float(precision_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)),
                        "recall_pos": float(recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)),
                        "f1_pos": float(f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)),
                        "specificity": float(specificity),
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    }
                    resultados.append(row)

                    # score combinando F1 e recall para priorizar detecção da falha
                    score = 0.7 * row["f1_pos"] + 0.3 * row["recall_pos"]

                    if score > melhor_score:
                        melhor_score = score
                        melhor_row = row.copy()

                        path_cm = os.path.join(
                            pasta_out,
                            f"{nome_base}_{nome_modelo}_{falha}_jan{janela}_passo{passo}_cm.png"
                        )
                        _plot_confusion_binaria(
                            cm,
                            f"{nome_modelo} | {falha} | janela={janela} | passo={passo}",
                            path_cm
                        )
                        melhor_row["path_cm"] = path_cm

            if melhor_row is not None:
                melhores_rows.append(melhor_row)

    df_resultados = pd.DataFrame(resultados)
    df_melhores = pd.DataFrame(melhores_rows)

    path_resultados = os.path.join(pasta_out, f"{nome_base}_resultados_todos.csv")
    path_melhores = os.path.join(pasta_out, f"{nome_base}_melhores_configuracoes.csv")
    path_json = os.path.join(pasta_out, f"{nome_base}_melhores_configuracoes.json")

    if not df_resultados.empty:
        df_resultados = df_resultados.sort_values(
            ["modelo", "falha", "janela", "passo_real"],
            ascending=[True, True, True, True]
        ).reset_index(drop=True)
        df_resultados.to_csv(path_resultados, index=False)

        # ==============================
        # MUDANÇA:
        # gera tabelas separadas no formato mais próximo do que o professor pediu
        # uma por falha + modelo + janela
        # ==============================
        pasta_tabelas = os.path.join(pasta_out, "tabelas_metricas")
        os.makedirs(pasta_tabelas, exist_ok=True)

        for falha in FALHAS_ALVO:
            for nome_modelo in modelos:
                df_sub = df_resultados[
                    (df_resultados["falha"] == falha) &
                    (df_resultados["modelo"] == nome_modelo)
                ].copy()

                if df_sub.empty:
                    continue

                for janela in sorted(df_sub["janela"].unique()):
                    bloco = df_sub[df_sub["janela"] == janela].copy()
                    if bloco.empty:
                        continue

                    tabela = bloco[[
                        "passo_real", "tp", "fp", "tn", "fn",
                        "accuracy", "precision_pos", "recall_pos", "f1_pos", "balanced_accuracy"
                    ]].copy()

                    tabela = tabela.rename(columns={
                        "passo_real": "Deslocamento",
                        "tp": "TP",
                        "fp": "FP",
                        "tn": "TN",
                        "fn": "FN",
                        "accuracy": "ACC",
                        "precision_pos": "Precision",
                        "recall_pos": "Recall",
                        "f1_pos": "F1",
                        "balanced_accuracy": "Balanced_ACC"
                    })

                    nome_csv = f"{nome_base}_{falha}_{nome_modelo}_janela_{janela}.csv"
                    tabela.to_csv(os.path.join(pasta_tabelas, nome_csv), index=False)

        # Gera gráfico comparando janelas por falha e modelo
        for falha in FALHAS_ALVO:
            for nome_modelo in modelos:
                df_falha = df_resultados[
                    (df_resultados["falha"] == falha) &
                    (df_resultados["modelo"] == nome_modelo)
                ].copy()

                if df_falha.empty:
                    continue

                path_cmp = _plot_comparacao_janelas(
                    df_falha,
                    f"{falha}_{nome_modelo}",
                    pasta_out,
                    nome_base
                )
                if path_cmp:
                    paths_comparacao[f"{falha}_{nome_modelo}"] = path_cmp

    if not df_melhores.empty:
        df_melhores = df_melhores.sort_values(["modelo", "falha"]).reset_index(drop=True)
        df_melhores.to_csv(path_melhores, index=False)

        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(df_melhores.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    plt.close("all")

    return {
        "resultados_todos": df_resultados,
        "melhores_configuracoes": df_melhores,
        "paths": {
            "resultados_todos_csv": path_resultados,
            "melhores_configuracoes_csv": path_melhores,
            "melhores_configuracoes_json": path_json,
            "comparacao_janelas": paths_comparacao,
            "pasta_tabelas_metricas": os.path.join(pasta_out, "tabelas_metricas")
        }
    }


# ==============================
# MANTIDO:
# nome antigo da função para não quebrar código antigo
# Agora ela chama a nova função com vários modelos
# ==============================
def avaliar_xgb_por_falha_e_janela(
    df_base: pd.DataFrame,
    coluna: str,
    janelas=None,
    passos_cfg=None,
    n_splits: int = 5,
    pasta_out: str = "resultados",
    nome_base: str = "saida",
    rotulo_modo: str = "fim"
):
    return avaliar_modelos_por_falha_e_janela(
        df_base=df_base,
        coluna=coluna,
        janelas=janelas,
        passos_cfg=passos_cfg,
        modelos=["xgboost", "random_forest", "mlp", "catboost"],
        n_splits=n_splits,
        pasta_out=pasta_out,
        nome_base=nome_base,
        rotulo_modo=rotulo_modo
    )