# ml_pipeline.py

import os
import json
import math
import numpy as np
import pandas as pd
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

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


FALHAS_ALVO = ["oscilacao", "stuck", "stuck_at_zero", "lacuna", "queda"]


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
        # janela totalmente vazia
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
    """
      - 'fim'      -> y=1 se a ÚLTIMA leitura da janela for da falha alvo
      - 'qualquer' -> y=1 se QUALQUER ponto da janela contiver a falha alvo
      - 'maioria'  -> y=1 se mais de 50% da janela for da falha alvo
    """
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    rows = []

    if len(df) < janela:
        return pd.DataFrame()

    for fim in range(janela - 1, len(df), passo):
        ini = fim - janela + 1
        sub = df.iloc[ini:fim + 1].copy()

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
        feats["Datetime_fim"] = sub["Datetime"].iloc[-1]
        rows.append(feats)

    ds = pd.DataFrame(rows)
    if ds.empty:
        return ds

    ds = ds.sort_values("Datetime_fim").reset_index(drop=True)
    return ds


def _plot_confusion_binaria(cm, titulo, path_png):
    _paper_style()
    fig = plt.figure(figsize=(5.0, 4.2))
    ax = plt.gca()

    ax.imshow(cm)
    ax.set_title(titulo)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")

    classes = ["negativo", "positivo"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    _savefig(path_png)
    plt.close(fig)


def _resolver_passo(passo_cfg, janela):
    if isinstance(passo_cfg, str):
        if passo_cfg.lower() == "n/3":
            return max(1, janela // 3)
        raise ValueError(f"Passo string não suportado: {passo_cfg}")
    return int(passo_cfg)


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
    if not _HAS_XGB:
        raise ImportError(

        )

    os.makedirs(pasta_out, exist_ok=True)

    if janelas is None:
        janelas = list(range(10, 101, 10))

    if passos_cfg is None:
        passos_cfg = [1, 5, "n/3"]

    resultados = []
    melhores_rows = []

    for falha in FALHAS_ALVO:
        melhor_f1 = -1.0
        melhor_row = None

        for janela in janelas:
            for passo_cfg in passos_cfg:
                passo = _resolver_passo(passo_cfg, janela)

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
                X = ds.drop(columns=["y", "falha_alvo", "janela", "passo", "Datetime_fim"]).copy()

                for c in X.columns:
                    X[c] = pd.to_numeric(X[c], errors="coerce")
                X = X.fillna(0)

                positivos = int(y.sum())
                negativos = int((y == 0).sum())

                # precisa haver exemplos das duas classes
                if positivos < 2 or negativos < 2:
                    continue

                tscv = TimeSeriesSplit(n_splits=n_splits)

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

                    ratio = (ytr == 0).sum() / max(1, (ytr == 1).sum())

                    modelo = XGBClassifier(
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

                    modelo.fit(Xtr, ytr)
                    ypred = modelo.predict(Xte).astype(int)

                    y_true_all.extend(yte.tolist())
                    y_pred_all.extend(ypred.tolist())
                    splits_validos += 1

                if splits_validos == 0:
                    continue

                y_true_all = np.array(y_true_all, dtype=int)
                y_pred_all = np.array(y_pred_all, dtype=int)

                cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

                row = {
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
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                }
                resultados.append(row)

                if row["f1_pos"] > melhor_f1:
                    melhor_f1 = row["f1_pos"]
                    melhor_row = row.copy()

                    path_cm = os.path.join(
                        pasta_out,
                        f"{nome_base}_{falha}_jan{janela}_passo{passo}_cm.png"
                    )
                    _plot_confusion_binaria(
                        cm,
                        f"{falha} | janela={janela} | passo={passo}",
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
            ["falha", "f1_pos", "recall_pos", "precision_pos"],
            ascending=[True, False, False, False]
        ).reset_index(drop=True)
        df_resultados.to_csv(path_resultados, index=False)

    if not df_melhores.empty:
        df_melhores = df_melhores.sort_values("falha").reset_index(drop=True)
        df_melhores.to_csv(path_melhores, index=False)

        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(df_melhores.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    return {
        "resultados_todos": df_resultados,
        "melhores_configuracoes": df_melhores,
        "paths": {
            "resultados_todos_csv": path_resultados,
            "melhores_configuracoes_csv": path_melhores,
            "melhores_configuracoes_json": path_json
        }
    }