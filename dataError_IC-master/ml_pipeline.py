#PIPELINE

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.base import clone

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


def criar_features_temporais(df: pd.DataFrame, coluna: str, lags: int = 18) -> pd.DataFrame:
    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    serie = pd.to_numeric(df[coluna], errors="coerce")

    # ausência
    df[f"{coluna}_is_nan"] = serie.isna().astype(int)

    # lags
    for k in range(1, lags + 1):
        df[f"{coluna}_lag_{k}"] = serie.shift(k)

    # diferenças
    df[f"{coluna}_diff_1"] = serie.diff(1)
    df[f"{coluna}_diff_2"] = serie.diff(2)
    df[f"{coluna}_diff_3"] = serie.diff(3)

    df[f"{coluna}_diff_abs_1"] = serie.diff(1).abs()
    df[f"{coluna}_diff_abs_2"] = serie.diff(2).abs()
    df[f"{coluna}_diff_abs_3"] = serie.diff(3).abs()

    # rolling
    win = max(5, lags)

    df[f"{coluna}_roll_mean"] = serie.rolling(win, min_periods=1).mean()
    df[f"{coluna}_roll_std"] = serie.rolling(win, min_periods=1).std()
    df[f"{coluna}_roll_min"] = serie.rolling(win, min_periods=1).min()
    df[f"{coluna}_roll_max"] = serie.rolling(win, min_periods=1).max()
    df[f"{coluna}_roll_median"] = serie.rolling(win, min_periods=1).median()

    df[f"{coluna}_amplitude"] = df[f"{coluna}_roll_max"] - df[f"{coluna}_roll_min"]

    # zero e repetição
    df[f"{coluna}_is_zero"] = serie.fillna(np.nan).eq(0).astype(float)
    df[f"{coluna}_prop_zero"] = df[f"{coluna}_is_zero"].rolling(win, min_periods=1).mean()

    df[f"{coluna}_igual_anterior"] = (serie == serie.shift(1)).astype(float)
    df[f"{coluna}_prop_repetido"] = df[f"{coluna}_igual_anterior"].rolling(win, min_periods=1).mean()

    # nan na janela
    df[f"{coluna}_nan_count"] = df[f"{coluna}_is_nan"].rolling(win, min_periods=1).sum()
    df[f"{coluna}_nan_prop"] = df[f"{coluna}_is_nan"].rolling(win, min_periods=1).mean()

    # mudanças de sinal
    diff1 = serie.diff()
    sinal = np.sign(diff1)
    mudou_sinal = (sinal * sinal.shift(1) < 0).astype(float)
    df[f"{coluna}_mudanca_sinal"] = mudou_sinal.rolling(win, min_periods=1).sum()

    # energia da variação
    df[f"{coluna}_energia_variacao"] = diff1.abs().rolling(win, min_periods=1).sum()

    # tendência
    df[f"{coluna}_tendencia"] = serie.diff(win)

    # distância da média local
    df[f"{coluna}_desvio_media_local"] = (serie - df[f"{coluna}_roll_mean"]).abs()

    # coeficiente de variação local
    df[f"{coluna}_coef_var_local"] = (
        df[f"{coluna}_roll_std"] / (df[f"{coluna}_roll_mean"].abs() + 1e-6)
    )

    # trecho constante
    df[f"{coluna}_roll_std_fill"] = df[f"{coluna}_roll_std"].fillna(0)
    df[f"{coluna}_quase_constante"] = (df[f"{coluna}_roll_std_fill"] < 1e-6).astype(float)

    # =============================
    # FEATURES ESPECÍFICAS PARA STUCK-AT-ZERO
    # =============================
    zero_mask = serie.fillna(np.nan).eq(0)

    tempo_zero_continuo = []
    cont = 0
    for v in zero_mask:
        if v:
            cont += 1
        else:
            cont = 0
        tempo_zero_continuo.append(cont)

    df[f"{coluna}_tempo_zero_continuo"] = tempo_zero_continuo

    quase_zero_mask = serie.fillna(np.nan).abs().le(0.1)

    tempo_quase_zero_continuo = []
    cont_qz = 0
    for v in quase_zero_mask:
        if v:
            cont_qz += 1
        else:
            cont_qz = 0
        tempo_quase_zero_continuo.append(cont_qz)

    df[f"{coluna}_tempo_quase_zero_continuo"] = tempo_quase_zero_continuo

    df[f"{coluna}_zero_count_janela"] = zero_mask.astype(float).rolling(win, min_periods=1).sum()

    df[f"{coluna}_janela_toda_zero"] = (
        df[f"{coluna}_zero_count_janela"] >= win
    ).astype(float)

    df[f"{coluna}_janela_quase_toda_zero"] = (
        df[f"{coluna}_prop_zero"] >= 0.8
    ).astype(float)

    df[f"{coluna}_zero_e_baixa_var"] = (
        (df[f"{coluna}_prop_zero"] >= 0.5) &
        (df[f"{coluna}_roll_std_fill"] < 1e-6)
    ).astype(float)

    df[f"{coluna}_zero_diff_abs"] = (
        df[f"{coluna}_is_zero"] * df[f"{coluna}_diff_abs_1"].fillna(0)
    )

    df[f"{coluna}_score_stuck_zero"] = (
        0.35 * df[f"{coluna}_prop_zero"].fillna(0) +
        0.35 * (df[f"{coluna}_tempo_zero_continuo"] / max(1, win)) +
        0.30 * (1 - np.clip(df[f"{coluna}_roll_std_fill"], 0, 1))
    )

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(str)

    return df


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


def _plot_confusion(cm, classes, title, path_png):
    _paper_style()
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = plt.gca()

    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    _savefig(path_png)
    plt.close(fig)


def _plot_f1_por_classe(df_metricas_por_classe: pd.DataFrame, path_png: str):
    _paper_style()
    fig = plt.figure(figsize=(12, 4.2))
    ax = plt.gca()

    modelos = sorted(df_metricas_por_classe["modelo"].unique().tolist())
    classes = sorted(df_metricas_por_classe["classe"].unique().tolist())

    x = np.arange(len(classes))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(modelos))

    for off, m in zip(offsets, modelos):
        sub = df_metricas_por_classe[df_metricas_por_classe["modelo"] == m]
        f1s = []
        for c in classes:
            row = sub[sub["classe"] == c]
            f1s.append(float(row["f1"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + off, f1s, width=width, label=m)

    ax.set_title("F1-score por classe (comparação entre modelos)")
    ax.set_xlabel("Classe")
    ax.set_ylabel("F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.legend(loc="best")

    _savefig(path_png)
    plt.close(fig)


def _build_models(random_state=42):
    models = {
        "RF": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced_subsample",
            max_depth=None,
            min_samples_leaf=1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=800,
            random_state=random_state
        )
    }

    if _HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss"
        )

    if _HAS_CAT:
        models["CAT"] = CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False
        )

    return models


def avaliar_modelos(
    df_feat: pd.DataFrame,
    col_alvo: str,
    lags: int = 18,
    n_splits: int = 5,
    salvar_saidas: bool = True,
    pasta_out: str = "resultados",
    nome_base: str = "saida"
):
    os.makedirs(pasta_out, exist_ok=True)

    df = df_feat.copy().sort_values("Datetime").reset_index(drop=True)

    cols_excluir = {"Datetime", "label"}
    feature_cols = [c for c in df.columns if c not in cols_excluir]

    X = df[feature_cols].copy()
    y = df["label"].astype(str).copy()

    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].ffill().bfill()
    X = X.fillna(0)

    le = LabelEncoder()
    le.fit(sorted(y.unique().tolist()))
    classes = le.classes_.tolist()
    y_enc_global = le.transform(y)

    base_models = _build_models(random_state=42)

    if not _HAS_XGB:
        print("⚠️ XGBoost não está disponível.")
    if not _HAS_CAT:
        print("⚠️ CatBoost não está disponível.")

    agg_conf = {m: np.zeros((len(classes), len(classes)), dtype=int) for m in base_models}
    agg_metrics = {m: [] for m in base_models}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits_ok = 0

    for split_id, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr_g, yte_g = y_enc_global[tr_idx], y_enc_global[te_idx]

        if len(np.unique(ytr_g)) < 2 or len(np.unique(yte_g)) < 2:
            print(f"⚠️ Split {split_id}: treino/teste com poucas classes. Pulando.")
            continue

        splits_ok += 1

        for nome, base_model in base_models.items():
            modelo = clone(base_model)

            if nome == "XGB":
                classes_presentes = np.unique(ytr_g)
                map_g2l = {g: i for i, g in enumerate(classes_presentes)}
                map_l2g = {i: g for g, i in map_g2l.items()}

                ytr_l = np.array([map_g2l[g] for g in ytr_g], dtype=int)

                modelo.set_params(
                    objective="multi:softprob",
                    num_class=len(classes_presentes)
                )
                modelo.fit(Xtr, ytr_l)

                ypred_l = modelo.predict(Xte).astype(int)
                ypred_g = np.array([map_l2g[i] for i in ypred_l], dtype=int)
            else:
                modelo.fit(Xtr, ytr_g)
                ypred_g = modelo.predict(Xte).astype(int)

            cm = confusion_matrix(yte_g, ypred_g, labels=np.arange(len(classes)))
            agg_conf[nome] += cm

            acc = float(accuracy_score(yte_g, ypred_g))

            prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
                yte_g, ypred_g, average="macro", zero_division=0
            )
            prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
                yte_g, ypred_g, average="weighted", zero_division=0
            )

            agg_metrics[nome].append({
                "split": split_id,
                "accuracy": acc,
                "macro_f1": float(f1_m),
                "weighted_f1": float(f1_w),
                "macro_precision": float(prec_m),
                "macro_recall": float(rec_m),
                "weighted_precision": float(prec_w),
                "weighted_recall": float(rec_w),
            })

    if splits_ok == 0:
        raise ValueError(
            "Nenhum split válido foi gerado. "
            "Aumente a janela temporal ou garanta falhas no trecho final."
        )

    rows_por_classe = []
    rows_resumo = []

    for nome in base_models.keys():
        cm = agg_conf[nome]

        y_true_flat = []
        y_pred_flat = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                n = cm[i, j]
                if n > 0:
                    y_true_flat.extend([i] * n)
                    y_pred_flat.extend([j] * n)

        y_true_flat = np.array(y_true_flat, dtype=int)
        y_pred_flat = np.array(y_pred_flat, dtype=int)

        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true_flat,
            y_pred_flat,
            labels=np.arange(len(classes)),
            zero_division=0
        )

        for c_idx, c_name in enumerate(classes):
            rows_por_classe.append({
                "modelo": nome,
                "classe": c_name,
                "precision": float(prec[c_idx]),
                "recall": float(rec[c_idx]),
                "f1": float(f1[c_idx]),
                "support": int(sup[c_idx])
            })

        dfm = pd.DataFrame(agg_metrics[nome])
        rows_resumo.append({
            "modelo": nome,
            "accuracy": float(dfm["accuracy"].mean()) if len(dfm) else 0.0,
            "macro_f1": float(dfm["macro_f1"].mean()) if len(dfm) else 0.0,
            "weighted_f1": float(dfm["weighted_f1"].mean()) if len(dfm) else 0.0
        })

    df_metricas_por_classe = pd.DataFrame(rows_por_classe)
    df_resumo_modelos = pd.DataFrame(rows_resumo)

    out_metricas = os.path.join(pasta_out, f"{nome_base}_metricas_por_classe.csv")
    out_resumo = os.path.join(pasta_out, f"{nome_base}_resumo_modelos.csv")
    out_f1 = os.path.join(pasta_out, f"{nome_base}_f1_por_classe.png")
    out_json = os.path.join(pasta_out, f"{nome_base}_resultados.json")

    if salvar_saidas:
        df_metricas_por_classe.to_csv(out_metricas, index=False)
        df_resumo_modelos.to_csv(out_resumo, index=False)

        for nome in base_models.keys():
            cm = agg_conf[nome]
            out_cm = os.path.join(pasta_out, f"{nome_base}_confusion_{nome}.png")
            _plot_confusion(cm, classes, f"Confusion Matrix - {nome}", out_cm)

        _plot_f1_por_classe(df_metricas_por_classe, out_f1)

        payload = {
            "nome_base": nome_base,
            "classes": classes,
            "splits_validos": splits_ok,
            "resumo_modelos": df_resumo_modelos.to_dict(orient="records"),
            "metricas_por_classe": df_metricas_por_classe.to_dict(orient="records")
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "classes": classes,
        "splits_validos": splits_ok,
        "metricas_por_classe": df_metricas_por_classe,
        "resumo_modelos": df_resumo_modelos,
        "paths": {
            "metricas_por_classe_csv": out_metricas,
            "resumo_modelos_csv": out_resumo,
            "f1_por_classe_png": out_f1,
            "resultados_json": out_json
        }
    }