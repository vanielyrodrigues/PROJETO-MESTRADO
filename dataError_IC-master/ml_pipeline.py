# ml_pipeline.py

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



# CARACTERÍSTICAS TEMPORAIS

def criar_features_temporais(df: pd.DataFrame, coluna: str, lags: int = 12) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("Datetime")

    for k in range(1, lags + 1):
        df[f"{coluna}_lag_{k}"] = df[coluna].shift(k)

    df[f"{coluna}_diff_abs"] = df[coluna].diff().abs()

    win = max(3, min(12, lags))
    df[f"{coluna}_roll_mean"] = df[coluna].rolling(window=win, min_periods=1).mean()
    df[f"{coluna}_roll_std"] = df[coluna].rolling(window=win, min_periods=1).std()

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(str)

    return df


# PLOTS (paper style)

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


# MODELOS

def _build_models(random_state=42):
    models = {
        "RF": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced_subsample"
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=600,
            random_state=random_state
        )
    }

    if _HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1
        )

    if _HAS_CAT:
        models["CAT"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.08,
            depth=6,
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False
        )

    return models



# AVALIAÇÃO

def avaliar_modelos(
    df_feat: pd.DataFrame,
    col_alvo: str,
    lags: int = 12,
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

    # sem warning
    X = X.ffill().bfill().fillna(0)

    # encoder GLOBAL (só para referência e para montar as matrizes finais)
    le = LabelEncoder()
    le.fit(sorted(y.unique().tolist()))
    classes = le.classes_.tolist()
    y_enc_global = le.transform(y)

    base_models = _build_models(random_state=42)
    if not _HAS_XGB:
        print("⚠️ XGBoost não está disponível (pacote xgboost não encontrado).")
    if not _HAS_CAT:
        print("⚠️ CatBoost não está disponível (pacote catboost não encontrado).")

    agg_conf = {m: np.zeros((len(classes), len(classes)), dtype=int) for m in base_models}
    agg_metrics = {m: [] for m in base_models}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits_ok = 0

    for split_id, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr_g, yte_g = y_enc_global[tr_idx], y_enc_global[te_idx]  # índices globais

        if len(np.unique(ytr_g)) < 2 or len(np.unique(yte_g)) < 2:
            print(f"⚠️ Split {split_id}: treino/teste com poucas classes. Pulando.")
            continue

        splits_ok += 1

        for nome, base_model in base_models.items():
            modelo = clone(base_model)


            # FIX DEFINITIVO DO XGB: re-encode local 0..K-1 no split

            if nome == "XGB":
                classes_presentes = np.unique(ytr_g)  # ex.: [0, 1, 3]
                map_g2l = {g: i for i, g in enumerate(classes_presentes)}  # {0:0, 1:1, 3:2}
                map_l2g = {i: g for g, i in map_g2l.items()}

                ytr_l = np.array([map_g2l[g] for g in ytr_g], dtype=int)
                # yte pode ter classes fora do treino; isso é ok, só não dá pra "transformar" no local
                # a gente mantém yte global e só converte ypred para global depois

                modelo.set_params(objective="multi:softprob", num_class=len(classes_presentes))
                modelo.fit(Xtr, ytr_l)

                ypred_l = modelo.predict(Xte).astype(int)
                ypred_g = np.array([map_l2g[i] for i in ypred_l], dtype=int)

            else:
                # outros modelos aceitam labels globais sem problema
                modelo.fit(Xtr, ytr_g)
                ypred_g = modelo.predict(Xte).astype(int)

            # matriz de confusão GLOBAL (tamanho fixo, sem quebrar)
            cm = confusion_matrix(yte_g, ypred_g, labels=np.arange(len(classes)))
            agg_conf[nome] += cm

            # métricas globais do split
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
            "Nenhum split válido foi gerado (treino/teste com classes insuficientes). "
            "Aumente a janela temporal ou garanta falhas também no trecho final (teste)."
        )

    # métricas por classe a partir da confusão agregada
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
            y_true_flat, y_pred_flat, labels=np.arange(len(classes)), zero_division=0
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
