#ml_pipeline
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score
)

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
            maior = max(maior, atual)
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

    feats = {
        "n": len(vals),
        "nan_count": int(np.isnan(vals).sum()),
        "nan_prop": float(np.isnan(vals).mean()) if len(vals) else 0.0,
    }

    if len(vals) == 0 or np.isnan(vals).all():
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

    s_fill = pd.Series(vals).interpolate(limit_direction="both").ffill().bfill()
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
    rotulo_modo:
      - 'fim'      -> y=1 se a última leitura da janela for da falha alvo
      - 'qualquer' -> y=1 se qualquer ponto da janela contiver a falha alvo
      - 'maioria'  -> y=1 se mais de 50% da janela for da falha alvo
    """

    df = df.copy().sort_values("Datetime").reset_index(drop=True)

    rows = []

    if len(df) < janela:
        return pd.DataFrame()

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

    ds = pd.DataFrame(rows)
    if ds.empty:
        return ds

    return ds.sort_values("Datetime_fim").reset_index(drop=True)


def contar_eventos_binarios(vetor) -> int:
    """
    Contagem solicitada pelo professor:
    sequências consecutivas de 1 contam como um único evento.
    """
    normal = True
    cont = 0

    for valor in vetor:
        if int(valor) == 1:
            if normal:
                cont += 1
                normal = False
        else:
            normal = True

    return int(cont)


def _plot_confusion_binaria(cm, titulo, path_png):
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


def _plot_predicoes_temporais(ds_teste: pd.DataFrame, y_true, y_pred, falha: str, path_png: str):
    _paper_style()
    fig = plt.figure(figsize=(12, 4))

    x = pd.to_datetime(ds_teste["Datetime_fim"])
    plt.plot(x, y_true, linewidth=1.5, label="Real")
    plt.plot(x, y_pred, linewidth=1.2, linestyle="--", label="Predito")

    plt.title(f"Eventos reais x detectados – {falha}")
    plt.xlabel("Tempo final da janela")
    plt.ylabel("0=Normal | 1=Falha")
    plt.yticks([0, 1], ["Normal", "Falha"])
    plt.legend()

    _savefig(path_png)
    plt.close(fig)


def _resolver_passo(passo_cfg, janela):
    if isinstance(passo_cfg, str):
        if passo_cfg.lower() == "n/3":
            return max(1, janela // 3)
        raise ValueError(f"Passo string não suportado: {passo_cfg}")
    return int(passo_cfg)


def _plot_comparacao_janelas(df_resultados_falha: pd.DataFrame, falha: str, pasta_out: str, nome_base: str):
    if df_resultados_falha.empty:
        return None

    df_plot = (
        df_resultados_falha
        .sort_values(
            ["janela", "f1_pos", "recall_pos", "precision_pos"],
            ascending=[True, False, False, False]
        )
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


def _preparar_X_y(ds: pd.DataFrame):
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

    return X, y


def avaliar_modelos_treino_teste_eventos(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    coluna: str,
    janelas=None,
    passos_cfg=None,
    modelos=None,
    pasta_out: str = "resultados",
    nome_base: str = "saida",
    rotulo_modo: str = "fim",
    nome_teste: str = "teste"
):
    """
    Fluxo novo pedido pelo professor:
    1) Treina em uma base com falhas.
    2) Aplica na base SEM falhas ou COM falhas.
    3) Conta eventos contínuos, e não apenas janelas.
    """

    os.makedirs(pasta_out, exist_ok=True)

    if janelas is None:
        janelas = list(range(20, 101, 10))

    if passos_cfg is None:
        passos_cfg = [5, 10, "n/3"]

    if modelos is None:
        modelos = ["xgboost", "random_forest", "mlp", "catboost"]

    resultados = []
    melhores_rows = []
    paths_comparacao = {}
    paths_temporais = {}

    for falha in FALHAS_ALVO:
        for nome_modelo in modelos:
            melhor_score = -1.0
            melhor_row = None

            for janela in janelas:
                for passo_cfg in passos_cfg:
                    passo = _resolver_passo(passo_cfg, janela)

                    if passo > janela:
                        continue

                    ds_treino = montar_dataset_janelas(
                        df=df_treino,
                        coluna=coluna,
                        falha_alvo=falha,
                        janela=janela,
                        passo=passo,
                        rotulo_modo=rotulo_modo
                    )

                    ds_teste = montar_dataset_janelas(
                        df=df_teste,
                        coluna=coluna,
                        falha_alvo=falha,
                        janela=janela,
                        passo=passo,
                        rotulo_modo=rotulo_modo
                    )

                    if ds_treino.empty or ds_teste.empty:
                        continue

                    Xtr, ytr = _preparar_X_y(ds_treino)
                    Xte, yte = _preparar_X_y(ds_teste)

                    positivos = int(ytr.sum())
                    negativos = int((ytr == 0).sum())

                    if positivos < 2 or negativos < 2:
                        continue

                    pos = int((ytr == 1).sum())
                    neg = int((ytr == 0).sum())
                    ratio = (neg / max(1, pos)) * 1.5

                    modelo = _criar_modelo(nome_modelo, ratio)
                    if modelo is None:
                        continue

                    modelo.fit(Xtr, ytr)

                    if nome_modelo.lower() == "xgboost":
                        probs = modelo.predict_proba(Xte)[:, 1]
                        ypred = (probs >= 0.30).astype(int)
                    elif nome_modelo.lower() == "catboost":
                        probs = modelo.predict_proba(Xte)[:, 1]
                        ypred = (probs >= 0.50).astype(int)
                    else:
                        ypred = modelo.predict(Xte).astype(int)

                    cm = confusion_matrix(yte, ypred, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()

                    eventos_reais = contar_eventos_binarios(yte)
                    eventos_detectados = contar_eventos_binarios(ypred)
                    diferenca_eventos = eventos_detectados - eventos_reais

                    specificity = tn / (tn + fp + 1e-9)

                    # Quando não há positivos no teste sem erro, algumas métricas ficam naturalmente 0.
                    row = {
                        "teste": nome_teste,
                        "modelo": nome_modelo,
                        "falha": falha,
                        "janela": int(janela),
                        "passo_cfg": str(passo_cfg),
                        "passo_real": int(passo),
                        "amostras_treino": int(len(ds_treino)),
                        "amostras_teste": int(len(ds_teste)),
                        "positivos_treino": int(positivos),
                        "negativos_treino": int(negativos),
                        "positivos_teste": int(yte.sum()),
                        "negativos_teste": int((yte == 0).sum()),
                        "eventos_reais_teste": int(eventos_reais),
                        "eventos_detectados": int(eventos_detectados),
                        "diferenca_eventos": int(diferenca_eventos),
                        "accuracy": float(accuracy_score(yte, ypred)),
                        "balanced_accuracy": float(balanced_accuracy_score(yte, ypred)),
                        "precision_pos": float(precision_score(yte, ypred, pos_label=1, zero_division=0)),
                        "recall_pos": float(recall_score(yte, ypred, pos_label=1, zero_division=0)),
                        "f1_pos": float(f1_score(yte, ypred, pos_label=1, zero_division=0)),
                        "specificity": float(specificity),
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    }
                    resultados.append(row)

                    # Para teste com erro, prioriza F1/Recall.
                    # Para teste sem erro, prioriza poucos falsos positivos e eventos_detectados=0.
                    if int(yte.sum()) == 0:
                        score = row["accuracy"] - (0.05 * eventos_detectados) - (0.01 * fp)
                    else:
                        score = 0.7 * row["f1_pos"] + 0.3 * row["recall_pos"] - (0.03 * abs(diferenca_eventos))

                    if score > melhor_score:
                        melhor_score = score
                        melhor_row = row.copy()

                        path_cm = os.path.join(
                            pasta_out,
                            f"{nome_base}_{nome_teste}_{nome_modelo}_{falha}_jan{janela}_passo{passo}_cm.png"
                        )
                        _plot_confusion_binaria(
                            cm,
                            f"{nome_teste} | {nome_modelo} | {falha} | janela={janela} | passo={passo}",
                            path_cm
                        )
                        melhor_row["path_cm"] = path_cm

                        path_pred = os.path.join(
                            pasta_out,
                            f"{nome_base}_{nome_teste}_{nome_modelo}_{falha}_jan{janela}_passo{passo}_eventos.png"
                        )
                        _plot_predicoes_temporais(ds_teste, yte, ypred, falha, path_pred)
                        melhor_row["path_eventos"] = path_pred
                        paths_temporais[f"{nome_teste}_{nome_modelo}_{falha}"] = path_pred

            if melhor_row is not None:
                melhores_rows.append(melhor_row)

    df_resultados = pd.DataFrame(resultados)
    df_melhores = pd.DataFrame(melhores_rows)

    path_resultados = os.path.join(pasta_out, f"{nome_base}_{nome_teste}_resultados_todos.csv")
    path_melhores = os.path.join(pasta_out, f"{nome_base}_{nome_teste}_melhores_configuracoes.csv")
    path_json = os.path.join(pasta_out, f"{nome_base}_{nome_teste}_melhores_configuracoes.json")

    if not df_resultados.empty:
        df_resultados = df_resultados.sort_values(
            ["teste", "modelo", "falha", "janela", "passo_real"],
            ascending=[True, True, True, True, True]
        ).reset_index(drop=True)
        df_resultados.to_csv(path_resultados, index=False)

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
                        "passo_real", "eventos_reais_teste", "eventos_detectados", "diferenca_eventos",
                        "tp", "fp", "tn", "fn",
                        "accuracy", "precision_pos", "recall_pos", "f1_pos", "balanced_accuracy"
                    ]].copy()

                    tabela = tabela.rename(columns={
                        "passo_real": "Deslocamento",
                        "eventos_reais_teste": "Eventos_reais",
                        "eventos_detectados": "Eventos_detectados",
                        "diferenca_eventos": "Diferenca_eventos",
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

                    nome_csv = f"{nome_base}_{nome_teste}_{falha}_{nome_modelo}_janela_{janela}.csv"
                    tabela.to_csv(os.path.join(pasta_tabelas, nome_csv), index=False)

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
                    f"{nome_teste}_{falha}_{nome_modelo}",
                    pasta_out,
                    nome_base
                )
                if path_cmp:
                    paths_comparacao[f"{nome_teste}_{falha}_{nome_modelo}"] = path_cmp

    if not df_melhores.empty:
        df_melhores = df_melhores.sort_values(["teste", "modelo", "falha"]).reset_index(drop=True)
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
            "predicoes_temporais": paths_temporais,
            "pasta_tabelas_metricas": os.path.join(pasta_out, "tabelas_metricas")
        }
    }


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
    """
    Compatibilidade com o código antigo.
    Agora usa a própria base como treino e teste.
    """
    return avaliar_modelos_treino_teste_eventos(
        df_treino=df_base,
        df_teste=df_base,
        coluna=coluna,
        janelas=janelas,
        passos_cfg=passos_cfg,
        modelos=modelos,
        pasta_out=pasta_out,
        nome_base=nome_base,
        rotulo_modo=rotulo_modo,
        nome_teste="validacao_mesma_base"
    )


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
