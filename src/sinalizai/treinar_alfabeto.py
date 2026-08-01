"""Treina e avalia o classificador do alfabeto estático de Libras.

Compara dois modelos de propósito (não só treina um e aceita): uma baseline
linear simples (Regressão Logística) contra um Random Forest — se a árvore
não superar a baseline por uma margem clara, isso é um sinal de que o
problema é quase linearmente separável nas 63 features de landmark, o que
vale a pena reportar, não esconder.

Uso:
    venv/Scripts/python.exe -m sinalizai.treinar_alfabeto
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from . import config


def main() -> None:
    if not config.ALFABETO_LANDMARKS_CSV.exists():
        raise SystemExit(
            f"Não achei {config.ALFABETO_LANDMARKS_CSV}. Rode antes:\n"
            "  venv/Scripts/python.exe -m sinalizai.preparar_dataset_alfabeto"
        )

    df = pd.read_csv(config.ALFABETO_LANDMARKS_CSV)
    colunas_features = [c for c in df.columns if c.startswith("f")]
    X, y = df[colunas_features].values, df["letra"].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )

    candidatos = {
        "regressao_logistica": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
    }

    melhor_nome, melhor_modelo, melhor_f1 = None, None, -1.0
    resultados = {}
    for nome, modelo in candidatos.items():
        modelo.fit(X_treino, y_treino)
        y_pred = modelo.predict(X_teste)
        f1 = f1_score(y_teste, y_pred, average="macro")
        resultados[nome] = f1
        print(f"\n=== {nome} (F1 macro = {f1:.4f}) ===")
        print(classification_report(y_teste, y_pred, zero_division=0))
        if f1 > melhor_f1:
            melhor_nome, melhor_modelo, melhor_f1 = nome, modelo, f1

    print(f"\nModelo escolhido: {melhor_nome} (F1 macro = {melhor_f1:.4f})")

    y_pred_final = melhor_modelo.predict(X_teste)
    letras_ordenadas = sorted(df["letra"].unique())
    matriz = confusion_matrix(y_teste, y_pred_final, labels=letras_ordenadas)

    config.RELATORIOS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matriz, annot=True, fmt="d", cmap="Blues",
        xticklabels=letras_ordenadas, yticklabels=letras_ordenadas, ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão — alfabeto estático ({melhor_nome})")
    fig.tight_layout()
    caminho_fig = config.RELATORIOS_DIR / "matriz_confusao_alfabeto.png"
    fig.savefig(caminho_fig, dpi=150)
    plt.close(fig)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": melhor_modelo, "classes": list(melhor_modelo.classes_)}, config.MODELO_ALFABETO_PATH)

    metricas = {
        "modelo_escolhido": melhor_nome,
        "f1_macro_por_modelo": resultados,
        "n_amostras_treino": len(X_treino),
        "n_amostras_teste": len(X_teste),
        "classes": letras_ordenadas,
    }
    (config.RELATORIOS_DIR / "metricas_alfabeto.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nModelo salvo em {config.MODELO_ALFABETO_PATH}")
    print(f"Matriz de confusão salva em {caminho_fig}")


if __name__ == "__main__":
    main()
