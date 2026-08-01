"""Treina o classificador LSTM de palavras dinâmicas em cima do .npz gerado
por preparar_dataset_palavras.py.

Datasets de palavra tendem a ser pequenos (poucas dezenas de amostras por
classe, ver README) — por isso o treino inclui uma augmentação simples
(jitter gaussiano nas coordenadas normalizadas) multiplicando cada amostra
de treino algumas vezes, em vez de confiar em early-stopping sozinho pra
não sobreajustar num dataset minúsculo.

Uso:
    venv/Scripts/python.exe -m sinalizai.treinar_palavras
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from . import config
from .modelo_dinamico import ClassificadorPalavrasLSTM

EPOCAS = 60
TAXA_APRENDIZADO = 1e-3
FATOR_AUGMENTACAO = 8  # cada amostra de treino vira N versões com ruído leve
DESVIO_JITTER = 0.02


class DatasetSequencias(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, aumentar: bool = False):
        self.X, self.y, self.aumentar = X, y, aumentar
        self.fator = FATOR_AUGMENTACAO if aumentar else 1

    def __len__(self) -> int:
        return len(self.X) * self.fator

    def __getitem__(self, idx: int):
        base = idx % len(self.X)
        seq = self.X[base].copy()
        if self.aumentar and idx >= len(self.X):
            seq = seq + np.random.normal(0, DESVIO_JITTER, seq.shape).astype(np.float32)
        return torch.from_numpy(seq), self.y[base]


def main() -> None:
    if not config.PALAVRAS_LANDMARKS_NPZ.exists():
        raise SystemExit(
            f"Não achei {config.PALAVRAS_LANDMARKS_NPZ}. Rode antes:\n"
            "  venv/Scripts/python.exe -m sinalizai.preparar_dataset_palavras"
        )

    dados = np.load(config.PALAVRAS_LANDMARKS_NPZ, allow_pickle=True)
    X, y_texto = dados["X"], dados["y"]

    codificador = LabelEncoder()
    y = codificador.fit_transform(y_texto)
    classes = list(codificador.classes_)

    contagens = np.bincount(y)
    pode_estratificar = contagens.min() >= 2
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_STATE,
        stratify=y if pode_estratificar else None,
    )

    ds_treino = DatasetSequencias(X_treino, y_treino, aumentar=True)
    ds_teste = DatasetSequencias(X_teste, y_teste, aumentar=False)
    dl_treino = DataLoader(ds_treino, batch_size=16, shuffle=True)
    dl_teste = DataLoader(ds_teste, batch_size=16, shuffle=False)

    modelo = ClassificadorPalavrasLSTM(dim_entrada=config.DIM_FEATURES_FRAME, num_classes=len(classes))
    otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)
    perda_fn = nn.CrossEntropyLoss()

    melhor_f1, melhor_estado = -1.0, None
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        perda_total = 0.0
        for xb, yb in dl_treino:
            otimizador.zero_grad()
            saida = modelo(xb)
            perda = perda_fn(saida, yb)
            perda.backward()
            otimizador.step()
            perda_total += perda.item() * xb.size(0)

        modelo.eval()
        preds, reais = [], []
        with torch.no_grad():
            for xb, yb in dl_teste:
                saida = modelo(xb)
                preds.extend(saida.argmax(dim=1).tolist())
                reais.extend(yb.tolist())
        f1 = f1_score(reais, preds, average="macro", zero_division=0)
        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}

        if epoca % 10 == 0 or epoca == EPOCAS:
            print(f"Época {epoca:3d} — perda treino: {perda_total / len(ds_treino):.4f} — F1 macro teste: {f1:.4f}")

    modelo.load_state_dict(melhor_estado)
    modelo.eval()
    preds, reais = [], []
    with torch.no_grad():
        for xb, yb in dl_teste:
            saida = modelo(xb)
            preds.extend(saida.argmax(dim=1).tolist())
            reais.extend(yb.tolist())

    matriz = confusion_matrix(reais, preds, labels=list(range(len(classes))))
    config.RELATORIOS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Purples", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão — palavras dinâmicas (LSTM)")
    fig.tight_layout()
    fig.savefig(config.RELATORIOS_DIR / "matriz_confusao_palavras.png", dpi=150)
    plt.close(fig)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), config.MODELO_PALAVRAS_PATH)
    config.PALAVRAS_LABELS_PATH.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")

    print(f"\nMelhor F1 macro (teste): {melhor_f1:.4f}")
    print(f"Modelo salvo em {config.MODELO_PALAVRAS_PATH}")
    print(f"Classes salvas em {config.PALAVRAS_LABELS_PATH}")


if __name__ == "__main__":
    main()
