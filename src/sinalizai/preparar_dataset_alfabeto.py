"""Converte as imagens brutas do alfabeto (data/raw/alfabeto/<LETRA>/*.jpg) em um
CSV de landmarks normalizados (data/landmarks/alfabeto_landmarks.csv), um vetor
de 63 features por imagem + a letra correspondente.

Uso:
    venv/Scripts/python.exe -m sinalizai.preparar_dataset_alfabeto
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from . import config
from .landmarks import HandLandmarkExtractor


def main() -> None:
    if not config.ALFABETO_RAW_DIR.exists():
        raise SystemExit(
            f"Não achei {config.ALFABETO_RAW_DIR}. Rode antes:\n"
            "  venv/Scripts/python.exe -m sinalizai.baixar_dataset_alfabeto"
        )

    extrator = HandLandmarkExtractor()
    linhas = []
    sem_deteccao = 0
    total = 0

    letras = sorted(p.name for p in config.ALFABETO_RAW_DIR.iterdir() if p.is_dir())
    for letra in letras:
        pasta = config.ALFABETO_RAW_DIR / letra
        imagens = sorted(pasta.glob("*.jpg"))
        for caminho in tqdm(imagens, desc=f"Letra {letra}"):
            total += 1
            vetor = extrator.extrair_de_arquivo(str(caminho))
            if vetor is None:
                sem_deteccao += 1
                continue
            linha = {f"f{i}": v for i, v in enumerate(vetor)}
            linha["letra"] = letra
            linha["arquivo"] = caminho.name
            linhas.append(linha)

    extrator.fechar()

    config.LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(linhas)
    df.to_csv(config.ALFABETO_LANDMARKS_CSV, index=False)

    taxa = 100 * (total - sem_deteccao) / total if total else 0
    print(f"\nOK: {len(df)} amostras salvas em {config.ALFABETO_LANDMARKS_CSV}")
    print(f"Detecção de mão: {total - sem_deteccao}/{total} imagens ({taxa:.1f}%)")
    print(f"Imagens sem mão detectada (ficaram de fora): {sem_deteccao}")
    print(df["letra"].value_counts().sort_index())


if __name__ == "__main__":
    main()
