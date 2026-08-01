"""Converte vídeos brutos de palavras (data/raw/palavras/<PALAVRA>/*.mp4) em
sequências de landmarks normalizados, reamostradas pra um tamanho fixo, e
salva tudo num único .npz (data/landmarks/palavras_landmarks.npz).

Uso:
    venv/Scripts/python.exe -m sinalizai.preparar_dataset_palavras
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from . import config
from .landmarks import HolisticSequenceExtractor

EXTENSOES_VIDEO = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def main() -> None:
    if not config.PALAVRAS_RAW_DIR.exists():
        raise SystemExit(
            f"Não achei {config.PALAVRAS_RAW_DIR}. Grave amostras com:\n"
            "  venv/Scripts/python.exe -m sinalizai.gravar_amostras_palavras --palavra <nome>\n"
            "(ou organize vídeos baixados do V-LIBRASIL nessa estrutura — ver README)"
        )

    palavras = sorted(p.name for p in config.PALAVRAS_RAW_DIR.iterdir() if p.is_dir())
    if len(palavras) < 2:
        raise SystemExit(
            f"Preciso de pelo menos 2 palavras diferentes (pastas) em {config.PALAVRAS_RAW_DIR}, "
            f"achei {len(palavras)}."
        )

    extrator = HolisticSequenceExtractor()
    sequencias, rotulos = [], []
    for palavra in palavras:
        pasta = config.PALAVRAS_RAW_DIR / palavra
        videos = sorted({v for padrao in EXTENSOES_VIDEO for v in pasta.glob(padrao)})
        for caminho in tqdm(videos, desc=f"Palavra {palavra}"):
            seq = extrator.extrair_de_video(str(caminho))
            if seq is None:
                continue
            sequencias.append(seq)
            rotulos.append(palavra)
    extrator.fechar()

    if not sequencias:
        raise SystemExit("Nenhum vídeo processado com sucesso — confira se os arquivos abrem no OpenCV.")

    config.LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.PALAVRAS_LANDMARKS_NPZ,
        X=np.array(sequencias, dtype=np.float32),
        y=np.array(rotulos),
    )
    print(f"\nOK: {len(sequencias)} amostras salvas em {config.PALAVRAS_LANDMARKS_NPZ}")
    for palavra in palavras:
        print(f"  {palavra}: {rotulos.count(palavra)} amostras")


if __name__ == "__main__":
    main()
