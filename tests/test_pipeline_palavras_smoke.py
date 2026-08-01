"""Teste de fumaça do pipeline de palavras dinâmicas: gera vídeos sintéticos
(fotos reais do alfabeto reaproveitadas com leve deslocamento entre frames,
simulando movimento) pra garantir que extração -> preparo -> treino rodam
ponta a ponta sem erro de forma/tipo, sem depender de uma webcam real nem
do dataset de palavras (que exige gravação/download separado — ver README).

Não valida ACURÁCIA nenhuma (não é sinal de Libras de verdade) — só que o
pipeline mecanicamente funciona antes de rodar em dados reais.

Uso:
    venv/Scripts/python.exe tests/test_pipeline_palavras_smoke.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from sinalizai import config
from sinalizai.landmarks import HolisticSequenceExtractor

PASTA_TESTE = config.BASE_DIR / "data" / "raw" / "_smoke_test_palavras"


def gerar_video_sintetico(caminho_imagem_base: Path, caminho_saida: Path, n_frames: int = 20):
    frame_base = cv2.imread(str(caminho_imagem_base))
    altura, largura = frame_base.shape[:2]
    writer = cv2.VideoWriter(str(caminho_saida), cv2.VideoWriter_fourcc(*"mp4v"), 15, (largura, altura))
    for i in range(n_frames):
        deslocamento = int(5 * np.sin(i / 3))
        matriz = np.float32([[1, 0, deslocamento], [0, 1, 0]])
        frame = cv2.warpAffine(frame_base, matriz, (largura, altura), borderMode=cv2.BORDER_REPLICATE)
        writer.write(frame)
    writer.release()


def main():
    if PASTA_TESTE.exists():
        shutil.rmtree(PASTA_TESTE)

    fonte_a = next((config.ALFABETO_RAW_DIR / "A").glob("*.jpg"))
    fonte_b = next((config.ALFABETO_RAW_DIR / "B").glob("*.jpg"))

    for nome, fonte in [("classe_a", fonte_a), ("classe_b", fonte_b)]:
        pasta = PASTA_TESTE / nome
        pasta.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            gerar_video_sintetico(fonte, pasta / f"{nome}_{i}.mp4")

    print(f"Vídeos sintéticos gerados em {PASTA_TESTE}")

    extrator = HolisticSequenceExtractor()
    sequencias, rotulos = [], []
    deteccoes_mao = 0
    for pasta_classe in sorted(PASTA_TESTE.iterdir()):
        for video in sorted(pasta_classe.glob("*.mp4")):
            seq, pista_facial = extrator.extrair_de_video(str(video))
            assert seq is not None, f"Falha ao extrair {video}"
            assert seq.shape == (config.SEQUENCE_LENGTH, config.DIM_FEATURES_FRAME), seq.shape
            assert pista_facial is None or isinstance(pista_facial, str)
            if np.any(seq != 0):
                deteccoes_mao += 1
            sequencias.append(seq)
            rotulos.append(pasta_classe.name)
    extrator.fechar()

    print(f"OK: {len(sequencias)} sequências extraídas, shape {sequencias[0].shape}")
    print(f"Sequências com pelo menos 1 landmark não-zero detectado: {deteccoes_mao}/{len(sequencias)}")

    shutil.rmtree(PASTA_TESTE)
    print("\nSMOKE TEST OK — pipeline de palavras roda ponta a ponta sem erro de forma/tipo.")


if __name__ == "__main__":
    main()
