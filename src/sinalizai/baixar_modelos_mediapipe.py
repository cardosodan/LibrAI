"""Baixa os bundles de modelo do MediaPipe Tasks (hand_landmarker.task,
holistic_landmarker.task) direto do repositório oficial do Google — não vão
pro Git (ver .gitignore) porque são artefatos binários grandes e
redistribuídos pelo próprio Google, sem motivo pra versionar cópia própria.

Uso:
    venv/Scripts/python.exe -m sinalizai.baixar_modelos_mediapipe
"""
from __future__ import annotations

import requests

from . import config

MODELOS = {
    config.HAND_LANDMARKER_TASK: (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    config.HOLISTIC_LANDMARKER_TASK: (
        "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
        "holistic_landmarker/float16/latest/holistic_landmarker.task"
    ),
}


def main() -> None:
    config.MEDIAPIPE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for destino, url in MODELOS.items():
        if destino.exists():
            print(f"Já existe: {destino}")
            continue
        print(f"Baixando {url} ...")
        resposta = requests.get(url, timeout=60)
        resposta.raise_for_status()
        destino.write_bytes(resposta.content)
        print(f"OK: {destino} ({destino.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
