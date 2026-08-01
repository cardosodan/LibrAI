"""Baixa o Brazilian Sign Language Alphabet Dataset (Passos, Fernandes & Comunello,
UNIVALI) direto do GitHub público e extrai em data/raw/alfabeto/<LETRA>/.

Dataset: 4.411 imagens, 15 letras estáticas do alfabeto manual de Libras
(A,B,C,D,E,I,L,M,N,O,R,S,U,V,W). Licença MIT — cópia salva em
docs/creditos/ pra referência.

Uso:
    venv/Scripts/python.exe -m sinalizai.baixar_dataset_alfabeto
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests

from . import config

URL_ZIP = "https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset/archive/refs/heads/master.zip"
NOME_PASTA_REPO = "Brazilian-Sign-Language-Alphabet-Dataset-master"


def main() -> None:
    if config.ALFABETO_RAW_DIR.exists() and any(config.ALFABETO_RAW_DIR.iterdir()):
        print(f"{config.ALFABETO_RAW_DIR} já existe e não está vazio — pulando download.")
        return

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.RAW_DIR / "alfabeto_repo.zip"

    print(f"Baixando {URL_ZIP} ...")
    resposta = requests.get(URL_ZIP, timeout=60)
    resposta.raise_for_status()
    zip_path.write_bytes(resposta.content)

    print("Extraindo...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(config.RAW_DIR)

    pasta_extraida = config.RAW_DIR / NOME_PASTA_REPO
    shutil.move(str(pasta_extraida / "dataset"), str(config.ALFABETO_RAW_DIR))

    docs_creditos = config.BASE_DIR / "docs" / "creditos"
    docs_creditos.mkdir(parents=True, exist_ok=True)
    shutil.copy(pasta_extraida / "LICENSE", docs_creditos / "LICENSE_dataset_alfabeto.txt")
    shutil.copy(pasta_extraida / "README.md", docs_creditos / "README_dataset_alfabeto_original.md")

    shutil.rmtree(pasta_extraida)
    zip_path.unlink()

    n_imagens = sum(1 for _ in config.ALFABETO_RAW_DIR.rglob("*.jpg"))
    print(f"OK: {n_imagens} imagens em {config.ALFABETO_RAW_DIR}")


if __name__ == "__main__":
    main()
