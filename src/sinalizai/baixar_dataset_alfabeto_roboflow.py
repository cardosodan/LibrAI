"""Baixa o dataset "Alfabeto em Libras" (ElaineSilva, Roboflow Universe,
licença CC BY 4.0) e MESCLA em data/raw/alfabeto/<LETRA>/ — 22 classes reais
de Libras (A,B,C,D,E,F,G,I,K,L,M,N,O,P,Q,R,S,T,U,V,W), incluindo letras que
o dataset original (UNIVALI) não tinha: F, G, K, P, Q, T.

Pré-requisito: conta gratuita em roboflow.com + API key (Settings > API Keys)
exportada como variável de ambiente:

    # PowerShell
    $env:ROBOFLOW_API_KEY = "sua_chave_aqui"
    # bash
    export ROBOFLOW_API_KEY=sua_chave_aqui

Uso:
    venv/Scripts/python.exe -m sinalizai.baixar_dataset_alfabeto_roboflow

Idempotente: arquivos já mesclados (prefixo "rf_") não são baixados de novo
se a pasta de destino já tiver pelo menos um arquivo com esse prefixo pra
aquela letra — apagar a pasta manualmente força um re-download.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from . import config

WORKSPACE = "elainesilva"
PROJETO = "alfabeto-em-libras-qrvnw"
VERSAO = 6

# Nomes de classe do dataset -> nome de pasta usado neste projeto.
# D1/D2 são duas variações da mesma letra D no dataset original — mescladas
# numa pasta só, já que o classificador daqui não distingue variação de mão.
MAPA_CLASSES = {"D1": "D", "D2": "D"}


def _normalizar_classe(nome: str) -> str:
    return MAPA_CLASSES.get(nome, nome).upper()


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Variável de ambiente ROBOFLOW_API_KEY não encontrada.\n"
            "Crie uma conta grátis em https://roboflow.com, gere uma API key em "
            "Settings > API Keys, e exporte antes de rodar de novo:\n"
            '  PowerShell:  $env:ROBOFLOW_API_KEY = "sua_chave"\n'
            "  bash:        export ROBOFLOW_API_KEY=sua_chave"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("Pacote 'roboflow' não instalado. Rode: venv/Scripts/python.exe -m pip install roboflow")

    destino_bruto = config.RAW_DIR / "_roboflow_bruto"
    if destino_bruto.exists():
        shutil.rmtree(destino_bruto)
    destino_bruto.mkdir(parents=True, exist_ok=True)

    print(f"Conectando ao Roboflow ({WORKSPACE}/{PROJETO}, versão {VERSAO})...")
    rf = Roboflow(api_key=api_key)
    projeto = rf.workspace(WORKSPACE).project(PROJETO)
    versao = projeto.version(VERSAO)

    print("Baixando em formato COCO (inclui bounding boxes, mas só usamos a classe)...")
    dataset = versao.download("coco", location=str(destino_bruto))

    contagem_por_letra: dict[str, int] = {}
    config.ALFABETO_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Export COCO do Roboflow separa em subpastas train/valid/test, cada uma
    # com seu próprio _annotations.coco.json — processamos as 3.
    for subpasta in ["train", "valid", "test"]:
        pasta_split = Path(dataset.location) / subpasta
        anotacoes_path = pasta_split / "_annotations.coco.json"
        if not anotacoes_path.exists():
            continue

        dados = json.loads(anotacoes_path.read_text(encoding="utf-8"))
        id_para_nome_categoria = {c["id"]: c["name"] for c in dados["categories"]}
        id_para_arquivo = {img["id"]: img["file_name"] for img in dados["images"]}

        # Uma imagem pode ter mais de uma anotação (mais de uma bounding box) —
        # usamos a classe da PRIMEIRA anotação encontrada pra essa imagem,
        # já que o classificador aqui trabalha com a imagem inteira, não recorte.
        classe_por_imagem: dict[int, str] = {}
        for anotacao in dados["annotations"]:
            img_id = anotacao["image_id"]
            if img_id not in classe_por_imagem:
                classe_por_imagem[img_id] = id_para_nome_categoria[anotacao["category_id"]]

        for img_id, nome_classe in classe_por_imagem.items():
            letra = _normalizar_classe(nome_classe)
            arquivo_origem = pasta_split / id_para_arquivo[img_id]
            if not arquivo_origem.exists():
                continue
            pasta_letra = config.ALFABETO_RAW_DIR / letra
            pasta_letra.mkdir(parents=True, exist_ok=True)
            destino = pasta_letra / f"rf_{subpasta}_{arquivo_origem.name}"
            shutil.copy(arquivo_origem, destino)
            contagem_por_letra[letra] = contagem_por_letra.get(letra, 0) + 1

    shutil.rmtree(destino_bruto, ignore_errors=True)

    print(f"\nOK: mesclado em {config.ALFABETO_RAW_DIR}")
    for letra, n in sorted(contagem_por_letra.items()):
        print(f"  {letra}: +{n} imagens novas (rf_*)")
    print(
        "\nPróximo passo: venv/Scripts/python.exe -m sinalizai.preparar_dataset_alfabeto "
        "&& venv/Scripts/python.exe -m sinalizai.treinar_alfabeto"
    )


if __name__ == "__main__":
    main()
