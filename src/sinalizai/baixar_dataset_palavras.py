"""Baixa (melhor esforço) um subconjunto do V-LIBRASIL via API do Kaggle.

O V-LIBRASIL (Rodrigues, UFPE — https://libras.cin.ufpe.br/) tem 1.364
palavras/termos, 4.089 vídeos, 3 sinalizantes profissionais, ~11GB — grande
demais e sob licença CC BY-NC-ND (não-comercial, sem obras derivadas) pra
baixar/redistribuir automaticamente aqui. Este script só automatiza o
download bruto; a curadoria de QUAIS palavras usar e a organização em pastas
por palavra (`data/raw/palavras/<palavra>/*.mp4`) é manual, de propósito —
não tenho como inspecionar a estrutura interna do dataset sem uma conta
Kaggle autenticada.

Pré-requisitos:
    1. Conta no Kaggle + aceitar os termos do dataset em
       https://www.kaggle.com/datasets/davimedio01/v-librasil
    2. Token de API em ~/.kaggle/kaggle.json (Account > Create New API Token)
    3. pip install kaggle

Uso:
    venv/Scripts/python.exe -m sinalizai.baixar_dataset_palavras

IMPORTANTE — licença: CC BY-NC-ND 4.0 = uso não-comercial, sem redistribuir
versão modificada. Bom pra portfólio/pesquisa/aprendizado; NÃO usar como
dado de um produto comercial sem falar com o autor.
"""
from __future__ import annotations

import subprocess
import sys

from . import config

DESTINO = config.RAW_DIR / "palavras_v_librasil_bruto"


def main() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Pacote 'kaggle' não instalado. Rode:\n"
            "  venv/Scripts/python.exe -m pip install kaggle\n"
            "E configure ~/.kaggle/kaggle.json antes de tentar de novo."
        )

    DESTINO.mkdir(parents=True, exist_ok=True)
    print(f"Baixando V-LIBRASIL (~11GB) em {DESTINO} — isso pode demorar bastante...")
    resultado = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", "davimedio01/v-librasil", "-p", str(DESTINO), "--unzip"],
        check=False,
    )
    if resultado.returncode != 0:
        raise SystemExit(
            "Download falhou. Confirme que ~/.kaggle/kaggle.json existe e que você "
            "aceitou os termos do dataset na página do Kaggle antes de rodar de novo."
        )

    print(f"\nOK — arquivos brutos em {DESTINO}.")
    print(
        "PRÓXIMO PASSO (manual): inspecione a estrutura baixada (provavelmente um "
        "CSV com o glossário + pasta de vídeos por sinalizante) e organize os vídeos "
        f"das palavras que você quer treinar em:\n"
        f"  {config.PALAVRAS_RAW_DIR}/<palavra>/*.mp4\n"
        "Recomendo começar com 15-30 palavras de uso comum (cumprimentos, números, "
        "necessidades básicas) em vez das 1.364 inteiras — ver README."
    )


if __name__ == "__main__":
    main()
