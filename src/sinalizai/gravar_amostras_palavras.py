"""Ferramenta de gravação: grava amostras de vídeo curtas pra treinar o
classificador de palavras dinâmicas, direto da webcam.

Esse é o caminho PRINCIPAL pra ter dados de palavras (ver README — o dataset
público V-LIBRASIL existe mas pesa ~11GB, exige conta Kaggle e tem licença
não-comercial CC BY-NC-ND, então gravar um vocabulário próprio pequeno é o
caminho mais direto pra um MVP funcionando de verdade).

Uso:
    venv/Scripts/python.exe -m sinalizai.gravar_amostras_palavras --palavra ola --amostras 15

Durante a gravação:
    ESPAÇO  -> começa a gravar uma amostra de N segundos
    Q       -> encerra o programa
"""
from __future__ import annotations

import argparse
import time

import cv2

from . import config

DURACAO_AMOSTRA_SEGUNDOS = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--palavra", required=True, help="Nome da palavra/sinal (usado como nome da pasta e rótulo)")
    parser.add_argument("--amostras", type=int, default=15, help="Quantas amostras gravar nesta sessão")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam (0 = padrão)")
    args = parser.parse_args()

    pasta = config.PALAVRAS_RAW_DIR / args.palavra
    pasta.mkdir(parents=True, exist_ok=True)
    ja_existentes = len(list(pasta.glob("*.mp4")))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {args.camera}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Gravando amostras de '{args.palavra}' em {pasta}")
    print(f"Já existem {ja_existentes} amostras dessa palavra.")
    print("ESPAÇO = gravar uma amostra | Q = sair\n")

    gravadas = 0
    while gravadas < args.amostras:
        ok, frame = cap.read()
        if not ok:
            break

        texto = f"'{args.palavra}': {gravadas}/{args.amostras} — ESPAÇO grava, Q sai"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gravar amostra de palavra", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

        if tecla == ord(" "):
            indice = ja_existentes + gravadas + 1
            caminho_saida = pasta / f"{args.palavra}_{indice:03d}.mp4"
            writer = cv2.VideoWriter(
                str(caminho_saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
            )
            inicio = time.time()
            print(f"Gravando amostra {indice}...")
            while time.time() - inicio < DURACAO_AMOSTRA_SEGUNDOS:
                ok, frame = cap.read()
                if not ok:
                    break
                restante = DURACAO_AMOSTRA_SEGUNDOS - (time.time() - inicio)
                cv2.putText(frame, f"GRAVANDO ({restante:.1f}s)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                writer.write(frame)
                cv2.imshow("Gravar amostra de palavra", frame)
                cv2.waitKey(1)
            writer.release()
            gravadas += 1
            print(f"Salvo: {caminho_saida}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{gravadas} amostras novas gravadas. Total de '{args.palavra}': {ja_existentes + gravadas}")


if __name__ == "__main__":
    main()
