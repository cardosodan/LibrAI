"""Demo em tempo real pela webcam: reconhece letras do alfabeto continuamente
e, sob pedido (tecla P), grava 2s de vídeo e classifica como uma palavra.

Uso:
    venv/Scripts/python.exe -m sinalizai.demo_webcam

Teclas:
    P       -> grava 2s e classifica como palavra (exige classificador_palavras.pt treinado)
    Q       -> sai
"""
from __future__ import annotations

import json
import time

import cv2
import joblib
import numpy as np
import torch

from . import config
from .landmarks import HandLandmarkExtractor, HolisticSequenceExtractor
from .modelo_dinamico import ClassificadorPalavrasLSTM

DURACAO_CAPTURA_PALAVRA_S = 2.0
LIMIAR_CONFIANCA_LETRA = 0.55


def carregar_modelo_alfabeto():
    if not config.MODELO_ALFABETO_PATH.exists():
        print(f"[aviso] {config.MODELO_ALFABETO_PATH} não existe — modo alfabeto desativado.")
        print("        Rode: venv/Scripts/python.exe -m sinalizai.treinar_alfabeto")
        return None
    return joblib.load(config.MODELO_ALFABETO_PATH)


def carregar_modelo_palavras():
    if not (config.MODELO_PALAVRAS_PATH.exists() and config.PALAVRAS_LABELS_PATH.exists()):
        print(f"[aviso] Modelo de palavras não encontrado — modo palavra desativado.")
        print("        Rode: venv/Scripts/python.exe -m sinalizai.gravar_amostras_palavras --palavra X")
        print("              venv/Scripts/python.exe -m sinalizai.preparar_dataset_palavras")
        print("              venv/Scripts/python.exe -m sinalizai.treinar_palavras")
        return None, None
    classes = json.loads(config.PALAVRAS_LABELS_PATH.read_text(encoding="utf-8"))
    modelo = ClassificadorPalavrasLSTM(dim_entrada=config.DIM_FEATURES_FRAME, num_classes=len(classes))
    modelo.load_state_dict(torch.load(config.MODELO_PALAVRAS_PATH, map_location="cpu"))
    modelo.eval()
    return modelo, classes


def main() -> None:
    pacote_alfabeto = carregar_modelo_alfabeto()
    modelo_palavras, classes_palavras = carregar_modelo_palavras()

    extrator_mao = HandLandmarkExtractor() if pacote_alfabeto else None
    extrator_holistic = HolisticSequenceExtractor() if modelo_palavras else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Não consegui abrir a webcam (índice 0).")

    mensagem_palavra, mensagem_ate = "", 0.0

    print("SinalizAI — demo ao vivo. P = classificar palavra (2s) | Q = sair")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        if extrator_mao is not None:
            vetor = extrator_mao.extrair_de_bgr(frame)
            if vetor is not None:
                modelo = pacote_alfabeto["modelo"]
                probs = modelo.predict_proba([vetor])[0]
                indice = int(np.argmax(probs))
                letra, confianca = modelo.classes_[indice], probs[indice]
                if confianca >= LIMIAR_CONFIANCA_LETRA:
                    cv2.putText(frame, f"Letra: {letra} ({confianca:.0%})", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if time.time() < mensagem_ate:
            cv2.putText(frame, mensagem_palavra, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 150, 0), 2)

        cv2.putText(frame, "P = palavra | Q = sair", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("SinalizAI", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

        if tecla == ord("p") and extrator_holistic is not None:
            frames_capturados = []
            inicio = time.time()
            while time.time() - inicio < DURACAO_CAPTURA_PALAVRA_S:
                ok, f = cap.read()
                if not ok:
                    break
                f = cv2.flip(f, 1)
                restante = DURACAO_CAPTURA_PALAVRA_S - (time.time() - inicio)
                cv2.putText(f, f"GRAVANDO PALAVRA ({restante:.1f}s)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("SinalizAI", f)
                cv2.waitKey(1)
                frames_capturados.append(f)

            seq = extrator_holistic.extrair_de_frames(frames_capturados)
            if seq is not None:
                with torch.no_grad():
                    entrada = torch.from_numpy(seq).unsqueeze(0)
                    saida = modelo_palavras(entrada)
                    probs = torch.softmax(saida, dim=1)[0]
                    indice = int(probs.argmax())
                    mensagem_palavra = f"Palavra: {classes_palavras[indice]} ({probs[indice]:.0%})"
            else:
                mensagem_palavra = "Não detectei nada — tente de novo mais perto da câmera."
            mensagem_ate = time.time() + 4.0

    cap.release()
    cv2.destroyAllWindows()
    if extrator_mao:
        extrator_mao.fechar()
    if extrator_holistic:
        extrator_holistic.fechar()


if __name__ == "__main__":
    main()
