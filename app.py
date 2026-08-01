"""SinalizAI — demo web do reconhecimento de Libras.

Front-end captura frames da webcam do NAVEGADOR (getUserMedia) e manda pro
back-end via POST (base64) — mesma arquitetura de sempre (Flask servindo API
+ HTML), sem streaming/WebSocket, pra manter simples de rodar/hospedar (ex:
Render, igual aos outros projetos).

Rotas:
    GET  /                        página da demo
    POST /api/reconhecer-letra    1 frame -> letra do alfabeto + confiança
    POST /api/reconhecer-palavra  N frames (~2s) -> palavra + confiança
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sinalizai import config
from sinalizai.landmarks import HandLandmarkExtractor, HolisticSequenceExtractor
from sinalizai.modelo_dinamico import ClassificadorPalavrasLSTM

app = Flask(__name__)

_pacote_alfabeto = None
_extrator_mao = None
_modelo_palavras = None
_classes_palavras = None
_extrator_holistic = None


def _carregar_modelos() -> None:
    global _pacote_alfabeto, _extrator_mao, _modelo_palavras, _classes_palavras, _extrator_holistic

    if config.MODELO_ALFABETO_PATH.exists():
        try:
            _pacote_alfabeto = joblib.load(config.MODELO_ALFABETO_PATH)
            _extrator_mao = HandLandmarkExtractor()
            print("[SinalizAI] Modelo do alfabeto carregado.")
        except FileNotFoundError as erro:
            _pacote_alfabeto = None
            print(f"[SinalizAI] Alfabeto desativado: {erro}")
    else:
        print("[SinalizAI] Modelo do alfabeto ausente — /api/reconhecer-letra vai retornar erro claro.")

    if config.MODELO_PALAVRAS_PATH.exists() and config.PALAVRAS_LABELS_PATH.exists():
        try:
            _classes_palavras = json.loads(config.PALAVRAS_LABELS_PATH.read_text(encoding="utf-8"))
            _modelo_palavras = ClassificadorPalavrasLSTM(
                dim_entrada=config.DIM_FEATURES_FRAME, num_classes=len(_classes_palavras)
            )
            _modelo_palavras.load_state_dict(torch.load(config.MODELO_PALAVRAS_PATH, map_location="cpu"))
            _modelo_palavras.eval()
            _extrator_holistic = HolisticSequenceExtractor()
            print("[SinalizAI] Modelo de palavras carregado.")
        except FileNotFoundError as erro:
            _modelo_palavras = None
            print(f"[SinalizAI] Palavras desativado: {erro}")
    else:
        print("[SinalizAI] Modelo de palavras ausente — /api/reconhecer-palavra vai retornar erro claro.")


def _base64_para_frame(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    dados = base64.b64decode(b64)
    array = np.frombuffer(dados, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template(
        "index.html",
        alfabeto_disponivel=_pacote_alfabeto is not None,
        palavras_disponivel=_modelo_palavras is not None,
        letras=config.LETRAS_ESTATICAS,
        palavras=_classes_palavras or [],
    )


@app.route("/api/reconhecer-letra", methods=["POST"])
def reconhecer_letra():
    if _pacote_alfabeto is None:
        return jsonify({"erro": "Modelo do alfabeto não foi treinado neste servidor."}), 503

    corpo = request.get_json(silent=True) or {}
    frame_b64 = corpo.get("frame")
    if not frame_b64:
        return jsonify({"erro": "Campo 'frame' (base64) é obrigatório."}), 400

    frame = _base64_para_frame(frame_b64)
    if frame is None:
        return jsonify({"erro": "Não consegui decodificar a imagem."}), 400

    vetor = _extrator_mao.extrair_de_bgr(frame)
    if vetor is None:
        return jsonify({"detectado": False})

    modelo = _pacote_alfabeto["modelo"]
    probs = modelo.predict_proba([vetor])[0]
    indice = int(np.argmax(probs))
    return jsonify({
        "detectado": True,
        "letra": modelo.classes_[indice],
        "confianca": float(probs[indice]),
    })


@app.route("/api/reconhecer-palavra", methods=["POST"])
def reconhecer_palavra():
    if _modelo_palavras is None:
        return jsonify({"erro": "Modelo de palavras não foi treinado neste servidor."}), 503

    corpo = request.get_json(silent=True) or {}
    frames_b64 = corpo.get("frames") or []
    if len(frames_b64) < 2:
        return jsonify({"erro": "Envie uma lista 'frames' com pelo menos 2 imagens base64."}), 400

    frames = [_base64_para_frame(f) for f in frames_b64]
    frames = [f for f in frames if f is not None]
    if not frames:
        return jsonify({"erro": "Não consegui decodificar nenhum frame."}), 400

    seq = _extrator_holistic.extrair_de_frames(frames, fps=len(frames) / 2.0)
    if seq is None:
        return jsonify({"detectado": False})

    with torch.no_grad():
        entrada = torch.from_numpy(seq).unsqueeze(0)
        probs = torch.softmax(_modelo_palavras(entrada), dim=1)[0]
        indice = int(probs.argmax())

    return jsonify({
        "detectado": True,
        "palavra": _classes_palavras[indice],
        "confianca": float(probs[indice]),
    })


_carregar_modelos()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5060, debug=False)
