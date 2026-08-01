"""SinalizAI — demo web do reconhecimento de Libras.

Front-end captura frames da webcam do NAVEGADOR (getUserMedia) e manda pro
back-end via POST (base64) — mesma arquitetura de sempre (Flask servindo API
+ HTML), sem streaming/WebSocket, pra manter simples de rodar/hospedar (ex:
Render, igual aos outros projetos).

Rotas:
    GET  /                        página da demo
    POST /api/reconhecer-letra    1 frame -> letra do alfabeto + confiança
    POST /api/reconhecer-palavra  N frames (~2s) -> palavra + confiança
    POST /api/traduzir            texto em português -> tradução (inglês, por enquanto)
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from deep_translator import GoogleTranslator
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sinalizai import config
from sinalizai.landmarks import HandLandmarkExtractor, HolisticSequenceExtractor
from sinalizai.modelo_dinamico import ClassificadorPalavrasLSTM

app = Flask(__name__)


def _versao_estatico(caminho_relativo: str) -> int:
    """Retorna a data de modificação do arquivo estático (epoch, int) pra usar
    como query string de cache-busting (?v=...) nos links de CSS/JS — sem
    isso, o navegador às vezes reaproveita uma versão em cache mesmo depois
    de um F5 normal, e cada mudança de estilo exigiria explicar "dá um
    hard-refresh" de novo."""
    caminho = Path(app.static_folder) / caminho_relativo
    return int(caminho.stat().st_mtime) if caminho.exists() else 0


app.jinja_env.globals["versao_estatico"] = _versao_estatico

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

    vetor, pontos = _extrator_mao.extrair_com_pontos_de_bgr(frame)
    if vetor is None:
        return jsonify({"detectado": False})

    modelo = _pacote_alfabeto["modelo"]
    probs = modelo.predict_proba([vetor])[0]
    ordem = np.argsort(probs)[::-1]
    indice = int(ordem[0])
    candidatos = [
        {"letra": modelo.classes_[i], "confianca": float(probs[i])}
        for i in ordem[:3]
    ]
    return jsonify({
        "detectado": True,
        "letra": modelo.classes_[indice],
        "confianca": float(probs[indice]),
        "candidatos": candidatos,
        "landmarks": pontos,
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

    seq, pista_facial = _extrator_holistic.extrair_de_frames(frames, fps=len(frames) / 2.0)
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
        "pista_facial": pista_facial,
    })


# Idiomas de destino suportados por enquanto — só inglês, a pedido do usuário
# ("pode fazer apenas português e inglês por enquanto"). Adicionar francês/
# espanhol/etc no futuro é só acrescentar uma entrada aqui, o resto do
# pipeline (GoogleTranslator via deep-translator) já suporta qualquer par.
IDIOMAS_SUPORTADOS = {"en": "English"}

_cache_traducao: dict[tuple[str, str], str] = {}


@app.route("/api/traduzir", methods=["POST"])
def traduzir():
    corpo = request.get_json(silent=True) or {}
    texto = (corpo.get("texto") or "").strip()
    idioma = (corpo.get("idioma") or "en").strip().lower()

    if not texto:
        return jsonify({"erro": "Campo 'texto' é obrigatório."}), 400
    if idioma not in IDIOMAS_SUPORTADOS:
        return jsonify({"erro": f"Idioma '{idioma}' não suportado ainda. Disponíveis: {list(IDIOMAS_SUPORTADOS)}"}), 400

    chave = (texto, idioma)
    if chave in _cache_traducao:
        return jsonify({"texto_original": texto, "idioma_alvo": idioma, "traducao": _cache_traducao[chave], "do_cache": True})

    ultimo_erro = None
    for tentativa in range(3):
        try:
            traducao = GoogleTranslator(source="pt", target=idioma).translate(texto)
            _cache_traducao[chave] = traducao
            return jsonify({"texto_original": texto, "idioma_alvo": idioma, "traducao": traducao, "do_cache": False})
        except Exception as erro:  # serviço de tradução externo — instável por natureza, vale tentar de novo
            ultimo_erro = erro
            time.sleep(0.5)

    return jsonify({"erro": f"Falha ao traduzir depois de 3 tentativas: {ultimo_erro}"}), 502


_carregar_modelos()

if __name__ == "__main__":
    # PORT vem do ambiente em hosts como Hugging Face Spaces (exige 7860) —
    # em dev local, sem essa variável, cai no 8080 de sempre.
    porta = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=porta, debug=False)
