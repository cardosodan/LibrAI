"""Extração de landmarks (pontos-chave) via MediaPipe Tasks API.

Duas estratégias, escolhidas conforme a natureza de cada tarefa (ver README
para a justificativa completa dessa escolha):

- `HandLandmarkExtractor` — só as mãos (HandLandmarker), usado no alfabeto
  estático: a letra é 100% definida pela forma da mão numa única pose, então
  carregar pose/rosto seria peso morto.
- `HolisticSequenceExtractor` — mãos + pose (HolisticLandmarker), usado nas
  palavras dinâmicas: muitos sinais de Libras usam também braço/tronco, e o
  significado depende da trajetória ao longo do tempo, não de um frame só.

Ambas normalizam os pontos (translação + escala) pra que o resultado não
dependa de onde a pessoa está no quadro nem da distância até a câmera —
sem isso, o classificador aprenderia posição/tamanho em vez do sinal em si.
"""
from __future__ import annotations

import numpy as np

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from . import config

_EPS = 1e-6


def _distancia(a, b) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _checar_bundle(caminho) -> None:
    if not caminho.exists():
        raise FileNotFoundError(
            f"Bundle de modelo do MediaPipe não encontrado: {caminho}\nRode antes:\n"
            "  venv/Scripts/python.exe -m sinalizai.baixar_modelos_mediapipe"
        )


class HandLandmarkExtractor:
    """Extrai um vetor de 63 features (21 pontos x, y, z) de UMA mão numa imagem estática."""

    def __init__(self, num_hands: int = 1, min_confidence: float = 0.5):
        _checar_bundle(config.HAND_LANDMARKER_TASK)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.HAND_LANDMARKER_TASK)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)

    def extrair_de_bgr(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Recebe um frame no formato do OpenCV (BGR) e devolve o vetor normalizado, ou None se nenhuma mão foi encontrada."""
        resultado = self._detectar(frame_bgr)
        if not resultado.hand_landmarks:
            return None
        return self._normalizar(resultado.hand_landmarks[0])

    def extrair_com_pontos_de_bgr(self, frame_bgr: np.ndarray):
        """Igual `extrair_de_bgr`, mas também devolve os 21 pontos BRUTOS (x, y
        normalizados 0-1 relativos ao tamanho da imagem, sem a translação/escala
        aplicada pro classificador) — usado só pelo front-end pra desenhar o
        esqueleto da mão em cima do vídeo, não pra classificação."""
        resultado = self._detectar(frame_bgr)
        if not resultado.hand_landmarks:
            return None, None
        pontos = resultado.hand_landmarks[0]
        vetor = self._normalizar(pontos)
        pontos_brutos = [[p.x, p.y] for p in pontos]
        return vetor, pontos_brutos

    def _detectar(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect(img)

    def extrair_de_arquivo(self, caminho: str) -> np.ndarray | None:
        frame = cv2.imread(caminho)
        if frame is None:
            return None
        return self.extrair_de_bgr(frame)

    @staticmethod
    def _normalizar(pontos) -> np.ndarray:
        coords = np.array([[p.x, p.y, p.z] for p in pontos], dtype=np.float32)
        origem = coords[0]  # pulso
        escala = _distancia(coords[9], origem) or _EPS  # pulso -> base do dedo médio
        return ((coords - origem) / escala).flatten()

    def fechar(self):
        self._landmarker.close()


class HolisticSequenceExtractor:
    """Extrai uma sequência de vetores de 225 features (pose + 2 mãos) de um vídeo."""

    def __init__(self, min_confidence: float = 0.5):
        _checar_bundle(config.HOLISTIC_LANDMARKER_TASK)
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.HOLISTIC_LANDMARKER_TASK)),
            running_mode=vision.RunningMode.VIDEO,
            min_hand_landmarks_confidence=min_confidence,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(opts)
        # O modo VIDEO exige timestamps estritamente crescentes durante toda a
        # vida do objeto landmarker, não só dentro de uma chamada — reusar a
        # mesma instância pra vários vídeos (bem mais rápido que recarregar o
        # modelo a cada arquivo) exige acumular uma base entre um vídeo e outro.
        self._proxima_base_ms = 0

    def extrair_de_video(self, caminho: str) -> np.ndarray | None:
        cap = cv2.VideoCapture(caminho)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return None
        return self.extrair_de_frames(frames, fps)

    def extrair_de_frames(self, frames: list[np.ndarray], fps: float = 30.0) -> np.ndarray | None:
        """Igual a `extrair_de_video`, mas recebe frames BGR já em memória (uso: webcam ao vivo)."""
        frames_features = []
        timestamp_ms = self._proxima_base_ms
        for indice, frame in enumerate(frames):
            timestamp_ms = self._proxima_base_ms + int((indice / fps) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            resultado = self._landmarker.detect_for_video(img, timestamp_ms)
            frames_features.append(self._vetor_do_frame(resultado))
        self._proxima_base_ms = timestamp_ms + 1000  # margem de 1s antes do próximo lote
        if not frames_features:
            return None
        return _reamostrar(np.array(frames_features, dtype=np.float32), config.SEQUENCE_LENGTH)

    @staticmethod
    def _vetor_do_frame(resultado) -> np.ndarray:
        pose = _pontos_ou_zeros(resultado.pose_landmarks, config.NUM_LANDMARKS_POSE)
        mao_esq = _pontos_ou_zeros(resultado.left_hand_landmarks, config.NUM_LANDMARKS_MAO)
        mao_dir = _pontos_ou_zeros(resultado.right_hand_landmarks, config.NUM_LANDMARKS_MAO)

        if resultado.pose_landmarks:
            ombro_esq = np.array([pose[11 * 3], pose[11 * 3 + 1], pose[11 * 3 + 2]])
            ombro_dir = np.array([pose[12 * 3], pose[12 * 3 + 1], pose[12 * 3 + 2]])
            origem = (ombro_esq + ombro_dir) / 2
            escala = _distancia(ombro_esq, ombro_dir) or _EPS
        else:
            origem = np.zeros(3)
            escala = 1.0

        def normalizar_bloco(bloco: np.ndarray, presente: bool) -> np.ndarray:
            if not presente:
                return bloco  # já é zero
            pontos = bloco.reshape(-1, 3)
            return ((pontos - origem) / escala).flatten()

        pose_norm = normalizar_bloco(pose, bool(resultado.pose_landmarks))
        mao_esq_norm = normalizar_bloco(mao_esq, bool(resultado.left_hand_landmarks))
        mao_dir_norm = normalizar_bloco(mao_dir, bool(resultado.right_hand_landmarks))
        return np.concatenate([pose_norm, mao_esq_norm, mao_dir_norm])

    def fechar(self):
        self._landmarker.close()


def _pontos_ou_zeros(pontos, n: int) -> np.ndarray:
    if not pontos:
        return np.zeros(n * 3, dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in pontos], dtype=np.float32).flatten()


def _reamostrar(sequencia: np.ndarray, tamanho_alvo: int) -> np.ndarray:
    """Estica ou encolhe a sequência (T variável) pra exatamente `tamanho_alvo` frames.

    Sinais têm duração diferente entre execuções (mesma pessoa nunca sinaliza
    exatamente na mesma velocidade duas vezes) — reamostrar por interpolação
    linear no eixo do tempo preserva a FORMA do movimento em vez de só cortar
    ou preencher com zero, o que destruiria o final ou o início do gesto.
    """
    t_original = sequencia.shape[0]
    if t_original == tamanho_alvo:
        return sequencia
    indices_alvo = np.linspace(0, t_original - 1, tamanho_alvo)
    indices_base = np.arange(t_original)
    saida = np.empty((tamanho_alvo, sequencia.shape[1]), dtype=np.float32)
    for coluna in range(sequencia.shape[1]):
        saida[:, coluna] = np.interp(indices_alvo, indices_base, sequencia[:, coluna])
    return saida
