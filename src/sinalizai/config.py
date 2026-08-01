"""Configurações centrais do SinalizAI: caminhos, hiperparâmetros e listas de classes.

Mantido num único módulo para que scripts de dados, treino e inferência nunca
divirjam sobre onde os arquivos vivem ou quais classes existem.
"""
from pathlib import Path

# --- Caminhos -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LANDMARKS_DIR = DATA_DIR / "landmarks"

ALFABETO_RAW_DIR = RAW_DIR / "alfabeto"
PALAVRAS_RAW_DIR = RAW_DIR / "palavras"

ALFABETO_LANDMARKS_CSV = LANDMARKS_DIR / "alfabeto_landmarks.csv"
PALAVRAS_LANDMARKS_NPZ = LANDMARKS_DIR / "palavras_landmarks.npz"

MODELS_DIR = BASE_DIR / "models"
MEDIAPIPE_MODELS_DIR = MODELS_DIR / "mediapipe"
HAND_LANDMARKER_TASK = MEDIAPIPE_MODELS_DIR / "hand_landmarker.task"
HOLISTIC_LANDMARKER_TASK = MEDIAPIPE_MODELS_DIR / "holistic_landmarker.task"

MODELO_ALFABETO_PATH = MODELS_DIR / "classificador_alfabeto.joblib"
MODELO_PALAVRAS_PATH = MODELS_DIR / "classificador_palavras.pt"
PALAVRAS_LABELS_PATH = MODELS_DIR / "palavras_labels.json"

RELATORIOS_DIR = BASE_DIR / "relatorios"

# --- Classes do alfabeto -------------------------------------------------
# Só sinais ESTÁTICOS do alfabeto manual de Libras entram aqui. As letras
# H, J, K, X, Y, Z (entre outras, dependendo da fonte) exigem MOVIMENTO pra
# serem sinalizadas corretamente — um classificador de pose única (uma foto)
# não tem informação suficiente pra diferenciá-las de outras letras, então
# ficam de fora desta fase por design, não por esquecimento. Ver README.
LETRAS_ESTATICAS = ["A", "B", "C", "D", "E", "I", "L", "M", "N", "O", "R", "S", "U", "V", "W"]

# --- Hiperparâmetros de landmarks -----------------------------------------
NUM_LANDMARKS_MAO = 21          # pontos por mão (padrão MediaPipe Hands)
DIM_LANDMARK = 3                # x, y, z por ponto
NUM_LANDMARKS_POSE = 33          # pontos de pose (padrão MediaPipe Pose)

# Dimensão do vetor de features por frame usado no modelo dinâmico (palavras):
# pose (33) + mão esquerda (21) + mão direita (21), cada um com x,y,z.
DIM_FEATURES_FRAME = (NUM_LANDMARKS_POSE + 2 * NUM_LANDMARKS_MAO) * DIM_LANDMARK  # 225

# Todo vídeo de palavra é reamostrado pra este número fixo de frames antes de
# entrar na rede — sinais têm duração variável, mas o LSTM precisa de uma
# sequência de tamanho consistente por lote (batch).
SEQUENCE_LENGTH = 30

RANDOM_STATE = 42
