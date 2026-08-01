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
# Só sinais ESTÁTICOS do alfabeto manual de Libras entram aqui. H, J, X, Y, Z
# exigem MOVIMENTO pra serem sinalizadas corretamente — um classificador de
# pose única (uma foto) não tem informação suficiente pra diferenciá-las de
# outras letras, então ficam de fora desta fase por design, não por
# esquecimento. Ver README. (K entrou nesta lista com a expansão via
# Roboflow — o dataset trata a pose inicial dela como estática o bastante
# pra classificar numa foto só, mesmo tendo algum movimento na sinalização
# completa.)
LETRAS_ESTATICAS = [
    "A", "B", "C", "D", "E", "F", "G", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
]

# --- Hiperparâmetros de landmarks -----------------------------------------
NUM_LANDMARKS_MAO = 21          # pontos por mão (padrão MediaPipe Hands)
DIM_LANDMARK = 3                # x, y, z por ponto
NUM_LANDMARKS_POSE = 33          # pontos de pose (padrão MediaPipe Pose)

# Subconjunto de pontos do Face Mesh (478 no total — a maioria irrelevante
# pra gramática de Libras) relevantes pras marcas não-manuais: sobrancelha e
# topo do olho (levantar sobrancelha = pergunta/surpresa), boca (aberta/larga
# muda intensidade/negação), e os cantos externos dos olhos como referência
# de escala (distância entre-olhos, estável pra normalizar o resto).
INDICES_ROSTO = {
    "sobrancelha_esq": 105, "olho_topo_esq": 159,
    "sobrancelha_dir": 334, "olho_topo_dir": 386,
    "labio_superior": 13, "labio_inferior": 14,
    "canto_boca_esq": 61, "canto_boca_dir": 291,
    "olho_externo_esq": 33, "olho_externo_dir": 263,
}
NUM_LANDMARKS_ROSTO = len(INDICES_ROSTO)  # 10

# Dimensão do vetor de features por frame usado no modelo dinâmico (palavras):
# pose (33) + mão esquerda (21) + mão direita (21) + rosto (10), cada um com
# x,y,z. O rosto entrou depois que o resto já estava construído — Libras usa
# expressão facial pra gramática de verdade (negação, pergunta, intensidade),
# não é enfeite; ver HolisticSequenceExtractor.
DIM_FEATURES_FRAME = (NUM_LANDMARKS_POSE + 2 * NUM_LANDMARKS_MAO + NUM_LANDMARKS_ROSTO) * DIM_LANDMARK  # 255

# Todo vídeo de palavra é reamostrado pra este número fixo de frames antes de
# entrar na rede — sinais têm duração variável, mas o LSTM precisa de uma
# sequência de tamanho consistente por lote (batch).
SEQUENCE_LENGTH = 30

RANDOM_STATE = 42
