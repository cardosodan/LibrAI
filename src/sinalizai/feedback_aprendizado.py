"""Feedback de aprendizado pro alfabeto: compara a pose atual do usuário com o
"centroide" (média das amostras reais de treino) de uma letra específica, e
gera uma dica em português de qual parte da mão ajustar.

Deliberadamente determinístico/geométrico — sem LLM, sem API key, funciona
100% offline e sempre disponível. Os limiares de "quão perto é perto o
suficiente" são calibrados POR LETRA a partir da própria variação real do
dataset de treino (`config.ALFABETO_LANDMARKS_CSV`), não um número global
chutado: letras como M/N/P têm uma variação intra-classe (mesma letra,
amostras diferentes) bem maior que A/B/R, então um limiar único penalizaria
umas e seria frouxo demais com outras.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Nomeação dos 21 pontos do MediaPipe Hands, na MESMA ordem usada por
# HandLandmarkExtractor._normalizar (pulso primeiro, depois cada dedo da
# base pra ponta) — só pra tornar a dica legível ("ajuste o indicador"),
# não afeta a classificação em si.
NOMES_PONTOS = [
    "pulso",
    "base do polegar", "meio do polegar", "junta do polegar", "ponta do polegar",
    "base do indicador", "meio do indicador", "junta do indicador", "ponta do indicador",
    "base do médio", "meio do médio", "junta do médio", "ponta do médio",
    "base do anelar", "meio do anelar", "junta do anelar", "ponta do anelar",
    "base do mindinho", "meio do mindinho", "junta do mindinho", "ponta do mindinho",
]

_perfis_letras: dict[str, dict] | None = None


def carregar_perfis_letras() -> dict[str, dict]:
    """Carrega (uma vez, cacheado no módulo) o perfil de cada letra a partir
    do CSV de treino do alfabeto — a mesma fonte usada por
    `treinar_alfabeto.py`, não um dataset separado. Cada perfil tem o
    centroide (pose "média") e os percentis 50/90 de distância das amostras
    reais até esse centroide, usados como referência de "normal" pra
    aquela letra especificamente."""
    global _perfis_letras
    if _perfis_letras is not None:
        return _perfis_letras
    if not config.ALFABETO_LANDMARKS_CSV.exists():
        _perfis_letras = {}
        return _perfis_letras

    df = pd.read_csv(config.ALFABETO_LANDMARKS_CSV)
    colunas_features = [c for c in df.columns if c.startswith("f")]

    perfis: dict[str, dict] = {}
    for letra, grupo in df.groupby("letra"):
        matriz = grupo[colunas_features].to_numpy(dtype=np.float32)
        centroide = matriz.mean(axis=0)
        pontos_centroide = centroide.reshape(-1, 3)
        distancias_treino = np.linalg.norm(
            matriz.reshape(len(matriz), -1, 3) - pontos_centroide, axis=2
        ).mean(axis=1)
        perfis[str(letra)] = {
            "centroide": centroide,
            "p50": float(np.percentile(distancias_treino, 50)),
            "p90": float(np.percentile(distancias_treino, 90)),
        }
    _perfis_letras = perfis
    return _perfis_letras


def gerar_feedback(letra_alvo: str, vetor_usuario: np.ndarray) -> dict | None:
    """Compara `vetor_usuario` (63 features normalizadas, mesmo formato usado
    pelo classificador) com o perfil de `letra_alvo`. Devolve um dict com
    similaridade aproximada (0-1), uma dica textual sobre o ponto que mais
    diverge, e o nome desse ponto — ou `None` se não houver perfil pra essa
    letra (ex: modelo de alfabeto não treinado, ou letra fora do dataset)."""
    perfis = carregar_perfis_letras()
    perfil = perfis.get(letra_alvo)
    if perfil is None:
        return None

    pontos_usuario = vetor_usuario.reshape(-1, 3)
    pontos_centroide = perfil["centroide"].reshape(-1, 3)
    distancias = np.linalg.norm(pontos_usuario - pontos_centroide, axis=1)
    distancia_media = float(np.mean(distancias))

    # Referência de "aceitável" é o próprio p90 de distância intra-classe
    # dessa letra (90% das amostras reais de treino caem até aqui) — dá uma
    # similaridade calibrada pela variação natural de CADA letra, em vez de
    # um limiar global que seria frouxo demais pra letras com pouca variação
    # (A, B, R) e rígido demais pras que têm muita (M, N, P).
    referencia = max(perfil["p90"], 1e-3)
    similaridade = float(np.clip(1.0 - distancia_media / (referencia * 1.4), 0.0, 1.0))

    indice_pior = int(np.argmax(distancias))
    nome_pior = NOMES_PONTOS[indice_pior] if indice_pior < len(NOMES_PONTOS) else f"ponto {indice_pior}"

    if distancia_media <= perfil["p50"]:
        dica = f"Muito próximo do padrão real de '{letra_alvo}' — mantenha assim."
    elif indice_pior == 0:
        dica = f"Reposicione a mão inteira — o pulso está numa posição bem diferente do esperado pra '{letra_alvo}'."
    else:
        # Direção do ajuste: compara a distância do ponto que mais diverge
        # até o PULSO (ponto 0), usuário vs. centroide — mais perto do pulso
        # que o esperado sugere dedo mais fechado/curvado; mais longe sugere
        # mais estendido/aberto.
        raio_usuario = float(np.linalg.norm(pontos_usuario[indice_pior] - pontos_usuario[0]))
        raio_centroide = float(np.linalg.norm(pontos_centroide[indice_pior] - pontos_centroide[0]))
        if raio_usuario < raio_centroide * 0.85:
            dica = f"A {nome_pior} está mais fechada/curvada do que o esperado pra '{letra_alvo}' — tente estender um pouco mais."
        elif raio_usuario > raio_centroide * 1.15:
            dica = f"A {nome_pior} está mais estendida/aberta do que o esperado pra '{letra_alvo}' — tente curvar um pouco mais."
        else:
            dica = f"Ajuste fino na {nome_pior} — a posição está próxima, mas não exata pra '{letra_alvo}'."

    return {
        "similaridade": round(similaridade, 3),
        "dica": dica,
        "parte_mais_divergente": nome_pior,
    }
