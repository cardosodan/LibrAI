"""Desambiguação geométrica pra pares de letras quase geometricamente
idênticos — Libras 'A' e 'S' são as duas o MESMO punho fechado, diferindo
só na posição do polegar (ao lado do punho em 'A', cruzado na frente dos
dedos em 'S').

**Descoberta desta rodada** (usuário reportou confusão constante entre A e
S ao vivo): comparando os centroides (pose "média" de cada letra, vindos de
`data/landmarks/alfabeto_landmarks.csv`) de TODAS as 210 combinações de
letras do alfabeto, A-S é o **9º par mais próximo** — não é impressão, é
uma das poses mais fisicamente parecidas que existem. Comparando ponto a
ponto, a ÚNICA diferença real está na PONTA DO POLEGAR (índice 4): distância
0.72 entre os centroides nesse ponto, contra ~0.05-0.3 em todos os outros 20.

**Testado e descartado**: usar só esse um ponto como critério de desempate
direto (`distância até o polegar do centroide de A` vs `de S`) não ajudou —
testado com ruído sintético simulando jitter de câmera ao vivo (várias
intensidades), foi neutro/levemente PIOR que simplesmente confiar no
modelo original. Um único ponto 3D é sensível demais a ruído sozinho,
mesmo sendo o ponto mais discriminativo em média — jogar fora as outras 20
dimensões (que ainda carregam sinal fraco, mas real) piora mais do que
ajuda.

**O que funcionou**: um classificador binário pequeno (Random Forest, só
A-vs-S, usando as 63 features normais — não só o polegar) treinado à parte
e usado como desempate SÓ quando o modelo principal (21 classes) já está em
dúvida entre exatamente essas duas letras (top-2 previsões == {A, S} com
confiança próxima). Testado com o mesmo ruído sintético: neutro em
condições limpas, ganho líquido real (mais ajuda que atrapalha) em
condições mais ruidosas — a aproximação mais parecida de câmera ao vivo que
dá pra simular sem gravação real. Ganho modesto, não uma correção mágica —
documentado honestamente, não vendido como mais do que é.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config

# Pares de letras conhecidos por serem quase geometricamente idênticos —
# cada entrada dispara o desambiguador binário só quando o modelo principal
# já está em dúvida entre EXATAMENTE essas duas letras. Adicionar um novo
# par aqui só é seguro se houver evidência real (relato do usuário +
# confirmação de distância de centroide baixa), não só teoria.
PARES_CONHECIDOS = [frozenset({"A", "S"})]

MARGEM_AMBIGUIDADE = 0.2  # diferença de confiança entre top1/top2 abaixo da qual consideramos "empate"

_classificadores_binarios: dict[frozenset, RandomForestClassifier] | None = None


def _treinar_classificadores_binarios() -> dict[frozenset, RandomForestClassifier]:
    if not config.ALFABETO_LANDMARKS_CSV.exists():
        return {}
    df = pd.read_csv(config.ALFABETO_LANDMARKS_CSV)
    colunas_features = [c for c in df.columns if c.startswith("f")]

    classificadores: dict[frozenset, RandomForestClassifier] = {}
    for par in PARES_CONHECIDOS:
        letra_a, letra_b = tuple(par)
        subconjunto = df[df["letra"].isin([letra_a, letra_b])]
        if subconjunto["letra"].nunique() < 2:
            continue
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=config.RANDOM_STATE
        )
        clf.fit(subconjunto[colunas_features].to_numpy(dtype=np.float32), subconjunto["letra"].to_numpy())
        classificadores[par] = clf
    return classificadores


def _obter_classificadores() -> dict[frozenset, RandomForestClassifier]:
    global _classificadores_binarios
    if _classificadores_binarios is None:
        _classificadores_binarios = _treinar_classificadores_binarios()
    return _classificadores_binarios


def desambiguar(vetor: np.ndarray, candidatos: list[dict]) -> tuple[str, float] | None:
    """Se os 2 candidatos mais prováveis do modelo principal formam um par
    CONHECIDO (ver `PARES_CONHECIDOS`) com confiança próxima, usa um
    classificador binário dedicado pra decidir entre as duas. Devolve
    `(letra, confianca)` só quando isso MUDA a decisão do modelo principal —
    `None` se não há o que desambiguar ou se o desempate concorda com o
    modelo original (a maioria dos casos, inclusive a maioria dos casos
    A/S — só entra em ação na fatia realmente ambígua)."""
    if len(candidatos) < 2:
        return None
    top1, top2 = candidatos[0], candidatos[1]
    if abs(top1["confianca"] - top2["confianca"]) > MARGEM_AMBIGUIDADE:
        return None

    par = frozenset({top1["letra"], top2["letra"]})
    classificadores = _obter_classificadores()
    clf = classificadores.get(par)
    if clf is None:
        return None

    probs = clf.predict_proba([vetor])[0]
    indice = int(np.argmax(probs))
    letra_escolhida = clf.classes_[indice]
    if letra_escolhida == top1["letra"]:
        return None

    return letra_escolhida, float(probs[indice])
