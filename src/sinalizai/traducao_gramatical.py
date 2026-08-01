"""Reorganização gramatical via LLM (opcional): Libras tem gramática própria
— ordem de palavras diferente do português (frequentemente tópico-comentário
ou SOV, sem artigos/preposições/conjugação verbal) — então a sequência crua
de letras/palavras que o usuário sinaliza (ex: "CASA EU IR") não é uma frase
gramatical em português de verdade. Antes desta rodada, essa sequência crua
ia direto pro Google Translate (pt->en), produzindo uma tradução em inglês
tão estranha quanto o português de entrada.

Este módulo faz UM passo extra antes da tradução: pede pra um LLM (Groq,
API compatível com o formato da OpenAI) reescrever a sequência crua como uma
frase em português gramatical, e SÓ ENTÃO ela é traduzida pro inglês.

Opt-in e com fallback gracioso: se `GROQ_API_KEY` não estiver configurada
(ou a chamada falhar por qualquer motivo — rede, limite de taxa, etc.),
`reorganizar_gramaticalmente` devolve `None` e quem chama usa o texto cru
como já fazia antes — nunca quebra o fluxo de tradução existente.
"""
from __future__ import annotations

import os

import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

PROMPT_SISTEMA = (
    "Você recebe uma sequência de palavras/letras soletradas em Libras (Língua "
    "Brasileira de Sinais), na ordem em que foram sinalizadas. Libras tem "
    "gramática própria, diferente do português: costuma vir sem artigos, sem "
    "preposições e numa ordem de palavras diferente (tópico-comentário ou "
    "SOV). Sua tarefa é reescrever essa sequência como UMA frase em português "
    "gramaticalmente natural, preservando o sentido original o máximo "
    "possível. Responda APENAS com a frase reescrita, sem aspas, sem "
    "explicações, sem comentários adicionais. Se a sequência já estiver "
    "gramatical, devolva-a como está."
)


def disponivel() -> bool:
    return bool(GROQ_API_KEY)


def reorganizar_gramaticalmente(texto_cru: str) -> str | None:
    """Chama o LLM pra reescrever `texto_cru` como português gramatical.
    Devolve `None` (nunca levanta exceção) se a chave não estiver configurada,
    a chamada falhar, ou a resposta vier vazia/idêntica ao texto original."""
    if not GROQ_API_KEY or not texto_cru.strip():
        return None

    try:
        resposta = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user", "content": texto_cru},
                ],
                "temperature": 0.3,
                "max_tokens": 200,
            },
            timeout=8,
        )
        resposta.raise_for_status()
        corpo = resposta.json()
        texto_reescrito = corpo["choices"][0]["message"]["content"].strip().strip('"')
    except Exception as erro:  # rede instável, limite de taxa, chave inválida, etc. — nunca deve derrubar /api/traduzir
        print(f"[SinalizAI] Reorganização gramatical via Groq falhou (seguindo com o texto original): {erro}")
        return None

    if not texto_reescrito or texto_reescrito.strip().lower() == texto_cru.strip().lower():
        return None
    return texto_reescrito
