"""Arquitetura do classificador de sinais dinâmicos (palavras).

Diferente do alfabeto (uma pose só, um vetor de features resolve), uma
palavra em Libras é uma TRAJETÓRIA — a mesma forma de mão em posições
diferentes ao longo do tempo pode ser um sinal completamente diferente. Por
isso aqui o modelo precisa processar uma sequência (LSTM), não um vetor
único (o Random Forest do alfabeto não serviria pra isso).

BiLSTM (bidirecional) porque, ao contrário de reconhecimento de fala em
tempo real, aqui já temos o clipe inteiro antes de classificar — não há
motivo pra abrir mão do contexto que vem "depois" no tempo.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificadorPalavrasLSTM(nn.Module):
    def __init__(self, dim_entrada: int, num_classes: int, dim_oculta: int = 128, num_camadas: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim_entrada,
            hidden_size=dim_oculta,
            num_layers=num_camadas,
            batch_first=True,
            dropout=dropout if num_camadas > 1 else 0.0,
            bidirectional=True,
        )
        self.classificador = nn.Sequential(
            nn.Linear(dim_oculta * 2, dim_oculta),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_oculta, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saida, _ = self.lstm(x)
        ultimo_passo = saida[:, -1, :]  # concatenação forward+backward no último timestep
        return self.classificador(ultimo_passo)
