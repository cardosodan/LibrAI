# SinalizAI

Reconhecimento de Libras (Língua Brasileira de Sinais) em tempo real via
webcam, usando MediaPipe + Machine Learning — sem depender de nenhuma API de
IA generativa paga: o pipeline inteiro roda localmente (CPU), do zero, com
dados reais.

Duas partes, propositalmente resolvidas com abordagens diferentes:

| | Alfabeto manual (datilologia) | Palavras dinâmicas |
|---|---|---|
| Natureza do sinal | Pose estática (uma foto já basta) | Movimento ao longo do tempo |
| Extração | MediaPipe **Hand Landmarker** (só mãos) | MediaPipe **Holistic Landmarker** (pose + mãos) |
| Classificador | Random Forest (venceu Regressão Logística — ver Resultados) | LSTM bidirecional (PyTorch) |
| Dataset | [Brazilian Sign Language Alphabet Dataset](https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset) (UNIVALI) + [Alfabeto em Libras](https://universe.roboflow.com/elainesilva/alfabeto-em-libras-qrvnw) (Roboflow, ElaineSilva) — 21 letras, ~6.100 imagens reais combinadas | Auto-gravado pela webcam (ver seção Dataset de Palavras) |
| Status | **Treinado e funcionando**, métricas reais abaixo | Pipeline completo e testado (smoke test), aguardando vídeos reais |

## Tradução ao vivo: Libras → Português → Inglês, com voz

A demo web (`app.py`) não só identifica letra por letra — ela **monta a
frase e traduz**, em tempo real:

1. **Reconhecimento contínuo e estável** (não é mais "clique e tire uma
   foto"): o navegador consulta `/api/reconhecer-letra` a cada 180ms, mas uma
   letra só é **confirmada** (entra na palavra) depois de aparecer estável
   em 4 leituras seguidas com confiança ≥ 60% — isso é o que resolve "rápida
   E certa" ao mesmo tempo: feedback quase instantâneo na tela (`letra
   atual`), mas sem deixar uma detecção isolada/ruidosa virar letra errada
   na palavra. Toda essa lógica de estabilidade roda no **cliente**
   (`static/js/app.js`), o backend continua stateless (1 frame → 1
   predição), mais simples de escalar/hospedar.
2. **Montagem automática de palavra e frase**: afastar a mão por ~0,9s fecha
   a palavra atual (junta as letras confirmadas); ~2,5s sem mão fecha a
   frase inteira e dispara a tradução sozinho. Também dá pra forçar isso na
   mão (botões "Fechar palavra" / "Traduzir agora" / "Nova frase").
3. **Tradução** via `/api/traduzir` (backend), usando `deep-translator`
   (wrapper livre/sem chave de API sobre o Google Translate) — **só
   português → inglês por enquanto**, a pedido explícito; adicionar
   francês/espanhol/etc é só acrescentar uma entrada em
   `IDIOMAS_SUPORTADOS` (`app.py`), o `GoogleTranslator` já suporta qualquer
   par de idiomas. Resultado cacheado em memória (mesma frase não traduz de
   novo).
4. **Voz**: assim que a tradução chega, o navegador fala o texto em inglês
   em voz alta via `window.speechSynthesis` (Web Speech API nativa do
   navegador — sem servidor de TTS, sem chave de API, sem depender de
   OS/plataforma no deploy).

**Recursos de interface adicionais** (pedido explícito do usuário — "faça
todos" numa lista de 11 melhorias):

- **Overlay do esqueleto da mão** desenhado em cima do vídeo (canvas
  transparente, `desenharEsqueleto()` em `app.js`) — mostra os 21 pontos e
  as conexões que o MediaPipe está enxergando, em tempo real.
- **Barra de progresso de confirmação** — enche conforme a letra atual
  acumula leituras estáveis, até confirmar.
- **Painel "modo desenvolvedor"** (checkbox) — mostra os 3 candidatos mais
  prováveis e a confiança de cada um, não só o vencedor.
- **Apagar última letra**, **editar a frase manualmente** (textarea, não
  mais texto fixo), **histórico de frases da sessão** com replay de áudio
  por item, e **botão de ouvir a tradução de novo**.
- **Indicador de conexão** (topo da página) — avisa se o servidor parar de
  responder, em vez de falhar silenciosamente.
- **Troca de câmera frontal/traseira** e layout responsivo pra celular.
- **Seção "Como funciona"** e **guia visual do alfabeto suportado** (miniaturas
  reais de cada uma das 15 letras) direto na página — pra quem abre o link
  sem passar pelo GitHub.

**Honestidade sobre o escopo**: isso funciona de ponta a ponta pro alfabeto
(soletração), que tem modelo real treinado. No modo palavra, cada
reconhecimento (ainda por clique + 2s de gravação, não contínuo) também
entra na frase e dispara tradução — mas como não há vocabulário de palavra
treinado neste repositório ainda (ver seção seguinte), isso está pronto pra
funcionar assim que houver dados, não é uma funcionalidade "fake" — só está
sem uso real até lá.

## Por que não "só jogar YOLOv8 em tudo"

YOLO resolve "onde está um objeto" (bounding box) — não "qual é a pose da
mão" nem "qual foi a trajetória do gesto". Reconhecimento de sinal depende de
**pontos-chave articulados no tempo**, não de detecção de objeto. Por isso o
pipeline usa:

- **MediaPipe Hand Landmarker** pro alfabeto: extrai 21 pontos 3D da mão por
  imagem. Como cada letra estática é 100% definida pela forma da mão numa
  pose só, um vetor de 63 números (21 × x,y,z) já é suficiente — carregar
  pose/rosto aqui seria peso morto.
- **MediaPipe Holistic Landmarker** pras palavras: adiciona os 33 pontos de
  pose (ombro, cotovelo, tronco) porque muitos sinais de Libras usam também
  o braço/corpo, não só a mão — e o significado de uma palavra está na
  **trajetória**, então o classificador tem que processar uma sequência
  (LSTM), não um vetor único.

Os dois extratores normalizam os pontos (translação pro pulso/centro dos
ombros + escala pela própria mão/largura dos ombros) — sem isso o modelo
aprenderia "a pessoa está longe da câmera" em vez do sinal em si.

## Resultados reais (alfabeto)

**Atualizado após expandir de 15 pra 21 letras** (mesclando o dataset original
da UNIVALI com o [Alfabeto em Libras](https://universe.roboflow.com/elainesilva/alfabeto-em-libras-qrvnw)
de ElaineSilva, Roboflow Universe, CC BY 4.0 — ver seção de créditos).
Rodando `treinar_alfabeto.py` sobre as ~6.100 imagens combinadas:

- **Detecção de mão pelo MediaPipe: 5.221/6.144 imagens (85,0%)** — as
  imagens sem detecção ficaram de fora do treino, não foram forçadas.
- Dois modelos comparados de propósito (nunca só um):

  | Modelo | F1 macro (teste, 20%) |
  |---|---|
  | Regressão Logística | 0,899 |
  | **Random Forest** | **0,939** |

  Desta vez o Random Forest venceu (o oposto do resultado com 15 letras,
  onde a Regressão Logística vencia) — com mais classes e handshapes mais
  próximos entre si (ex: F/T, que puxam a média pra baixo), a fronteira de
  decisão deixa de ser tão linearmente separável quanto antes. Comparar os
  dois modelos a cada retreino (não fixar um "vencedor" permanente) é
  exatamente o que permitiu perceber essa mudança.
- Matriz de confusão completa em [`relatorios/matriz_confusao_alfabeto.png`](relatorios/matriz_confusao_alfabeto.png),
  métricas por classe em [`relatorios/metricas_alfabeto.json`](relatorios/metricas_alfabeto.json).
- **Limitação honesta**: `F` (F1 0,69) e `T` (F1 0,72) são as classes mais
  fracas — poucas amostras (91 e 85 imagens antes do filtro de detecção,
  as menores do conjunto novo) combinadas com handshapes que aparentemente
  o modelo confunde mais entre si e com outras letras. Documentado, não
  escondido — classes com F1 baixo tendem a ser as recém-adicionadas com
  menos dado, não uma falha aleatória do pipeline.

## Por que 21 letras (não as 26)

O alfabeto manual de Libras tem letras que exigem **movimento** pra serem
sinalizadas corretamente. `H`, `J`, `X`, `Y`, `Z` ficam de fora do
classificador ESTÁTICO por design — um classificador de pose única (uma
foto) estruturalmente não tem informação temporal pra diferenciá-las de
outras letras. O caminho certo pra cobri-las é o mesmo usado pras palavras
(sequência + LSTM), item que está no roadmap.

`K` é um caso à parte: inicialmente catalogada aqui como "letra de
movimento" também, o dataset do Roboflow (ver Créditos) a trata como
estática — a pose inicial já é distinguível o suficiente numa foto só. Os
resultados confirmam isso na prática (F1 de 1,00 pra `K` com Random
Forest), então ela entrou na lista sem ressalva.

## Dataset de palavras: por que auto-gravado, não um dataset público baixado

Existe um dataset público de sinais isolados em Libras — o
[V-LIBRASIL](https://www.kaggle.com/datasets/davimedio01/v-librasil) (UFPE,
1.364 termos, 4.089 vídeos, 3 sinalizantes profissionais). Ele **não** foi
usado como caminho principal aqui por 3 motivos, todos verificados antes de
decidir:

1. **~11GB e exige conta Kaggle** — automatizar o download é possível (ver
   `baixar_dataset_palavras.py`), mas não dá pra fazer sem uma credencial
   pessoal.
2. **Licença CC BY-NC-ND 4.0** (não-comercial, sem obras derivadas) — ótimo
   pra pesquisa/portfólio/aprendizado, mas **não pode virar parte de um
   produto comercial** sem autorização do autor. Isso importa: deixar isso
   documentado explicitamente evita usar dado errado no lugar errado.
3. **Estrutura interna não verificável sem login** — não consegui inspecionar
   o layout exato de pastas/CSV do dataset sem autenticar, então o script de
   download automatizado é "melhor esforço" (baixa o bruto, mas a
   organização em `data/raw/palavras/<palavra>/*.mp4` é manual).

Por isso o caminho **principal e testado de verdade** é gravar um
vocabulário próprio pela webcam (`gravar_amostras_palavras.py`) — o que,
além de evitar os 3 problemas acima, é em si um exercício real de
engenharia de dados (protocolo de coleta, múltiplas tomadas, variação de
ângulo/iluminação) que vale mais pra portfólio do que só consumir um
dataset pronto.

## Pipeline testado ponta a ponta (sem dados reais de palavra ainda)

Como ainda não existe um vocabulário de palavras gravado neste repositório,
`tests/test_pipeline_palavras_smoke.py` valida que **o código mecanicamente
funciona** (extração → reamostragem → treino → salvamento, com as formas de
tensor corretas) usando vídeos sintéticos gerados a partir de fotos reais do
alfabeto — sem validar acurácia nenhuma (não é Libras de verdade). Isso é
uma escolha deliberada: relatar "funciona, ainda sem dado real" é mais
honesto que treinar em cima de um dataset improvisado e reportar uma
acurácia que não significa nada.

## Arquitetura do classificador de palavras (LSTM)

```
sequência (30 frames × 225 features) 
    → LSTM bidirecional, 2 camadas, 128 unidades ocultas
    → último passo de tempo (forward + backward concatenados)
    → Linear(256→128) → ReLU → Dropout → Linear(128→num_classes)
```

Bidirecional porque, diferente de reconhecimento de fala em tempo real, aqui
o clipe inteiro já está disponível antes de classificar — não há motivo pra
abrir mão do contexto que vem "depois" no tempo. Datasets de palavra tendem
a ser pequenos (poucas dezenas de amostras por classe gravando à mão), então
o treino aplica jitter gaussiano nas coordenadas como augmentação (cada
amostra de treino vira ~8 versões com ruído leve).

## Como rodar

### 0. Setup

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python.exe -m pip install -r requirements.txt
venv/Scripts/python.exe -m sinalizai.baixar_modelos_mediapipe
```

### 1. Alfabeto (já treinado neste repo — pra retreinar do zero)

```bash
venv/Scripts/python.exe -m sinalizai.baixar_dataset_alfabeto

# opcional, mas recomendado — expande de 15 pra 21 letras (F,G,K,P,Q,T);
# exige conta grátis em roboflow.com + API key em Settings > API Keys
$env:ROBOFLOW_API_KEY = "sua_chave"   # ou export no bash
venv/Scripts/python.exe -m sinalizai.baixar_dataset_alfabeto_roboflow

venv/Scripts/python.exe -m sinalizai.preparar_dataset_alfabeto
venv/Scripts/python.exe -m sinalizai.treinar_alfabeto
```

### 2. Palavras (vocabulário próprio — recomendado)

```bash
venv/Scripts/python.exe -m sinalizai.gravar_amostras_palavras --palavra ola --amostras 20
venv/Scripts/python.exe -m sinalizai.gravar_amostras_palavras --palavra obrigado --amostras 20
# ... repita pra cada palavra do seu vocabulário (recomendo começar com 10-20 palavras)
venv/Scripts/python.exe -m sinalizai.preparar_dataset_palavras
venv/Scripts/python.exe -m sinalizai.treinar_palavras
```

### 2b. Palavras (V-LIBRASIL — alternativa, ver licença acima)

```bash
venv/Scripts/python.exe -m pip install kaggle
# configure ~/.kaggle/kaggle.json antes
venv/Scripts/python.exe -m sinalizai.baixar_dataset_palavras
# organize manualmente em data/raw/palavras/<palavra>/*.mp4, depois:
venv/Scripts/python.exe -m sinalizai.preparar_dataset_palavras
venv/Scripts/python.exe -m sinalizai.treinar_palavras
```

### 3. Demo

```bash
# Desktop (OpenCV, mostra a janela da webcam direto)
venv/Scripts/python.exe -m sinalizai.demo_webcam

# Web (Flask — o mesmo modelo, front-end no navegador)
venv/Scripts/python.exe app.py
# abrir http://127.0.0.1:8080
```

Os dois modos degradam graciosamente: se só o modelo do alfabeto existir,
o reconhecimento de palavra é desativado com uma mensagem clara em vez de
quebrar.

## Deploy (Render)

```
Build Command: pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && python -m sinalizai.baixar_modelos_mediapipe
Start Command: gunicorn app:app
```

**Não verificado nesta sessão**: disponibilidade de wheel pré-compilada do
MediaPipe/PyTorch pra Python 3.14 no ambiente Linux do Render (confirmado
aqui só pra Windows/cp314). Se a wheel não existir pra a versão de Python
do Render, fixar `runtime.txt` numa versão mais antiga (3.11/3.12) resolve.

## Estrutura do projeto

```
SinalizAI/
├── app.py                          # Flask (demo web)
├── requirements.txt / Procfile
├── src/sinalizai/
│   ├── config.py                   # caminhos, classes, hiperparâmetros
│   ├── landmarks.py                # extratores MediaPipe (Hand / Holistic)
│   ├── modelo_dinamico.py          # arquitetura LSTM (PyTorch)
│   ├── baixar_modelos_mediapipe.py
│   ├── baixar_dataset_alfabeto.py / preparar_dataset_alfabeto.py / treinar_alfabeto.py
│   ├── gravar_amostras_palavras.py / baixar_dataset_palavras.py
│   ├── preparar_dataset_palavras.py / treinar_palavras.py
│   └── demo_webcam.py              # demo desktop OpenCV
├── templates/index.html + static/  # front-end da demo web
├── tests/test_pipeline_palavras_smoke.py
├── models/                         # modelos TREINADOS (versionados)
├── relatorios/                     # matrizes de confusão + métricas (versionados)
├── data/                           # raw/landmarks (NÃO versionados — ver .gitignore)
└── docs/creditos/                  # licenças/atribuição dos datasets usados
```

## Limitações conhecidas (documentadas de propósito, não escondidas)

- **Só letras estáticas** (21/26) — ver seção acima.
- **Classes desbalanceadas** no alfabeto pós-filtro de detecção — as letras
  recém-adicionadas via Roboflow (`F`, `G`, `K`, `P`, `Q`, `T`) têm bem menos
  amostras (60-100) que as herdadas da UNIVALI (`A`-`E`, 500+); `F`/`T` são
  as classes com F1 mais baixo, provavelmente por causa disso.
- **Modelo de palavras sem dado real ainda** — código completo e testado
  mecanicamente, mas nenhum vocabulário genuíno de Libras foi gravado/
  treinado neste repositório até agora.
- **Sem expressões não-manuais** (sobrancelha, boca, direção do olhar) — a
  gramática de Libras usa isso pra negação, pergunta, intensidade etc.; este
  projeto reconhece só a configuração manual/corporal.
- **Reconhecimento de palavra não é contínuo** — a demo exige apertar um
  botão e gravar uma janela fixa de 2s, não segmenta sinais dentro de um
  vídeo corrido (esse é um problema de pesquisa em aberto — "continuous
  sign language recognition" — de propósito fora do escopo de um MVP solo).
- **Tradução só português → inglês por enquanto** — a pedido explícito, não
  limitação técnica (ver seção "Tradução ao vivo" — adicionar idioma é
  trivial).
- **Letra repetida na mesma palavra exige um toque de "reset"**: a lógica de
  estabilidade só confirma uma letra NOVA quando ela é diferente da última
  confirmada — soletrar algo com letra dobrada (ex: "ss") exige afastar a
  mão brevemente entre as duas, senão a segunda ocorrência não conta como
  uma confirmação nova.
- **Voz depende do navegador** (`window.speechSynthesis`, Web Speech API) —
  funciona nos principais navegadores desktop, mas suporte/qualidade de voz
  em inglês pode variar por SO/navegador; sem fallback de TTS server-side.

## Próximos passos

- Augmentação mais forte pro modelo de palavras (espelhamento horizontal,
  time-warping) além do jitter gaussiano atual.
- Comparar o LSTM contra uma baseline de k-NN + Dynamic Time Warping
  (clássico em reconhecimento de gesto com poucos dados por classe).
- Cobrir as letras dinâmicas (H, J, K, X, Y, Z) reaproveitando o pipeline de
  sequência já construído pras palavras.
- Mais idiomas de tradução (francês, espanhol, ...) — trivial de adicionar
  em `IDIOMAS_SUPORTADOS` (`app.py`), só não veio nesta rodada por pedido
  explícito de focar em português→inglês primeiro.
- Validar com um sinalizante/intérprete de Libras de verdade — o feedback
  que mais importa e que nenhuma métrica offline substitui.

## Créditos e licenças

- **Alfabeto (base, 15 letras)**: [Brazilian Sign Language Alphabet Dataset](https://github.com/biankatpas/Brazilian-Sign-Language-Alphabet-Dataset)
  — Passos, Fernandes & Comunello, UNIVALI, licença MIT (cópia em
  [`docs/creditos/`](docs/creditos/)). Nota de transparência: as imagens têm
  origem em fotos de ASL (American Sign Language) curadas pelos autores
  originais pelas letras com a mesma configuração manual em Libras — não
  são fotos nativas de sinalizantes de Libras.
- **Alfabeto (expansão, +6 letras: F, G, K, P, Q, T)**: [Alfabeto em Libras](https://universe.roboflow.com/elainesilva/alfabeto-em-libras-qrvnw)
  — ElaineSilva, Roboflow Universe, licença CC BY 4.0. Essas, sim, são fotos
  reais de sinalizantes de Libras (não derivadas de ASL).
- **V-LIBRASIL** (mencionado, não usado como dado de treino nesta versão):
  Rodrigues, UFPE, licença CC BY-NC-ND 4.0.
- **MediaPipe** (Apache 2.0) — Google.
