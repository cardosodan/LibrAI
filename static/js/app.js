const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlayEsqueleto = document.getElementById("overlay-esqueleto");
const overlay = document.getElementById("overlay");
const overlayTexto = document.getElementById("overlay-texto");
const visorScanner = document.getElementById("visor-scanner");
const resultado = document.getElementById("resultado");
const btnGravarPalavra = document.getElementById("btn-gravar-palavra");
const abas = document.querySelectorAll(".aba");
const modoAlfabetoEl = document.getElementById("modo-alfabeto");
const modoPalavraEl = document.getElementById("modo-palavra");
const estadoConexao = document.getElementById("estado-conexao");
const estadoConexaoTexto = document.getElementById("estado-conexao-texto");

const elLetraAtual = document.getElementById("letra-atual");
const elPalavraAtual = document.getElementById("palavra-atual");
const elFraseAtual = document.getElementById("frase-atual");
const elFraseTraduzida = document.getElementById("frase-traduzida");
const barraProgressoFill = document.getElementById("barra-progresso-fill");
const btnApagarLetra = document.getElementById("btn-apagar-letra");
const btnFecharPalavra = document.getElementById("btn-fechar-palavra");
const btnTraduzirAgora = document.getElementById("btn-traduzir-agora");
const btnRepetirAudio = document.getElementById("btn-repetir-audio");
const btnNovaFrase = document.getElementById("btn-nova-frase");
const btnCamera = document.getElementById("btn-camera");
const btnTrocarCamera = document.getElementById("btn-trocar-camera");
const chkModoDev = document.getElementById("chk-modo-dev");
const painelDev = document.getElementById("painel-dev");
const listaHistorico = document.getElementById("lista-historico");

const btnTema = document.getElementById("btn-tema");
const toastLetra = document.getElementById("toast-letra");
const sparklineCanvas = document.getElementById("sparkline-confianca");
const telemetriaDev = document.getElementById("telemetria-dev");
const telemetriaFps = document.getElementById("telemetria-fps");
const telemetriaLatencia = document.getElementById("telemetria-latencia");
const btnTelaCheia = document.getElementById("btn-tela-cheia");
const painelCamera = document.querySelector(".painel-camera");
const btnCopiarFrase = document.getElementById("btn-copiar-frase");
const modalLetra = document.getElementById("modal-letra");
const modalLetraTitulo = document.getElementById("modal-letra-titulo");
const modalLetraImg = document.getElementById("modal-letra-img");
const modalLetraStatus = document.getElementById("modal-letra-status");
const btnFecharModal = document.getElementById("btn-fechar-modal");
const feedbackAprendizado = document.getElementById("feedback-aprendizado");
const pistaFacial = document.getElementById("pista-facial");

let modoAtual = "alfabeto";
let poolingAtivo = false;
let gravandoPalavra = false;
let streamAtual = null;
let facingModeAtual = "user";

const ctx = canvas.getContext("2d");
const ctxEsqueleto = overlayEsqueleto.getContext("2d");

// --- Parâmetros do reconhecimento contínuo (soletração) -------------------
// "Rápida e certa identificação": o polling é rápido (feedback quase
// instantâneo em letra-atual), mas uma letra só é CONFIRMADA (adicionada na
// palavra) depois de aparecer estável por N leituras seguidas — evita que
// uma detecção isolada/ruidosa vire uma letra errada na palavra.
const INTERVALO_POLL_MS = 180;
const JANELA_ESTABILIDADE = 4;      // leituras seguidas iguais pra confirmar
const CONFIANCA_MINIMA = 0.6;
const PAUSA_PALAVRA_MS = 900;       // sem mão por esse tempo -> fecha a palavra atual
const PAUSA_FRASE_MS = 2500;        // sem mão por esse tempo -> fecha a frase e traduz
const FALHAS_PARA_MOSTRAR_ERRO = 3; // polls seguidos falhando até avisar "sem conexão"

let letraCandidata = null;
let contagemCandidata = 0;
let letraConfirmada = null;
let palavraAtual = "";
let fraseAtual = "";
let ultimaFraseTraduzida = "";
let ultimaTraducaoTexto = "";
let semMaoDesde = null;
let falhasConsecutivas = 0;
let historicoSessao = [];
let letrasPraticadas = new Set();

// --- Esqueleto da mão (conexões padrão MediaPipe Hands, 21 pontos) --------
const CONEXOES_MAO = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // polegar
  [0, 5], [5, 6], [6, 7], [7, 8],       // indicador
  [0, 9], [9, 10], [10, 11], [11, 12],  // médio
  [0, 13], [13, 14], [14, 15], [15, 16], // anelar
  [0, 17], [17, 18], [18, 19], [19, 20], // mindinho
  [5, 9], [9, 13], [13, 17],             // palma
];

// --- Trilha de movimento (modo dev): guarda as últimas posições do centro
// da mão (base do dedo médio, ponto 9) e desenha com opacidade decrescente,
// como um rastro de cometa — só ativo junto com o esqueleto em modo dev.
const TAMANHO_TRILHA = 12;
let trilhaMao = [];

// --- Sparkline de confiança: rolling window das últimas leituras ----------
const TAMANHO_SPARKLINE = 40;
let historicoConfianca = [];
const ctxSparkline = sparklineCanvas ? sparklineCanvas.getContext("2d") : null;

// --- Telemetria de FPS/latência (modo dev) --------------------------------
let ultimosTemposPoll = [];
const JANELA_TELEMETRIA = 20;

async function iniciarCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: facingModeAtual },
      audio: false,
    });
    streamAtual = stream;
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    overlayEsqueleto.width = canvas.width;
    overlayEsqueleto.height = canvas.height;
    atualizarEspelhamento();
    overlay.style.display = "flex";
    overlayTexto.textContent = "Câmera pronta";
    setTimeout(() => (overlay.style.display = "none"), 1500);
    if (btnCamera) {
      btnCamera.textContent = "PARAR CÂMERA";
      btnCamera.classList.remove("desligada");
    }
    iniciarPollingAlfabeto();
  } catch (erro) {
    overlay.style.display = "flex";
    overlayTexto.textContent = "Não consegui acessar a câmera: " + erro.message;
    if (btnCamera) {
      btnCamera.textContent = "TENTAR NOVAMENTE";
      btnCamera.classList.add("desligada");
    }
  }
}

function atualizarEspelhamento() {
  // Câmera frontal ("user"): espelha, como um espelho de verdade.
  // Câmera traseira ("environment"): NÃO espelha — senão o mundo real apareceria invertido.
  const transformacao = facingModeAtual === "user" ? "scaleX(-1)" : "none";
  video.style.transform = transformacao;
  overlayEsqueleto.style.transform = transformacao;
}

function pararCamera() {
  poolingAtivo = false;
  if (streamAtual) {
    streamAtual.getTracks().forEach((faixa) => faixa.stop());
    streamAtual = null;
  }
  video.srcObject = null;
  ctxEsqueleto.clearRect(0, 0, overlayEsqueleto.width, overlayEsqueleto.height);
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  overlay.style.display = "flex";
  overlayTexto.textContent = "Câmera desligada";
  if (btnCamera) {
    btnCamera.textContent = "LIGAR CÂMERA";
    btnCamera.classList.add("desligada");
  }
  visorScanner.classList.remove("analisando", "confirmado");
}

function alternarCamera() {
  if (streamAtual) {
    pararCamera();
  } else {
    iniciarCamera();
  }
}

async function trocarCameraFrontalTraseira() {
  facingModeAtual = facingModeAtual === "user" ? "environment" : "user";
  if (streamAtual) {
    pararCamera();
    await iniciarCamera();
  }
}

function capturarFrameBase64() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7);
}

// O overlay do esqueleto só aparece em Modo Desenvolvedor — por padrão a
// câmera fica limpa. Isso corrige dois problemas reais de uma vez: (1) o
// desenho competia visualmente com a leitura do próprio gesto, e (2) as
// cores estavam hardcoded em tons antigos (azul/verde de antes da troca de
// paleta), nunca acompanharam a identidade nova. Traços finos e cor única
// (--accent-signal a baixa opacidade) agora, coerente com o resto do site.
// Lê o valor AO VIVO de uma CSS custom property — canvas 2D não entende
// var(--token) direto, e um hex hardcoded aqui já causou bug real antes (o
// esqueleto ficava com cor de uma paleta antiga depois de trocar de tema).
// Agora, com o alternador claro/escuro, isso importa ainda mais: sem isso,
// o overlay ficaria preso na cor de um dos dois temas.
function corVar(nome) {
  return getComputedStyle(document.documentElement).getPropertyValue(nome).trim();
}

function desenharEsqueleto(pontos) {
  ctxEsqueleto.clearRect(0, 0, overlayEsqueleto.width, overlayEsqueleto.height);
  if (!chkModoDev.checked) { trilhaMao = []; return; }
  if (!pontos) return;
  const w = overlayEsqueleto.width;
  const h = overlayEsqueleto.height;

  // Trilha: guarda a base do dedo médio (ponto 9, o "centro" da mão) e
  // desenha um rastro com opacidade decrescente, mais antigo = mais apagado.
  trilhaMao.push([pontos[9][0] * w, pontos[9][1] * h]);
  if (trilhaMao.length > TAMANHO_TRILHA) trilhaMao.shift();
  const corSinal = corVar("--accent-signal");
  trilhaMao.forEach(([x, y], i) => {
    const opacidade = ((i + 1) / trilhaMao.length) * 0.3;
    ctxEsqueleto.beginPath();
    ctxEsqueleto.fillStyle = `color-mix(in srgb, ${corSinal} ${Math.round(opacidade * 100)}%, transparent)`;
    ctxEsqueleto.arc(x, y, 3, 0, 2 * Math.PI);
    ctxEsqueleto.fill();
  });

  ctxEsqueleto.strokeStyle = `color-mix(in srgb, ${corSinal} 55%, transparent)`;
  ctxEsqueleto.lineWidth = 1.5;
  CONEXOES_MAO.forEach(([a, b]) => {
    ctxEsqueleto.beginPath();
    ctxEsqueleto.moveTo(pontos[a][0] * w, pontos[a][1] * h);
    ctxEsqueleto.lineTo(pontos[b][0] * w, pontos[b][1] * h);
    ctxEsqueleto.stroke();
  });

  ctxEsqueleto.fillStyle = `color-mix(in srgb, ${corVar("--text-primary")} 85%, transparent)`;
  pontos.forEach(([x, y]) => {
    ctxEsqueleto.beginPath();
    ctxEsqueleto.arc(x * w, y * h, 2.2, 0, 2 * Math.PI);
    ctxEsqueleto.fill();
  });
}

function falar(texto, idioma) {
  if (!texto || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(texto);
  utter.lang = idioma;
  window.speechSynthesis.speak(utter);
}

// --- Tema claro/escuro, persistido no localStorage -------------------------
function aplicarTema(tema) {
  if (tema === "light") {
    document.documentElement.setAttribute("data-theme", "light");
    if (btnTema) btnTema.textContent = "MODO ESCURO";
  } else {
    document.documentElement.removeAttribute("data-theme");
    if (btnTema) btnTema.textContent = "MODO CLARO";
  }
}

function alternarTema() {
  const claroAgora = document.documentElement.getAttribute("data-theme") === "light";
  const novoTema = claroAgora ? "dark" : "light";
  aplicarTema(novoTema);
  localStorage.setItem("sinalizai-tema", novoTema);
}

// --- Toast de letra confirmada ---------------------------------------------
function mostrarToastLetra(letra) {
  if (!toastLetra) return;
  toastLetra.textContent = letra;
  toastLetra.classList.remove("animar");
  // força reflow pra poder re-disparar a mesma animação em confirmações seguidas
  void toastLetra.offsetWidth;
  toastLetra.classList.add("animar");
}

// --- Sparkline de confiança --------------------------------------------
function registrarConfianca(valor) {
  historicoConfianca.push(valor);
  if (historicoConfianca.length > TAMANHO_SPARKLINE) historicoConfianca.shift();
  desenharSparkline();
}

function desenharSparkline() {
  if (!ctxSparkline) return;
  const larguraCss = sparklineCanvas.clientWidth || 200;
  const alturaCss = sparklineCanvas.clientHeight || 28;
  if (sparklineCanvas.width !== larguraCss) sparklineCanvas.width = larguraCss;
  if (sparklineCanvas.height !== alturaCss) sparklineCanvas.height = alturaCss;
  ctxSparkline.clearRect(0, 0, larguraCss, alturaCss);
  if (historicoConfianca.length < 2) return;

  ctxSparkline.strokeStyle = corVar("--accent-signal");
  ctxSparkline.lineWidth = 1.5;
  ctxSparkline.beginPath();
  historicoConfianca.forEach((valor, i) => {
    const x = (i / (TAMANHO_SPARKLINE - 1)) * larguraCss;
    const y = alturaCss - valor * alturaCss;
    if (i === 0) ctxSparkline.moveTo(x, y);
    else ctxSparkline.lineTo(x, y);
  });
  ctxSparkline.stroke();

  // linha de referência pontilhada no limiar mínimo de confiança
  ctxSparkline.strokeStyle = corVar("--text-muted");
  ctxSparkline.setLineDash([2, 3]);
  ctxSparkline.lineWidth = 1;
  const yLimiar = alturaCss - CONFIANCA_MINIMA * alturaCss;
  ctxSparkline.beginPath();
  ctxSparkline.moveTo(0, yLimiar);
  ctxSparkline.lineTo(larguraCss, yLimiar);
  ctxSparkline.stroke();
  ctxSparkline.setLineDash([]);
}

// --- Telemetria de FPS/latência (modo dev) ---------------------------------
function registrarTempoPoll(latenciaMs) {
  ultimosTemposPoll.push(latenciaMs);
  if (ultimosTemposPoll.length > JANELA_TELEMETRIA) ultimosTemposPoll.shift();
  if (!chkModoDev.checked || !telemetriaDev) return;
  telemetriaDev.hidden = false;
  const media = ultimosTemposPoll.reduce((a, b) => a + b, 0) / ultimosTemposPoll.length;
  telemetriaLatencia.textContent = media.toFixed(0);
  telemetriaFps.textContent = (1000 / Math.max(media, 1)).toFixed(1);
}

// --- Modo tela cheia da câmera ----------------------------------------------
function alternarTelaCheia() {
  const ativo = painelCamera.classList.toggle("tela-cheia");
  if (btnTelaCheia) btnTelaCheia.textContent = ativo ? "SAIR" : "EXPANDIR";
}

// --- Copiar frase ------------------------------------------------------------
async function copiarFrase() {
  const texto = [fraseAtual, ultimaTraducaoTexto].filter(Boolean).join(" / ");
  if (!texto) return;
  try {
    await navigator.clipboard.writeText(texto);
    btnCopiarFrase.classList.add("copiado");
    const textoOriginal = btnCopiarFrase.textContent;
    btnCopiarFrase.textContent = "COPIADO";
    setTimeout(() => {
      btnCopiarFrase.classList.remove("copiado");
      btnCopiarFrase.textContent = textoOriginal;
    }, 1500);
  } catch (erro) {
    resultado.textContent = "Não consegui copiar (permissão do navegador).";
  }
}

// --- Selo de progresso no guia do alfabeto ----------------------------------
function marcarLetraPraticada(letra) {
  if (letrasPraticadas.has(letra)) return;
  letrasPraticadas.add(letra);
  const item = document.querySelector(`.guia-item[data-letra="${letra}"]`);
  if (item) item.classList.add("praticada");
}

// --- Modal da letra do guia --------------------------------------------------
function abrirModalLetra(item) {
  const letra = item.dataset.letra;
  const img = item.querySelector("img");
  modalLetraTitulo.textContent = letra;
  modalLetraImg.src = img.src;
  modalLetraImg.alt = img.alt;
  modalLetraStatus.textContent = letrasPraticadas.has(letra)
    ? "Você já sinalizou essa letra com sucesso nesta sessão."
    : "Ainda não sinalizada nesta sessão — tente na aba Alfabeto.";
  modalLetra.hidden = false;
}

function fecharModalLetra() {
  modalLetra.hidden = true;
}

function atualizarUiFrase() {
  elPalavraAtual.textContent = palavraAtual || "_";
  if (document.activeElement !== elFraseAtual) {
    elFraseAtual.value = fraseAtual;
  }
}

function escapeHtml(texto) {
  const div = document.createElement("div");
  div.textContent = texto;
  return div.innerHTML;
}

function renderizarHistorico() {
  if (historicoSessao.length === 0) {
    listaHistorico.innerHTML = '<li class="historico-vazio">Nenhuma frase traduzida ainda.</li>';
    return;
  }
  listaHistorico.innerHTML = historicoSessao
    .map(
      (item, i) => `
      <li class="historico-item">
        <span class="historico-texto">
          <span class="pt">${escapeHtml(item.pt)}</span> → <span class="en">${escapeHtml(item.en)}</span>
          <span class="hora">${item.hora}</span>
        </span>
        <button data-idx="${i}" class="btn-historico-repetir" title="Ouvir de novo">🔊</button>
      </li>`
    )
    .join("");
  listaHistorico.querySelectorAll(".btn-historico-repetir").forEach((btn) => {
    btn.addEventListener("click", () => falar(historicoSessao[+btn.dataset.idx].en, "en-US"));
  });
}

function adicionarAoHistorico(pt, en) {
  const hora = new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
  historicoSessao.unshift({ pt, en, hora });
  renderizarHistorico();
}

async function traduzirFrase(texto) {
  const alvo = (texto || "").trim();
  if (!alvo) return;
  elFraseTraduzida.textContent = "Traduzindo...";
  try {
    const resposta = await fetch("/api/traduzir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto: alvo, idioma: "en" }),
    });
    const dados = await resposta.json();
    if (dados.erro) {
      elFraseTraduzida.textContent = dados.erro;
      return;
    }
    elFraseTraduzida.textContent = dados.traducao;
    ultimaTraducaoTexto = dados.traducao;
    adicionarAoHistorico(alvo, dados.traducao);
    falar(dados.traducao, "en-US");
  } catch (erro) {
    elFraseTraduzida.textContent = "Erro ao traduzir (conexão).";
  }
}

function fecharPalavraAtual() {
  if (!palavraAtual) return;
  fraseAtual = fraseAtual ? `${fraseAtual} ${palavraAtual}` : palavraAtual;
  palavraAtual = "";
  letraConfirmada = null;
  letraCandidata = null;
  contagemCandidata = 0;
  barraProgressoFill.style.width = "0%";
  barraProgressoFill.classList.remove("completa");
  visorScanner.classList.remove("analisando", "confirmado");
  atualizarUiFrase();
}

function apagarUltimaLetra() {
  if (!palavraAtual) return;
  palavraAtual = palavraAtual.slice(0, -1);
  letraConfirmada = null; // permite confirmar a mesma letra de novo, se for sinalizada outra vez
  atualizarUiFrase();
}

function novaFrase() {
  palavraAtual = "";
  fraseAtual = "";
  ultimaFraseTraduzida = "";
  letraConfirmada = null;
  letraCandidata = null;
  contagemCandidata = 0;
  semMaoDesde = null;
  barraProgressoFill.style.width = "0%";
  barraProgressoFill.classList.remove("completa");
  visorScanner.classList.remove("analisando", "confirmado");
  elLetraAtual.textContent = "—";
  elFraseTraduzida.textContent = "—";
  atualizarUiFrase();
}

function atualizarPainelDev(candidatos) {
  if (!chkModoDev.checked || !candidatos) {
    painelDev.hidden = true;
    return;
  }
  painelDev.hidden = false;
  painelDev.innerHTML = candidatos
    .map((c) => {
      const pct = (c.confianca * 100).toFixed(0);
      return `
      <div class="candidato-linha">
        <span class="candidato-letra">${c.letra}</span>
        <div class="candidato-barra"><div class="candidato-barra-fill" style="width:${pct}%"></div></div>
        <span class="candidato-pct">${pct}%</span>
      </div>`;
    })
    .join("");
}

function marcarConexao(ok) {
  if (ok) {
    falhasConsecutivas = 0;
    estadoConexaoTexto.textContent = "conectado";
    estadoConexao.className = "estado-conexao ok";
  } else {
    falhasConsecutivas++;
    if (falhasConsecutivas >= FALHAS_PARA_MOSTRAR_ERRO) {
      estadoConexaoTexto.textContent = "sem conexão com o servidor";
      estadoConexao.className = "estado-conexao erro";
    }
  }
}

function processarDeteccaoLetra(dados) {
  atualizarPainelDev(dados.candidatos);

  if (!dados.detectado) {
    desenharEsqueleto(null);
    letraCandidata = null;
    contagemCandidata = 0;
    barraProgressoFill.style.width = "0%";
    barraProgressoFill.classList.remove("completa");
    visorScanner.classList.remove("analisando", "confirmado");
    letraConfirmada = null;
    if (semMaoDesde === null) semMaoDesde = Date.now();

    const semMaoMs = Date.now() - semMaoDesde;
    if (semMaoMs > PAUSA_PALAVRA_MS) {
      fecharPalavraAtual();
    }
    if (semMaoMs > PAUSA_FRASE_MS && fraseAtual && fraseAtual !== ultimaFraseTraduzida) {
      ultimaFraseTraduzida = fraseAtual;
      traduzirFrase(fraseAtual);
    }
    return;
  }

  semMaoDesde = null;
  desenharEsqueleto(dados.landmarks);
  elLetraAtual.textContent = `${dados.letra} (${(dados.confianca * 100).toFixed(0)}%)`;
  registrarConfianca(dados.confianca);

  if (dados.confianca < CONFIANCA_MINIMA) return;

  if (dados.letra === letraCandidata) {
    contagemCandidata = Math.min(contagemCandidata + 1, JANELA_ESTABILIDADE);
  } else {
    letraCandidata = dados.letra;
    contagemCandidata = 1;
    barraProgressoFill.classList.remove("completa");
    visorScanner.classList.remove("confirmado");
  }
  barraProgressoFill.style.width = `${(contagemCandidata / JANELA_ESTABILIDADE) * 100}%`;

  // Se a letra em vista já é a última confirmada (usuário ainda segurando a
  // mesma pose), não reativa "analisando" — senão o flash de confirmação
  // seria interrompido no frame seguinte, 180ms depois.
  const jaConfirmadaEstaLetra = letraCandidata === letraConfirmada;
  if (!jaConfirmadaEstaLetra) {
    visorScanner.classList.add("analisando");
  }

  if (contagemCandidata >= JANELA_ESTABILIDADE && !jaConfirmadaEstaLetra) {
    letraConfirmada = letraCandidata;
    palavraAtual += letraConfirmada;
    atualizarUiFrase();
    mostrarToastLetra(letraConfirmada);
    marcarLetraPraticada(letraConfirmada);

    // Flash rápido em accent-confirm nos brackets (spec: "faz um flash rápido
    // antes de voltar ao repouso") — some sozinho depois de ~500ms, sem
    // precisar de outro evento pra "desligar" o estado.
    barraProgressoFill.classList.add("completa");
    visorScanner.classList.remove("analisando");
    visorScanner.classList.add("confirmado");
    setTimeout(() => visorScanner.classList.remove("confirmado"), 500);
  }
}

async function iniciarPollingAlfabeto() {
  if (poolingAtivo) return;
  poolingAtivo = true;
  while (poolingAtivo) {
    if (modoAtual === "alfabeto" && !gravandoPalavra) {
      const inicioPoll = performance.now();
      try {
        const frame = capturarFrameBase64();
        const resposta = await fetch("/api/reconhecer-letra", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame }),
        });
        const dados = await resposta.json();
        marcarConexao(true);
        registrarTempoPoll(performance.now() - inicioPoll);
        if (dados.erro) {
          elLetraAtual.textContent = dados.erro;
        } else {
          processarDeteccaoLetra(dados);
        }
      } catch (erro) {
        marcarConexao(false);
      }
    }
    await new Promise((r) => setTimeout(r, INTERVALO_POLL_MS));
  }
}

async function gravarEClassificarPalavra() {
  if (gravandoPalavra) return;
  if (!streamAtual) {
    resultado.textContent = 'Câmera está desligada — clique em "Ligar câmera" primeiro.';
    return;
  }
  gravandoPalavra = true;
  btnGravarPalavra.disabled = true;
  const frames = [];
  const duracaoMs = 2000;
  const intervaloMs = 100; // ~20 frames em 2s
  const inicio = Date.now();

  while (Date.now() - inicio < duracaoMs) {
    frames.push(capturarFrameBase64());
    resultado.textContent = `Gravando... ${((Date.now() - inicio) / 1000).toFixed(1)}s`;
    await new Promise((r) => setTimeout(r, intervaloMs));
  }

  resultado.textContent = "Classificando...";
  try {
    const resposta = await fetch("/api/reconhecer-palavra", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    });
    const dados = await resposta.json();
    if (dados.erro) {
      resultado.textContent = dados.erro;
    } else if (dados.detectado) {
      resultado.textContent = `Palavra: ${dados.palavra} (${(dados.confianca * 100).toFixed(0)}%)`;
      // Uma palavra reconhecida é, por si só, um limite claro — entra direto
      // na frase acumulada e já dispara tradução, sem esperar pausa nenhuma.
      fraseAtual = fraseAtual ? `${fraseAtual} ${dados.palavra}` : dados.palavra;
      atualizarUiFrase();
      ultimaFraseTraduzida = fraseAtual;
      traduzirFrase(fraseAtual);
    } else {
      resultado.textContent = "Não detectei nada — tente de novo mais perto da câmera.";
    }
  } catch (erro) {
    resultado.textContent = "Erro ao consultar o servidor.";
  }

  gravandoPalavra = false;
  btnGravarPalavra.disabled = false;
}

abas.forEach((aba) => {
  aba.addEventListener("click", () => {
    abas.forEach((a) => a.classList.remove("ativa"));
    aba.classList.add("ativa");
    modoAtual = aba.dataset.modo;
    modoAlfabetoEl.hidden = modoAtual !== "alfabeto";
    modoPalavraEl.hidden = modoAtual !== "palavra";
    resultado.textContent = "";
  });
});

if (btnGravarPalavra) btnGravarPalavra.addEventListener("click", gravarEClassificarPalavra);
if (btnApagarLetra) btnApagarLetra.addEventListener("click", apagarUltimaLetra);
if (btnFecharPalavra) btnFecharPalavra.addEventListener("click", fecharPalavraAtual);
if (btnTraduzirAgora) {
  btnTraduzirAgora.addEventListener("click", () => {
    fecharPalavraAtual();
    traduzirFrase(fraseAtual);
    ultimaFraseTraduzida = fraseAtual;
  });
}
if (btnRepetirAudio) {
  btnRepetirAudio.addEventListener("click", () => falar(ultimaTraducaoTexto, "en-US"));
}
if (btnNovaFrase) btnNovaFrase.addEventListener("click", novaFrase);
if (btnCamera) btnCamera.addEventListener("click", alternarCamera);
if (btnTrocarCamera) btnTrocarCamera.addEventListener("click", trocarCameraFrontalTraseira);
if (chkModoDev) {
  chkModoDev.addEventListener("change", () => {
    if (!chkModoDev.checked) {
      painelDev.hidden = true;
      if (telemetriaDev) telemetriaDev.hidden = true;
      if (pistaFacial) pistaFacial.hidden = true;
      ctxEsqueleto.clearRect(0, 0, overlayEsqueleto.width, overlayEsqueleto.height);
      trilhaMao = [];
    }
  });
}
if (elFraseAtual) {
  elFraseAtual.addEventListener("input", () => {
    fraseAtual = elFraseAtual.value;
  });
}

if (btnTema) btnTema.addEventListener("click", alternarTema);
if (btnTelaCheia) btnTelaCheia.addEventListener("click", alternarTelaCheia);
if (btnCopiarFrase) btnCopiarFrase.addEventListener("click", copiarFrase);
if (btnFecharModal) btnFecharModal.addEventListener("click", fecharModalLetra);
if (modalLetra) {
  modalLetra.addEventListener("click", (evento) => {
    if (evento.target === modalLetra) fecharModalLetra();
  });
}
document.querySelectorAll(".guia-item").forEach((item) => {
  item.addEventListener("click", () => abrirModalLetra(item));
});
document.addEventListener("keydown", (evento) => {
  if (evento.key !== "Escape") return;
  if (modalLetra && !modalLetra.hidden) fecharModalLetra();
  else if (painelCamera.classList.contains("tela-cheia")) alternarTelaCheia();
});

aplicarTema(localStorage.getItem("sinalizai-tema"));
iniciarCamera();
