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

// --- Esqueleto da mão (conexões padrão MediaPipe Hands, 21 pontos) --------
const CONEXOES_MAO = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // polegar
  [0, 5], [5, 6], [6, 7], [7, 8],       // indicador
  [0, 9], [9, 10], [10, 11], [11, 12],  // médio
  [0, 13], [13, 14], [14, 15], [15, 16], // anelar
  [0, 17], [17, 18], [18, 19], [19, 20], // mindinho
  [5, 9], [9, 13], [13, 17],             // palma
];

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

function desenharEsqueleto(pontos) {
  ctxEsqueleto.clearRect(0, 0, overlayEsqueleto.width, overlayEsqueleto.height);
  if (!pontos) return;
  const w = overlayEsqueleto.width;
  const h = overlayEsqueleto.height;

  ctxEsqueleto.strokeStyle = "rgba(108, 140, 255, 0.9)";
  ctxEsqueleto.lineWidth = 2;
  CONEXOES_MAO.forEach(([a, b]) => {
    ctxEsqueleto.beginPath();
    ctxEsqueleto.moveTo(pontos[a][0] * w, pontos[a][1] * h);
    ctxEsqueleto.lineTo(pontos[b][0] * w, pontos[b][1] * h);
    ctxEsqueleto.stroke();
  });

  ctxEsqueleto.fillStyle = "#3ddc84";
  pontos.forEach(([x, y]) => {
    ctxEsqueleto.beginPath();
    ctxEsqueleto.arc(x * w, y * h, 4, 0, 2 * Math.PI);
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
      try {
        const frame = capturarFrameBase64();
        const resposta = await fetch("/api/reconhecer-letra", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame }),
        });
        const dados = await resposta.json();
        marcarConexao(true);
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
    if (!chkModoDev.checked) painelDev.hidden = true;
  });
}
if (elFraseAtual) {
  elFraseAtual.addEventListener("input", () => {
    fraseAtual = elFraseAtual.value;
  });
}

iniciarCamera();
