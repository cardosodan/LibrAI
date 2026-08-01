const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const resultado = document.getElementById("resultado");
const btnGravarPalavra = document.getElementById("btn-gravar-palavra");
const abas = document.querySelectorAll(".aba");
const modoAlfabetoEl = document.getElementById("modo-alfabeto");
const modoPalavraEl = document.getElementById("modo-palavra");

const elLetraAtual = document.getElementById("letra-atual");
const elPalavraAtual = document.getElementById("palavra-atual");
const elFraseAtual = document.getElementById("frase-atual");
const elFraseTraduzida = document.getElementById("frase-traduzida");
const btnFecharPalavra = document.getElementById("btn-fechar-palavra");
const btnTraduzirAgora = document.getElementById("btn-traduzir-agora");
const btnNovaFrase = document.getElementById("btn-nova-frase");
const btnCamera = document.getElementById("btn-camera");

let modoAtual = "alfabeto";
let poolingAtivo = false;
let gravandoPalavra = false;
let streamAtual = null;

const ctx = canvas.getContext("2d");

// --- Parâmetros do reconhecimento contínuo (soletração) -------------------
// "Rápida e certa identificação": o polling é rápido (feedback quase
// instantâneo em letra-atual), mas uma letra só é CONFIRMADA (adicionada na
// palavra) depois de aparecer estável por N leituras seguidas — evita que
// uma detecção isolada/ruidosa vire uma letra errada na palavra.
const INTERVALO_POLL_MS = 180;
const JANELA_ESTABILIDADE = 4;      // últimas N leituras consideradas
const CONFIANCA_MINIMA = 0.6;
const PAUSA_PALAVRA_MS = 900;       // sem mão por esse tempo -> fecha a palavra atual
const PAUSA_FRASE_MS = 2500;        // sem mão por esse tempo -> fecha a frase e traduz

let historicoLetras = [];
let letraConfirmada = null;
let palavraAtual = "";
let fraseAtual = "";
let ultimaFraseTraduzida = "";
let semMaoDesde = null;

async function iniciarCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    streamAtual = stream;
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    overlay.style.display = "block";
    overlay.textContent = "Câmera pronta";
    setTimeout(() => (overlay.style.display = "none"), 1500);
    if (btnCamera) {
      btnCamera.textContent = "Parar câmera";
      btnCamera.classList.remove("desligada");
    }
    iniciarPollingAlfabeto();
  } catch (erro) {
    overlay.style.display = "block";
    overlay.textContent = "Não consegui acessar a câmera: " + erro.message;
  }
}

function pararCamera() {
  poolingAtivo = false;
  if (streamAtual) {
    streamAtual.getTracks().forEach((faixa) => faixa.stop());
    streamAtual = null;
  }
  video.srcObject = null;
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  overlay.style.display = "block";
  overlay.textContent = "Câmera desligada";
  if (btnCamera) {
    btnCamera.textContent = "Ligar câmera";
    btnCamera.classList.add("desligada");
  }
}

function alternarCamera() {
  if (streamAtual) {
    pararCamera();
  } else {
    iniciarCamera();
  }
}

function capturarFrameBase64() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7);
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
  elFraseAtual.textContent = fraseAtual || "—";
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
  historicoLetras = [];
  atualizarUiFrase();
}

function novaFrase() {
  palavraAtual = "";
  fraseAtual = "";
  ultimaFraseTraduzida = "";
  letraConfirmada = null;
  historicoLetras = [];
  semMaoDesde = null;
  elLetraAtual.textContent = "—";
  elFraseTraduzida.textContent = "—";
  atualizarUiFrase();
}

function processarDeteccaoLetra(dados) {
  if (!dados.detectado) {
    // Sem mão: qualquer letra "em progresso" perde a estabilidade acumulada,
    // e começamos a contar o tempo de pausa (fecha palavra, depois frase).
    historicoLetras = [];
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
  elLetraAtual.textContent = `${dados.letra} (${(dados.confianca * 100).toFixed(0)}%)`;

  if (dados.confianca < CONFIANCA_MINIMA) return;

  historicoLetras.push(dados.letra);
  if (historicoLetras.length > JANELA_ESTABILIDADE) historicoLetras.shift();

  const estavel =
    historicoLetras.length === JANELA_ESTABILIDADE &&
    historicoLetras.every((l) => l === historicoLetras[0]);

  if (estavel && historicoLetras[0] !== letraConfirmada) {
    letraConfirmada = historicoLetras[0];
    palavraAtual += letraConfirmada;
    atualizarUiFrase();
  }
}

async function iniciarPollingAlfabeto() {
  if (poolingAtivo) return;
  poolingAtivo = true;
  while (poolingAtivo) {
    if (modoAtual === "alfabeto" && !gravandoPalavra && elLetraAtual) {
      try {
        const frame = capturarFrameBase64();
        const resposta = await fetch("/api/reconhecer-letra", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame }),
        });
        const dados = await resposta.json();
        if (dados.erro) {
          elLetraAtual.textContent = dados.erro;
        } else {
          processarDeteccaoLetra(dados);
        }
      } catch (erro) {
        // Falha pontual de rede num poll não deveria travar o loop — só ignora e tenta de novo.
      }
    }
    await new Promise((r) => setTimeout(r, INTERVALO_POLL_MS));
  }
}

async function gravarEClassificarPalavra() {
  if (gravandoPalavra) return;
  if (!streamAtual) {
    resultado.textContent = "Câmera está desligada — clique em \"Ligar câmera\" primeiro.";
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

if (btnGravarPalavra) {
  btnGravarPalavra.addEventListener("click", gravarEClassificarPalavra);
}
if (btnFecharPalavra) {
  btnFecharPalavra.addEventListener("click", fecharPalavraAtual);
}
if (btnTraduzirAgora) {
  btnTraduzirAgora.addEventListener("click", () => {
    fecharPalavraAtual();
    traduzirFrase(fraseAtual);
    ultimaFraseTraduzida = fraseAtual;
  });
}
if (btnNovaFrase) {
  btnNovaFrase.addEventListener("click", novaFrase);
}
if (btnCamera) {
  btnCamera.addEventListener("click", alternarCamera);
}

iniciarCamera();
