const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const resultado = document.getElementById("resultado");
const btnGravarPalavra = document.getElementById("btn-gravar-palavra");
const abas = document.querySelectorAll(".aba");
const modoAlfabeto = document.getElementById("modo-alfabeto");
const modoPalavra = document.getElementById("modo-palavra");

let modoAtual = "alfabeto";
let poolingAtivo = false;
let gravandoPalavra = false;

const ctx = canvas.getContext("2d");

async function iniciarCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    overlay.textContent = "Câmera pronta";
    setTimeout(() => (overlay.style.display = "none"), 1500);
    iniciarPollingAlfabeto();
  } catch (erro) {
    overlay.textContent = "Não consegui acessar a câmera: " + erro.message;
  }
}

function capturarFrameBase64() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7);
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
        if (dados.erro) {
          resultado.textContent = dados.erro;
        } else if (dados.detectado) {
          resultado.textContent = `Letra: ${dados.letra} (${(dados.confianca * 100).toFixed(0)}%)`;
        } else {
          resultado.textContent = "Nenhuma mão detectada — aproxime-se da câmera.";
        }
      } catch (erro) {
        resultado.textContent = "Erro ao consultar o servidor.";
      }
    }
    await new Promise((r) => setTimeout(r, 400));
  }
}

async function gravarEClassificarPalavra() {
  if (gravandoPalavra) return;
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
    } else {
      resultado.textContent = "Não detectei pose/mãos — tente de novo, corpo mais visível.";
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
    modoAlfabeto.hidden = modoAtual !== "alfabeto";
    modoPalavra.hidden = modoAtual !== "palavra";
    resultado.textContent = "";
  });
});

if (btnGravarPalavra) {
  btnGravarPalavra.addEventListener("click", gravarEClassificarPalavra);
}

iniciarCamera();
