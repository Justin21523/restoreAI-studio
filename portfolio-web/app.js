const scenarios = {
  archive: {
    title: "Archive photo restore",
    subject: "Taipei street archive",
    palette: ["#212936", "#7dd3fc", "#f9a8d4", "#facc15"],
    latency: "1.8s",
  },
  portrait: {
    title: "Portrait face cleanup",
    subject: "Studio portrait",
    palette: ["#1f2937", "#f7c59f", "#8ecae6", "#fb7185"],
    latency: "1.5s",
  },
  product: {
    title: "Product detail upscale",
    subject: "E-commerce product crop",
    palette: ["#111827", "#c4b5fd", "#67e8f9", "#86efac"],
    latency: "1.2s",
  },
};

const $ = (id) => document.getElementById(id);
const beforeCanvas = $("before-canvas");
const afterCanvas = $("after-canvas");
const beforeCtx = beforeCanvas.getContext("2d");
const afterCtx = afterCanvas.getContext("2d");

let uploadedImage = null;
let currentScenario = "archive";

function drawGrid(ctx, width, height, palette, rough = true) {
  ctx.fillStyle = palette[0];
  ctx.fillRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, palette[1]);
  gradient.addColorStop(0.55, palette[2]);
  gradient.addColorStop(1, palette[3]);
  ctx.globalAlpha = rough ? 0.22 : 0.34;
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  ctx.globalAlpha = 1;

  for (let y = 36; y < height; y += 52) {
    for (let x = 28; x < width; x += 70) {
      const hue = (x + y) % palette.length;
      ctx.fillStyle = palette[hue];
      ctx.globalAlpha = rough ? 0.32 : 0.54;
      ctx.fillRect(x, y, 42 + ((x + y) % 38), 24 + (y % 34));
    }
  }
  ctx.globalAlpha = 1;
}

function drawScene(ctx, scenarioKey, enhanced) {
  const scenario = scenarios[scenarioKey];
  const { width, height } = ctx.canvas;
  drawGrid(ctx, width, height, scenario.palette, !enhanced);

  ctx.save();
  ctx.translate(width * 0.5, height * 0.52);

  if (scenarioKey === "portrait") {
    ctx.fillStyle = enhanced ? "#ffd7ba" : "#c78b72";
    ctx.beginPath();
    ctx.arc(0, -36, 86, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = enhanced ? "#0f172a" : "#1f2937";
    ctx.beginPath();
    ctx.arc(-30, -48, 8, 0, Math.PI * 2);
    ctx.arc(30, -48, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = enhanced ? "#9f1239" : "#7f1d1d";
    ctx.lineWidth = enhanced ? 5 : 3;
    ctx.beginPath();
    ctx.arc(0, -22, 32, 0.1, Math.PI - 0.1);
    ctx.stroke();
    ctx.fillStyle = enhanced ? "#38bdf8" : "#2563eb";
    ctx.fillRect(-112, 72, 224, 124);
  } else if (scenarioKey === "product") {
    ctx.rotate(-0.12);
    ctx.fillStyle = enhanced ? "#e5e7eb" : "#94a3b8";
    ctx.fillRect(-145, -100, 290, 210);
    ctx.fillStyle = enhanced ? "#22d3ee" : "#0891b2";
    ctx.fillRect(-100, -54, 200, 116);
    ctx.fillStyle = enhanced ? "#f8fafc" : "#cbd5e1";
    ctx.fillRect(-68, -22, 136, 52);
  } else {
    ctx.fillStyle = enhanced ? "#e2e8f0" : "#94a3b8";
    ctx.fillRect(-240, -10, 480, 170);
    ctx.fillStyle = enhanced ? "#0f172a" : "#1e293b";
    for (let i = -210; i <= 210; i += 70) ctx.fillRect(i, 20, 38, 94);
    ctx.fillStyle = enhanced ? "#fde68a" : "#b45309";
    ctx.fillRect(-260, 132, 520, 28);
  }
  ctx.restore();

  if (!enhanced) addNoise(ctx, 0.34);
  if (enhanced) sharpenOverlay(ctx);

  ctx.fillStyle = "rgba(8, 13, 20, 0.72)";
  ctx.fillRect(20, height - 74, 310, 50);
  ctx.fillStyle = "#f8fafc";
  ctx.font = "700 18px system-ui";
  ctx.fillText(enhanced ? "Restored output" : "Original input", 38, height - 44);
  ctx.fillStyle = "#aab4c2";
  ctx.font = "13px system-ui";
  ctx.fillText(scenario.subject, 38, height - 25);
}

function addNoise(ctx, amount) {
  const { width, height } = ctx.canvas;
  const image = ctx.getImageData(0, 0, width, height);
  for (let i = 0; i < image.data.length; i += 4) {
    const noise = (Math.random() - 0.5) * 255 * amount;
    image.data[i] += noise;
    image.data[i + 1] += noise;
    image.data[i + 2] += noise;
  }
  ctx.putImageData(image, 0, 0);
}

function sharpenOverlay(ctx) {
  const { width, height } = ctx.canvas;
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 24) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x + 80, height);
    ctx.stroke();
  }
}

function drawUploaded(enhanced) {
  const ctx = enhanced ? afterCtx : beforeCtx;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  const ratio = Math.min(
    ctx.canvas.width / uploadedImage.width,
    ctx.canvas.height / uploadedImage.height,
  );
  const width = uploadedImage.width * ratio;
  const height = uploadedImage.height * ratio;
  const x = (ctx.canvas.width - width) / 2;
  const y = (ctx.canvas.height - height) / 2;

  if (!enhanced) {
    ctx.filter = "blur(1px) contrast(0.82) saturate(0.72)";
  } else {
    const strength = Number($("strength").value);
    ctx.filter = `contrast(${1.05 + strength / 260}) saturate(${1.08 + strength / 360})`;
  }
  ctx.drawImage(uploadedImage, x, y, width, height);
  ctx.filter = "none";
  if (!enhanced) addNoise(ctx, 0.12);
  if (enhanced) sharpenOverlay(ctx);
}

function updateCompare() {
  const value = Number($("compare-slider").value);
  $("after-canvas").style.clipPath = `inset(0 ${100 - value}% 0 0)`;
  $("split-handle").style.left = `${value}%`;
}

function render() {
  currentScenario = $("scenario").value;
  $("asset-title").textContent = uploadedImage ? "Uploaded image restore" : scenarios[currentScenario].title;
  if (uploadedImage) {
    drawUploaded(false);
    drawUploaded(true);
  } else {
    drawScene(beforeCtx, currentScenario, false);
    drawScene(afterCtx, currentScenario, true);
  }
  updateMetrics();
  updatePayload();
  updateCompare();
}

function updateMetrics() {
  const scale = Number($("scale").value);
  const sourceWidth = uploadedImage ? uploadedImage.width : 720;
  $("metric-resolution").textContent = `${sourceWidth} -> ${sourceWidth * scale} px`;
  $("metric-latency").textContent = scenarios[currentScenario].latency;
}

function updatePayload(status = "ready") {
  const payload = {
    mode: "mock-safe",
    scenario: uploadedImage ? "uploaded-image" : currentScenario,
    scale: Number($("scale").value),
    strength: Number($("strength").value) / 100,
    device: "browser-canvas",
    status,
    artifact: "demo-output.png",
  };
  $("debug-json").textContent = JSON.stringify(payload, null, 2);
}

function setPipeline(stepIndex) {
  [...$("pipeline-list").children].forEach((item, index) => {
    item.classList.toggle("done", index < stepIndex);
    item.classList.toggle("active", index === stepIndex);
  });
}

async function runDemoFlow() {
  $("job-status").textContent = "running";
  $("job-status").classList.add("running");
  updatePayload("running");
  for (let i = 0; i < 5; i += 1) {
    setPipeline(i);
    await new Promise((resolve) => setTimeout(resolve, 420));
  }
  setPipeline(5);
  $("job-status").textContent = "complete";
  $("job-status").classList.remove("running");
  updatePayload("complete");
}

$("scenario").addEventListener("change", () => {
  uploadedImage = null;
  render();
});
$("scale").addEventListener("change", render);
$("strength").addEventListener("input", render);
$("compare-slider").addEventListener("input", updateCompare);
$("load-sample").addEventListener("click", () => {
  uploadedImage = null;
  render();
});
$("run-demo").addEventListener("click", runDemoFlow);
$("upload").addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = () => {
      uploadedImage = image;
      render();
    };
    image.src = reader.result;
  };
  reader.readAsDataURL(file);
});

render();
