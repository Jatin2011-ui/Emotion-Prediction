const EMOJI = { sadness: "😔", joy: "😊", love: "❤️", anger: "😠", fear: "😨", surprise: "😮" };
const COLORS = {
  sadness:  ["#3b82f6", "#1e3a8a"],
  joy:      ["#facc15", "#f97316"],
  love:     ["#ec4899", "#be123c"],
  anger:    ["#ef4444", "#7f1d1d"],
  fear:     ["#8b5cf6", "#312e81"],
  surprise: ["#14b8a6", "#0ea5e9"],
};

const $ = (id) => document.getElementById(id);
const textEl = $("text"), goBtn = $("go"), errEl = $("error"), card = $("card");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- small helpers ---------- */
function setMood(emotion) {
  const [c1, c2] = COLORS[emotion];
  gsap.to(document.documentElement, { "--c1": c1, "--c2": c2, duration: 1.2, ease: "power2.out" });
}

function countTo(el, value, delay = 0) {
  const o = { v: 0 };
  gsap.to(o, {
    v: value, duration: 1.1, delay, ease: "power2.out",
    onUpdate: () => (el.textContent = o.v.toFixed(1)),
  });
}

function setLoading(on) {
  goBtn.disabled = on;
  goBtn.classList.toggle("loading", on);
}

/* ---------- server status ---------- */
async function checkHealth() {
  const s = $("status"), label = s.querySelector("b");
  try {
    const r = await fetch("/health");
    const d = await r.json();
    s.className = "status " + (d.model_loaded ? "ok" : "bad");
    label.textContent = d.model_loaded ? "Model ready" : "Model not loaded";
  } catch {
    s.className = "status bad";
    label.textContent = "Server unreachable";
  }
}

/* ---------- render result ---------- */
function showResult(data) {
  const emotion = data.predicted_emotion;
  const [c1, c2] = COLORS[emotion] || COLORS.joy;
  setMood(emotion);

  $("emoji").textContent = EMOJI[emotion] || "🤔";
  $("emotion").textContent = emotion;
  $("result").hidden = false;

  const rows = Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1]);
  $("bars").innerHTML = rows.map(([label, p], i) => `
    <div class="bar ${i === 0 ? "top" : ""}">
      <span class="name">${EMOJI[label] || ""} ${label}</span>
      <div class="track"><div class="fill" style="background:${i === 0 ? `linear-gradient(90deg,${c1},${c2})` : "rgba(255,255,255,.28)"}"></div></div>
      <span class="pct">0%</span>
    </div>`).join("");

  if (reduceMotion) {
    document.querySelectorAll(".bar").forEach((bar, i) => {
      bar.querySelector(".fill").style.width = rows[i][1] * 100 + "%";
      bar.querySelector(".pct").textContent = (rows[i][1] * 100).toFixed(1) + "%";
    });
    $("conf").textContent = (data.confidence * 100).toFixed(1);
    return;
  }

  gsap.fromTo("#emoji", { scale: 0, rotate: -35 }, { scale: 1, rotate: 0, duration: 1, ease: "elastic.out(1, 0.5)" });
  gsap.fromTo("#emotion", { y: 16, opacity: 0 }, { y: 0, opacity: 1, duration: 0.6, delay: 0.15 });
  countTo($("conf"), data.confidence * 100);

  document.querySelectorAll(".bar").forEach((bar, i) => {
    const p = rows[i][1] * 100, delay = 0.25 + i * 0.08;
    gsap.to(bar.querySelector(".fill"), { width: p + "%", duration: 1.1, delay, ease: "power3.out" });
    const pct = bar.querySelector(".pct"), o = { v: 0 };
    gsap.to(o, { v: p, duration: 1.1, delay, ease: "power3.out", onUpdate: () => (pct.textContent = o.v.toFixed(1) + "%") });
  });

  // emotion-specific reactions
  if ((emotion === "joy" || emotion === "love") && typeof confetti === "function") {
    confetti({ particleCount: 130, spread: 85, origin: { y: 0.65 }, colors: [c1, c2, "#ffffff"] });
  }
  if (emotion === "anger" || emotion === "fear") {
    gsap.to(card, { keyframes: { x: [-10, 10, -7, 7, -3, 0], easeEach: "none" }, duration: 0.5 });
  }
  if (emotion === "surprise") {
    gsap.fromTo(card, { scale: 1 }, { scale: 1.025, yoyo: true, repeat: 1, duration: 0.18 });
  }
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------- predict ---------- */
async function analyze() {
  const text = textEl.value.trim();
  errEl.textContent = "";
  if (!text) {
    errEl.textContent = "Type a sentence first.";
    textEl.focus();
    return;
  }

  setLoading(true);
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const d = err.detail;
      throw new Error(typeof d === "string" ? d : (d && d[0] && d[0].msg) || "The server rejected the request.");
    }
    showResult(await res.json());
  } catch (e) {
    errEl.textContent = e.message || "Could not reach the server.";
  } finally {
    setLoading(false);
  }
}

/* ---------- events ---------- */
goBtn.addEventListener("click", analyze);
textEl.addEventListener("input", () => ($("count").textContent = `${textEl.value.length} / 2000`));
textEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
});
$("chips").addEventListener("click", (e) => {
  const t = e.target.closest("button")?.dataset.t;
  if (!t) return;
  textEl.value = t;
  textEl.dispatchEvent(new Event("input"));
  analyze();
});

/* One orchestrated entrance */
if (!reduceMotion) {
  gsap.from(card, { y: 40, opacity: 0, duration: 0.9, ease: "power3.out" });
}
checkHealth();
