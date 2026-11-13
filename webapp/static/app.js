"use strict";

const SCENARIOS = [
  { title: "Xét nghiệm đạt ngưỡng", note: "Nam 58 tuổi, HbA1c 7,5%", text: "Nam 58 tuổi, BMI 33,5, tăng huyết áp, HbA1c 7,5%, đường huyết 180 mg/dL. Nguy cơ của tôi thế nào?" },
  { title: "Mô hình và Rule mâu thuẫn", note: "HbA1c 6,6% nhưng xác suất mô hình thấp", text: "Nam 35 tuổi, BMI 24, HbA1c 6,6%, đường huyết 100 mg/dL, không hút thuốc" },
  { title: "Chưa có xét nghiệm", note: "Agent yêu cầu thêm dữ liệu, có lượt tiếp theo", text: "Nam 50 tuổi, BMI 29, tăng huyết áp, đang hút thuốc. Tôi có nguy cơ tiểu đường không?", next: "Tôi vừa có kết quả: HbA1c 5,4%, đường huyết lúc đói 96 mg/dL" },
  { title: "Hỏi kiến thức", note: "Tôi nên làm gì để phòng tránh?", text: "Tôi nên làm gì để phòng tránh bệnh tiểu đường?" },
  { title: "Ranh giới về thuốc", note: "Hệ thống từ chối kê đơn và tính liều", text: "Kê cho tôi metformin liều bao nhiêu mg một ngày?" },
  { title: "Hạ đường huyết", note: "Cảnh báo khi đường huyết 55 mg/dL", text: "Tôi run, vã mồ hôi, đường huyết đo được 55 mg/dL thì làm sao?" },
  { title: "Chuẩn bị đi khám", note: "Support Agent lập danh sách", text: "Giúp tôi chuẩn bị đi khám tiểu đường, cần mang gì và hỏi bác sĩ gì?" },
];
const AGENT = { intake: "Intake", safety: "Safety", prediction: "Prediction", policy: "Policy", knowledge: "Knowledge", advisor: "Advisor", support: "Support" };
const STEP = {
  PARSE: "đọc câu hỏi, cập nhật hồ sơ", PRECHECK: "kiểm tra an toàn đầu vào", PREDICT: "chạy DiaNet và RiskNet", DECIDE: "RL đề xuất, Rule kiểm tra",
  RETRIEVE: "truy xuất KB và đồ thị tri thức", COMPOSE: "soạn câu trả lời có trích dẫn", CHECKLIST: "lập danh sách chuẩn bị", REVIEW: "Critic duyệt bản nháp",
};
const INTENT = { assess: "Đánh giá nguy cơ", knowledge: "Hỏi kiến thức", support: "Trợ giúp đi khám", out_of_scope: "Ngoài phạm vi" };
const SAFETY = {
  ok: ["An toàn", "ok"], boundary: ["Ranh giới thuốc", "warn"], urgent: ["Cần xử trí ngay", "danger"], emergency: ["Khẩn cấp", "danger"],
};
const LIMITS = { age: [1, 120, "Tuổi"], bmi: [10, 70, "BMI"], hba1c: [3, 20, "HbA1c"], glucose: [20, 800, "Đường huyết"] };

const $ = (id) => document.getElementById(id);
const messages = $("messages"), form = $("profile-form"), question = $("question"), note = $("form-note");
const state = { session: newId(), turns: [], selected: -1, busy: false, dirty: false, resultsLoaded: false };

function newId() {
  return window.crypto && crypto.randomUUID ? crypto.randomUUID() : `s-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function num(value, digits = 1) {
  return Number(value).toFixed(digits).replace(".", ",");
}
function safeUrl(url) {
  return /^https?:\/\//i.test(url || "") ? esc(url) : "";
}

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Máy chủ báo lỗi ${res.status}`);
  }
  return res.json();
}

/* Câu trả lời: nội dung được escape trước, chỉ dựng lại in đậm, danh sách và trích dẫn */
function inline(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+)\]/g, '<button type="button" class="cite" data-cite="$1" aria-label="Xem nguồn $1">$1</button>');
}
function renderAnswer(text) {
  const html = [];
  for (const block of text.split(/\n\n+/)) {
    const lines = block.split("\n");
    let list = [];
    const flush = () => { if (list.length) { html.push(`<ul>${list.join("")}</ul>`); list = []; } };
    for (const line of lines) {
      if (line.startsWith("- ")) list.push(`<li>${inline(line.slice(2))}</li>`);
      else { flush(); html.push(`<p>${inline(line)}</p>`); }
    }
    flush();
  }
  return html.join("");
}

/* Hội thoại */
function renderWelcome() {
  messages.innerHTML = `
    <div class="welcome">
      <h1>Tư vấn về bệnh tiểu đường</h1>
      <p>Nhập chỉ số ở phần Hồ sơ hoặc gõ thẳng vào ô chat, ví dụ: Nữ 45 tuổi, BMI 26, HbA1c 6,1%. Mỗi câu trả lời đi qua các agent dự đoán, tra cứu và kiểm duyệt, kèm nguồn trích dẫn.</p>
      <h2>Tình huống mẫu</h2>
      <div class="scenarios">
        ${SCENARIOS.map((s, i) => `<button type="button" class="scenario" data-scenario="${i}"><b>${esc(s.title)}</b><span>${esc(s.note)}</span></button>`).join("")}
      </div>
    </div>`;
  $("chat-scroll").scrollTop = 0;
}
function scrollEnd() {
  const box = $("chat-scroll");
  box.scrollTop = box.scrollHeight;
}
function addUser(text) {
  messages.querySelector(".welcome")?.remove();
  const el = document.createElement("div");
  el.className = "msg-user";
  el.textContent = text;
  messages.append(el);
}
function addBot(turn, index) {
  const level = turn.safety.level;
  const el = document.createElement("article");
  el.className = `msg-bot ${level === "emergency" || level === "urgent" ? level : ""}`;
  el.dataset.turn = index;
  const agents = new Set(turn.path).size;
  const sources = turn.evidence.length;
  el.innerHTML = `
    <div class="answer">${renderAnswer(turn.answer)}</div>
    <div class="msg-foot">
      <span>${agents} agent${sources ? ` · ${sources} nguồn` : ""} · ${Math.round(turn.latency_ms)} ms</span>
      <button type="button" class="link-btn" data-open-turn="${index}">Xem phân tích</button>
    </div>`;
  messages.append(el);
}
function setBusy(on) {
  state.busy = on;
  document.querySelector(".send").disabled = on;
  messages.querySelector(".thinking")?.remove();
  if (on) {
    const el = document.createElement("div");
    el.className = "thinking";
    el.innerHTML = "<i></i> Các agent đang xử lý…";
    messages.append(el);
    scrollEnd();
  }
}
async function send(text, followUp = null) {
  text = text.trim();
  if (!text || state.busy) return;
  $("followup").hidden = true;
  addUser(text);
  setBusy(true);
  try {
    const turn = await api("/api/chat", { method: "POST", body: JSON.stringify({ session_id: state.session, text }) });
    setBusy(false);
    state.turns.push(turn);
    const index = state.turns.length - 1;
    addBot(turn, index);
    selectTurn(index);
    syncProfile(turn.profile);
    if (followUp) showFollowUp(followUp);
  } catch (err) {
    setBusy(false);
    const el = document.createElement("div");
    el.className = "msg-error";
    el.textContent = `Chưa gửi được câu hỏi: ${err.message === "Failed to fetch" ? "không kết nối được máy chủ, hãy kiểm tra cửa sổ start_webapp" : err.message}.`;
    messages.append(el);
  }
  scrollEnd();
}
function showFollowUp(text) {
  const box = $("followup");
  box.innerHTML = "<span>Lượt tiếp theo:</span>";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = text;
  btn.addEventListener("click", () => send(text));
  box.append(btn);
  box.hidden = false;
}

/* Bảng phân tích */
function selectTurn(index) {
  state.selected = index;
  messages.querySelectorAll(".msg-bot").forEach((el) => el.classList.toggle("selected", Number(el.dataset.turn) === index));
  renderInspector(state.turns[index]);
}
function meter(label, value, display, percent, tick, noteText) {
  return `<div class="meter">
    <div class="meter-head"><span>${label}</span><b>${esc(display)}</b></div>
    <div class="meter-track" role="img" aria-label="${esc(label)}: ${esc(display)}">
      <div class="meter-fill" style="width:${Math.min(100, Math.max(0, percent))}%"></div>
      ${tick === null ? "" : `<div class="meter-tick" style="left:${tick}%"></div>`}
    </div>
    <div class="meter-note">${esc(noteText)}</div>
  </div>`;
}
function predictionSection(pr) {
  const drivers = pr.drivers.map((d) => `<li>${esc(d.name)}: <b>${d.delta >= 0 ? "+" : "−"}${num(Math.abs(d.delta) * 100)} điểm %</b></li>`).join("");
  const whatIf = pr.what_if.map((d) => `<li>Nếu ${esc(d.name)}: <b>${d.delta >= 0 ? "+" : "−"}${num(Math.abs(d.delta))} điểm</b></li>`).join("");
  return `<section class="section"><h3>Deep Learning</h3>
    ${meter("DiaNet · xác suất", pr.p, pr.p_text, pr.p * 100, pr.threshold * 100,
      `${pr.scenario}. Vạch đen là ngưỡng phân loại ${num(pr.threshold * 100)}%, ${pr.above ? "đang trên ngưỡng" : "đang dưới ngưỡng"}.`)}
    ${meter("RiskNet · điểm nguy cơ", pr.score, `${num(pr.score)}/100`, pr.score, null, `Mức ${pr.tier}. Đây không phải xác suất mắc bệnh.`)}
    ${drivers ? `<ul class="plain-list">${drivers}</ul>` : ""}
    ${whatIf ? `<ul class="plain-list" style="margin-top:6px">${whatIf}</ul>` : ""}
  </section>`;
}
function decisionSection(de) {
  const labs = de.lab_checks.map((c) => `<li><b>${esc(c.test)} ${num(c.value, c.unit === "%" ? 1 : 0)}${c.unit === "%" ? "" : " "}${esc(c.unit)}</b>: ${esc(c.category)}${c.diagnostic ? ' <span class="chip danger">đạt ngưỡng</span>' : ""}</li>`).join("");
  return `<section class="section"><h3>RL + Rule</h3>
    <div class="decision">
      <div class="decision-row"><span>RL đề xuất</span><div>${esc(de.rl_label)}</div></div>
      <div class="decision-row ${de.rule ? "override" : ""}"><span>Rule</span><div>${de.rule ? `${esc(de.rule)}: ${esc(de.rule_label)}` : "Không can thiệp"}</div></div>
      <div class="decision-row final"><span>Quyết định</span><div><b>${esc(de.action_label)}</b></div></div>
    </div>
    ${labs ? `<ul class="plain-list" style="margin-top:8px">${labs}</ul>` : ""}
  </section>`;
}
function flowSection(turn) {
  const rows = turn.trace.map((t) => {
    const failed = t.status !== "ok";
    return `<li class="${failed ? "failed" : ""}"><div><b>${esc(AGENT[t.agent] || t.agent)}</b><small>${esc(failed ? t.status : STEP[t.kind] || t.kind)}</small></div><time>${num(t.ms)} ms</time></li>`;
  }).join("");
  return `<section class="section"><h3>Luồng agent</h3><ol class="flow" style="grid-column:1/-1">${rows}</ol>
    <div class="chips">${turn.tools.map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div></section>`;
}
function factsSection(turn) {
  if (!turn.facts.length) return "";
  const rows = turn.facts.map((f) => f.raw ? `<div class="triple">${esc(f.raw)}</div>`
    : `<div class="triple">${esc(f.subject)} <em>${esc(f.relation)}</em> ${esc(f.object)}</div>`).join("");
  return `<section class="section"><h3>Đồ thị tri thức</h3>${rows}</section>`;
}
function evidenceSection(turn) {
  if (!turn.evidence.length) return "";
  const rows = turn.evidence.map((e) => {
    const url = safeUrl(e.url);
    const origin = esc(e.publisher) + (e.locator ? `, ${esc(e.locator)}` : "");
    return `<div class="source" id="source-${e.n}">
      <div class="source-title">[${e.n}] ${esc(e.title)}</div>
      <div class="source-meta">${e.source_type === "book" ? "Sách" : "Web"} · ${url ? `<a href="${url}" target="_blank" rel="noreferrer">${origin}</a>` : origin} · ${esc(e.doc_id)}</div>
      <p>${esc(e.text)}</p>
    </div>`;
  }).join("");
  return `<section class="section"><h3>Bằng chứng</h3>${rows}</section>`;
}
function renderInspector(turn) {
  const body = $("inspector-body");
  if (!turn) {
    body.innerHTML = '<p class="empty-note">Sau mỗi câu trả lời, phần này hiện dự đoán, quyết định RL + Rule, luồng agent và nguồn trích dẫn.</p>';
    return;
  }
  const [safetyLabel, safetyClass] = SAFETY[turn.safety.level] || [turn.safety.level, ""];
  const warnings = [
    turn.expanded ? '<span class="chip">Tự mở rộng truy vấn bằng KG</span>' : "",
    turn.blocked ? '<span class="chip danger">Critic đã chặn bản nháp</span>' : "",
    turn.stale.length ? `<span class="chip warn">Nguồn cũ: ${esc(turn.stale.join(", "))}</span>` : "",
  ].join("");
  body.innerHTML = `
    <section class="section">
      <div class="chips">
        <span class="chip ${safetyClass}">${esc(safetyLabel)}</span>
        <span class="chip">${esc(INTENT[turn.intent] || turn.intent)}</span>${warnings}
      </div>
    </section>
    ${turn.prediction ? predictionSection(turn.prediction) : ""}
    ${turn.decision ? decisionSection(turn.decision) : ""}
    ${flowSection(turn)}
    ${factsSection(turn)}
    ${evidenceSection(turn)}
    <section class="section"><h3>Đánh giá nhanh</h3>
      <div class="stats">
        <div><span>Tin nhắn</span><b>${turn.messages}</b></div>
        <div><span>Vòng Critic</span><b>${turn.critic_rounds}</b></div>
        <div><span>Lỗi</span><b>${turn.failures}</b></div>
        <div><span>Thời gian</span><b>${Math.round(turn.latency_ms)} ms</b></div>
      </div>
    </section>`;
  body.closest(".inspector").scrollTop = 0;
}

/* Ngăn trượt ở màn hình hẹp */
function openDrawer(id) {
  $(id).classList.add("open");
  $("scrim").hidden = false;
}
function closeDrawers() {
  document.querySelectorAll(".panel.open").forEach((el) => el.classList.remove("open"));
  $("scrim").hidden = true;
}
const inspectorIsDrawer = () => window.matchMedia("(max-width: 1240px)").matches;

/* Hồ sơ */
function parseNumber(name) {
  const raw = form.elements[name].value.trim().replace(",", ".");
  if (!raw) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : NaN;
}
function setNote(text, kind = "") {
  note.textContent = text;
  note.className = `form-note ${kind}`;
}
function readProfile() {
  const data = {}, errors = [];
  for (const name of Object.keys(LIMITS)) {
    const [lo, hi, label] = LIMITS[name];
    const value = parseNumber(name);
    const bad = value !== null && !(value >= lo && value <= hi);
    form.elements[name].setAttribute("aria-invalid", bad ? "true" : "false");
    if (bad) errors.push(`${label} phải từ ${lo} đến ${hi}`);
    else if (value !== null) data[name] = value;
  }
  const f = form.elements;
  if (f.gender.value) data.gender = f.gender.value;
  if (f.smoking.value) data.smoking = f.smoking.value;
  if ("glucose" in data) data.glucose_kind = f.fasting.checked ? "fasting" : "unknown";
  data.hypertension = f.hypertension.checked ? 1 : 0;
  data.heart_disease = f.heart_disease.checked ? 1 : 0;
  data.family_history = f.family_history.checked ? 1 : 0;
  if (f.low_activity.checked) data.activity = "Low";
  return { data, errors };
}
async function saveProfile() {
  const { data, errors } = readProfile();
  if (errors.length) {
    setNote(`${errors.join("; ")}.`, "error");
    return false;
  }
  try {
    await api("/api/profile", { method: "POST", body: JSON.stringify({ session_id: state.session, profile: data }) });
    state.dirty = false;
    setNote("Đã lưu, các agent sẽ dùng hồ sơ này.", "ok");
    return true;
  } catch (err) {
    setNote(`Chưa lưu được: ${err.message}.`, "error");
    return false;
  }
}
function syncProfile(profile) {
  if (state.dirty) return; // không ghi đè phần người dùng đang sửa dở
  const f = form.elements, changed = [];
  const setText = (name, label, value) => {
    const text = value === undefined ? "" : String(value).replace(".", ",");
    if (f[name].value !== text) { f[name].value = text; if (text) changed.push(label); }
  };
  setText("age", "tuổi", profile.age);
  setText("hba1c", "HbA1c", profile.hba1c);
  setText("glucose", "đường huyết", profile.glucose);
  if (!(f.bmi.value && Number(f.bmi.value.replace(",", ".")) === profile.bmi)) setText("bmi", "BMI", profile.bmi);
  f.gender.value = profile.gender || "";
  f.smoking.value = profile.smoking || "";
  f.fasting.checked = profile.glucose_kind === "fasting";
  f.hypertension.checked = profile.hypertension === 1;
  f.heart_disease.checked = profile.heart_disease === 1;
  f.family_history.checked = profile.family_history === 1;
  f.low_activity.checked = profile.activity === "Low";
  if (changed.length) setNote(`Đã cập nhật từ hội thoại: ${changed.join(", ")}.`);
}
function clearForm() {
  form.reset();
  state.dirty = false;
  form.querySelectorAll("[aria-invalid]").forEach((el) => el.setAttribute("aria-invalid", "false"));
  setNote("");
}
function updateBmi() {
  const h = parseNumber("height"), w = parseNumber("weight");
  if (h > 0 && w > 0) form.elements.bmi.value = num(w / (h / 100) ** 2);
}

async function newSession() {
  api(`/api/session/${state.session}`, { method: "DELETE" }).catch(() => {});
  state.session = newId();
  state.turns = [];
  state.selected = -1;
  $("followup").hidden = true;
  clearForm();
  renderWelcome();
  renderInspector(null);
  closeDrawers();
  question.focus();
}

/* Trang kết quả */
const pct1 = (v) => `${num(v * 100)}%`;
function table(headers, rows, highlight = () => false) {
  return `<div class="table-wrap"><table>
    <thead><tr>${headers.map((h) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r, i) => `<tr class="${highlight(r, i) ? "hl" : ""}">${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div>`;
}
async function loadResults() {
  if (state.resultsLoaded) return;
  const box = $("results-body");
  box.innerHTML = '<p class="empty-note">Đang tải…</p>';
  try {
    const r = await api("/api/results");
    const full = r.system.find((s) => s.config.startsWith("B2"));
    const single = r.system.find((s) => s.config.startsWith("B0"));
    const both = r.dianet.find((d) => d["Trường hợp"].startsWith("Đủ"));
    const hybrid = r.retrieval.find((d) => d["Chế độ"] === "hybrid");
    const kb = Object.fromEntries(r.kb.map((k) => [k["Unnamed: 0"], k["Số lượng"]]));
    box.innerHTML = `
      <div>
        <h1>Kết quả thực nghiệm</h1>
        <p class="results-lead">Số liệu đọc trực tiếp từ thư mục results, do notebook sinh ra. Mức phần thưởng RL là giả định thiết kế của đồ án, không phải chi phí y tế thực.</p>
      </div>
      <div class="tiles">
        <div class="tile"><b>${pct1(full["Thành công"])}</b><span>kịch bản đạt với B2, so với ${pct1(single["Thành công"])} của một agent</span></div>
        <div class="tile"><b>${full["Vi phạm"]}</b><span>vi phạm an toàn của B2, so với ${single["Vi phạm"]} của một agent</span></div>
        <div class="tile"><b>${num(both["PR-AUC"], 3)}</b><span>PR-AUC của DiaNet khi đủ hai xét nghiệm</span></div>
        <div class="tile"><b>${num(hybrid["Recall@k"], 3)}</b><span>Recall@5 của truy xuất lai</span></div>
        <div class="tile"><b>${r.tests.passed}/${r.tests.total}</b><span>kiểm thử đạt, gồm tiêm lỗi</span></div>
      </div>
      <div class="card">
        <h2>So sánh cấu hình hệ thống</h2>
        <p>B2 là hệ đầy đủ. B3 bỏ Safety Agent, B4 bỏ Rule shield.</p>
        ${table(["Cấu hình", "Thành công", "Vi phạm", "Tin nhắn TB", "Công cụ TB", "Trễ trung vị"],
          r.system.map((s) => [s.config, pct1(s["Thành công"]), s["Vi phạm"], num(s["Tin nhắn TB"]), num(s["Công cụ TB"]), `${num(s["Trễ trung vị (ms)"])} ms`]),
          (row) => row[0].startsWith("B2"))}
      </div>
      <div class="card">
        <h2>Chính sách RL trên tập test</h2>
        <p>Vi phạm luật là số ca bỏ qua ngưỡng chẩn đoán hoặc không xét nghiệm người từ 35 tuổi.</p>
        ${table(["Chính sách", "Reward TB", "Sensitivity", "Specificity", "Xét nghiệm/người", "Bỏ sót", "Vi phạm luật"],
          r.policy.map((p) => [p["Chính sách"], num(p["Reward TB"], 3), num(p.Sensitivity, 3), num(p.Specificity, 3), num(p["Xét nghiệm/người"], 2), Math.round(p["Bỏ sót"]), Math.round(p["Vi phạm luật"])]),
          (row) => row[0] === "RL + Rule")}
      </div>
      <div class="two-col">
        <div class="card">
          <h2>DiaNet theo thông tin có sẵn</h2>
          <p>Một mô hình xử lý cả bốn trường hợp nhờ đầu vào mask.</p>
          ${table(["Trường hợp", "PR-AUC", "F1", "Recall"], r.dianet.map((d) => [d["Trường hợp"], num(d["PR-AUC"], 3), num(d.F1, 3), num(d.Recall, 3)]))}
        </div>
        <div class="card">
          <h2>Truy xuất với k = 5</h2>
          <p>Trên ${kb["Ghi chú (passage)"]} ghi chú, ${kb["Thực thể"]} thực thể và ${kb["Quan hệ"]} quan hệ.</p>
          ${table(["Chế độ", "Recall@5", "Hit@5", "MRR"], r.retrieval.map((d) => [d["Chế độ"], num(d["Recall@k"], 3), num(d["Hit@k"], 3), num(d.MRR, 3)]), (row) => row[0] === "hybrid")}
        </div>
      </div>
      <div class="card">
        <h2>Kiến trúc hệ thống</h2>
        <p>Coordinator điều phối các agent qua tin nhắn có schema.</p>
        <img src="/figures/fig_architecture.png" alt="Sơ đồ kiến trúc: DB, Deep Learning, KB và RAG, RL và Rule, các agent, Coordinator" loading="lazy">
      </div>
      <div class="card">
        <h2>Đồ thị tri thức</h2>
        <p>Một phần đồ thị quanh các thực thể trung tâm.</p>
        <img src="/figures/fig_kg.png" alt="Đồ thị tri thức quanh đái tháo đường, HbA1c, hút thuốc và hoạt động thể lực" loading="lazy">
      </div>`;
    state.resultsLoaded = true;
  } catch (err) {
    box.innerHTML = `<div class="msg-error">Chưa tải được số liệu: ${esc(err.message)}.</div>`;
  }
}
function showTab(name) {
  const chat = name === "chat";
  $("tab-chat").setAttribute("aria-selected", chat);
  $("tab-results").setAttribute("aria-selected", !chat);
  $("view-chat").hidden = !chat;
  $("view-results").hidden = chat;
  $("open-profile").style.visibility = chat ? "" : "hidden";
  closeDrawers();
  if (!chat) loadResults();
}

/* Sự kiện */
$("tab-chat").addEventListener("click", () => showTab("chat"));
$("tab-results").addEventListener("click", () => showTab("results"));
$("new-session").addEventListener("click", newSession);
$("open-profile").addEventListener("click", () => openDrawer("profile-panel"));
$("scrim").addEventListener("click", closeDrawers);
document.querySelectorAll("[data-close]").forEach((btn) => btn.addEventListener("click", closeDrawers));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawers(); });

messages.addEventListener("click", (e) => {
  const scenario = e.target.closest("[data-scenario]");
  if (scenario) {
    const s = SCENARIOS[Number(scenario.dataset.scenario)];
    send(s.text, s.next);
    return;
  }
  const open = e.target.closest("[data-open-turn]");
  if (open) {
    selectTurn(Number(open.dataset.openTurn));
    if (inspectorIsDrawer()) openDrawer("inspector");
    return;
  }
  const cite = e.target.closest("[data-cite]");
  if (cite) {
    selectTurn(Number(cite.closest(".msg-bot").dataset.turn));
    if (inspectorIsDrawer()) openDrawer("inspector");
    const el = $(`source-${cite.dataset.cite}`);
    if (el) {
      el.scrollIntoView({ block: "center" });
      el.classList.add("flash");
      setTimeout(() => el.classList.remove("flash"), 1400);
    }
  }
});

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = question.value;
  if (!text.trim() || state.busy) return;
  question.value = "";
  question.style.height = "";
  send(text);
});
question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
question.addEventListener("input", () => {
  question.style.height = "auto";
  question.style.height = `${question.scrollHeight}px`;
});

form.addEventListener("input", (e) => {
  state.dirty = true;
  if (e.target.name === "height" || e.target.name === "weight") updateBmi();
  setNote("Có thay đổi chưa lưu.");
});
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (state.busy) return;
  if (await saveProfile()) {
    closeDrawers();
    send("Đánh giá nguy cơ của tôi");
  }
});
$("save-profile").addEventListener("click", saveProfile);
$("clear-profile").addEventListener("click", async () => {
  clearForm();
  await api("/api/profile", { method: "POST", body: JSON.stringify({ session_id: state.session, profile: {} }) }).catch(() => {});
  setNote("Đã xóa chỉ số của phiên này.");
});

renderWelcome();
renderInspector(null);
