"""Lõi hệ đa agent cho web app, tách nguyên văn từ project01_05_15_haih.ipynb.

Module này không huấn luyện lại: DiaNet, RiskNet và Q-table được nạp từ results/.
Thống kê chuẩn hóa và ngưỡng F1 được tính lại từ cùng phép chia dữ liệu (SEED = 42)
nên khớp với notebook. Kiểm tra bằng: python webapp/check_engine.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # tránh xung đột OpenMP giữa PyTorch và scikit-learn trên Windows
import json, re, time, uuid, unicodedata, warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from torch import nn

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
torch.set_num_threads(1)  # máy 8 GB RAM, mạng nhỏ nên một luồng là đủ và ổn định hơn

# ---- Dữ liệu (Mục 1): cùng phép chia với notebook ----
raw = pd.read_csv(DATA / "diabetes_100k.csv")
df = raw.drop_duplicates()
df = df[df.gender != "Other"].reset_index(drop=True)
train_df, tmp = train_test_split(df, test_size=.30, random_state=SEED, stratify=df.diabetes)
val_df, test_df = train_test_split(tmp, test_size=.50, random_state=SEED, stratify=tmp.diabetes)

# ---- DiaNet (Mục 2) ----
class MaskedMLP(nn.Module):
    """MLP dùng chung cho DiaNet và RiskNet; mỗi đầu ra là một head."""
    def __init__(self, d_in, heads, hidden=(128, 64, 32), dropout=.1):
        super().__init__()
        layers, d = [], d_in
        for width in hidden:
            layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
            d = width
        self.body = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({name: nn.Linear(d, n) for name, n in heads.items()})

    def forward(self, x):
        h = self.body(x)
        return {name: head(h) for name, head in self.heads.items()}

SMOKE = ["never", "No Info", "current", "former", "ever", "not current"]
STATS = dict(age=(train_df.age.mean(), train_df.age.std()), bmi=(train_df.bmi.mean(), train_df.bmi.std()),
             h=(train_df.HbA1c_level.mean(), train_df.HbA1c_level.std()),
             g=(train_df.blood_glucose_level.mean(), train_df.blood_glucose_level.std()))
SCEN = ["none", "hba1c", "glucose", "both"]  # chỉ số = bit0 HbA1c + 2*bit1 glucose
SCEN_VI = {"none": "Chưa xét nghiệm", "hba1c": "Chỉ HbA1c", "glucose": "Chỉ đường huyết", "both": "Đủ hai xét nghiệm"}

def z(v, key):
    return ((np.asarray(v, dtype=np.float64) - STATS[key][0]) / STATS[key][1]).astype(np.float32)

def dia_arrays(d):
    """Trả về đặc trưng nền, HbA1c và đường huyết đã chuẩn hóa."""
    base = np.column_stack([z(d.age, "age"), z(d.bmi, "bmi"), d.hypertension, d.heart_disease, (d.gender == "Male") * 1.0]
                           + [(d.smoking_history == s) * 1.0 for s in SMOKE]).astype(np.float32)
    return base, z(d.HbA1c_level, "h"), z(d.blood_glucose_level, "g")

def dia_input(base, h, g, mh, mg):
    return np.column_stack([base, h * mh, g * mg, mh, mg]).astype(np.float32)

def predict_dia(model, arrays, scen):
    base, h, g = arrays
    k = SCEN.index(scen)
    mh, mg = np.full(len(h), k & 1, np.float32), np.full(len(h), k >> 1, np.float32)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(dia_input(base, h, g, mh, mg)))["cls"]).numpy().ravel()

def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    return float(th[np.argmax(f1[:-1])])

A_val = dia_arrays(val_df)
y_val = val_df.diabetes.values
dianet = MaskedMLP(A_val[0].shape[1] + 4, {"cls": 1})
dianet.load_state_dict(torch.load(RES / "dianet.pt", map_location="cpu"))
THRESH = {s: best_f1_threshold(y_val, predict_dia(dianet, A_val, s)) for s in SCEN}

# ---- RiskNet (Mục 3) ----
risk = pd.read_csv(DATA / "diabetes_risk_50k.csv")
ORD = {"Physical_Activity_Level": ["Low", "Moderate", "High"], "Diet_Quality": ["Poor", "Average", "Healthy"],
       "Sugar_Intake_Level": ["Low", "Moderate", "High"], "Stress_Level": ["Low", "Moderate", "High"],
       "Smoking_Status": ["Never", "Former", "Current"], "Alcohol_Consumption": ["Never", "Occasionally", "Frequently"]}
BIN = ["Family_History_Diabetes", "Hypertension", "Heart_Disease", "Fatty_Liver", "PCOS"]
NUM = ["Age", "BMI", "Waist_Circumference_cm", "HbA1c", "Fasting_Blood_Sugar", "Blood_Pressure_Systolic",
       "Exercise_Hours_Per_Week", "Sleep_Hours", "Insulin_Level", "Triglycerides", "HDL", "LDL", "Total_Cholesterol"]
RISK_COLS = NUM + list(ORD) + BIN
TIERS = ["Low", "Moderate", "High"]

def risk_frame(d):
    X = d[NUM].astype(float).copy()
    for c, levels in ORD.items():
        X[c] = d[c].map({v: i for i, v in enumerate(levels)}).astype(float)
    for c in BIN:
        X[c] = d[c].map({"No": 0., "Yes": 1.})
    return X[RISK_COLS]

R = risk_frame(risk)
yr, yc = risk.Diabetes_Risk_Score.values.astype(np.float32), risk.Diabetes_Risk.map({t: i for i, t in enumerate(TIERS)}).values
idx_tr, idx_tmp = train_test_split(np.arange(len(R)), test_size=.30, random_state=SEED, stratify=yc)
idx_va, idx_te = train_test_split(idx_tmp, test_size=.50, random_state=SEED, stratify=yc[idx_tmp])
R_mu, R_sd = R.iloc[idx_tr].mean(), R.iloc[idx_tr].std()

def risk_input(frame, keep=None):
    """Chuẩn hóa, điền 0 cho giá trị thiếu và nối mask đã biết/chưa biết."""
    Xz = ((frame - R_mu) / R_sd).astype(np.float32)
    M = Xz.notna().astype(np.float32).values
    if keep is not None:
        M = M * keep
    return np.hstack([np.nan_to_num(Xz.values) * M, M]).astype(np.float32)

risknet = MaskedMLP(len(RISK_COLS) * 2, {"score": 1, "tier": 3}, hidden=(256, 128, 64), dropout=0.)
risknet.load_state_dict(torch.load(RES / "risknet.pt", map_location="cpu"))

def risk_predict(frame, keep=None):
    risknet.eval()
    with torch.no_grad():
        out = risknet(torch.tensor(risk_input(frame, keep)))
    return out["score"].squeeze(1).numpy() * 100, out["tier"].softmax(1).numpy()

# ---- KB: Knowledge Graph và RAG (Mục 4) ----
docs = json.loads((DATA / "kb_documents.json").read_text(encoding="utf-8"))
DOC = {d["doc_id"]: d for d in docs}
triples = pd.read_csv(DATA / "kb_triples.csv")
thresholds = pd.read_csv(DATA / "kb_thresholds.csv")

G = nx.MultiDiGraph()
for t in triples.itertuples():
    G.add_node(t.subject, kind=t.subject_type); G.add_node(t.object, kind=t.object_type)
    G.add_edge(t.subject, t.object, relation=t.relation, doc_id=t.doc_id)

def out_of(node, relation=None):
    return [(node, e["relation"], o, e["doc_id"]) for _, o, e in G.out_edges(node, data=True) if relation in (None, e["relation"])]

def into(node, relation=None):
    return [(s, e["relation"], node, e["doc_id"]) for s, _, e in G.in_edges(node, data=True) if relation in (None, e["relation"])]

SYN = [("tieu duong", "dai thao duong"), ("tuyp", "type"), ("typ ", "type "), ("tip ", "type "), ("a1c", "hba1c"), ("hbhba1c", "hba1c"),
       ("tut duong", "ha duong huyet"), ("tut duong huyet", "ha duong huyet"), ("tap the duc", "van dong hoat dong the luc"),
       ("nuoc ngot", "do uong co duong"), ("thuoc la", "hut thuoc"), ("mang bau", "mang thai thai ky"), ("bau", "thai ky"),
       ("ngat", "mat y thuc"), ("glucose", "duong huyet glucose"), ("xu ly", "xu tri")]

def norm(s):
    s = str(s).lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def expand(q):
    s = norm(q)
    for a, b in SYN:
        s = s.replace(a, b)
    return s

ENTITY_KEYS = {norm(n).replace("diabetes", "dai thao duong"): n for n in G.nodes if len(n) >= 3 and G.nodes[n]["kind"] != "Feature"}

def link_entities(q):
    s = expand(q)
    found = [n for k, n in ENTITY_KEYS.items() if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", s)]
    return sorted(set(found), key=len, reverse=True)

# Khác notebook: câu hỏi tự nhiên có "tiểu đường" làm cụm này lấn át xếp hạng, nên thêm từ khóa của chủ đề vào truy vấn.
# Chỉ áp dụng cho truy vấn của người dùng; chỉ mục KB và bộ đánh giá truy xuất giữ nguyên.
TOPIC_HINTS = [
    (("phong tranh", "phong ngua", "phong benh", "de phong", "tranh bi", "lam sao de khong bi"), "phong type 2 giam can van dong an uong"),
    (("nen an", "an gi", "an uong", "che do an", "thuc don", "kieng", "thuc pham"), "phuong phap dia an an uong do uong"),
    (("tap the duc", "tap gi", "van dong", "the thao"), "van dong khi co dai thao duong khuyen nghi van dong"),
    (("chua khoi", "khoi han", "chua duoc"), "dieu tri kiem soat benh type 2"),
    (("dau hieu",), "bieu hien thuong gap trieu chung"),
    ((r"^(benh )?(tieu duong|dai thao duong) la",), "cac loai diabetes sinh ly glucose va phan loai"),
]

def with_topic_hints(q):
    s = norm(q)
    return " ".join([q] + [hint for keys, hint in TOPIC_HINTS if any(re.search(k, s) for k in keys)])

class HybridRetriever:
    def __init__(self, docs, dims=64):
        self.docs = docs
        self.ids = [d["doc_id"] for d in docs]
        text = [expand(d["title"] + ". " + d["text"]) for d in docs]
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.W = self.word.fit_transform(text)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        C = self.char.fit_transform(text)
        self.svd = TruncatedSVD(dims, random_state=SEED).fit(C)
        self.V = self._unit(self.svd.transform(C))
        self.edge_docs = defaultdict(set)
        for t in triples.itertuples():
            self.edge_docs[t.subject].add(t.doc_id); self.edge_docs[t.object].add(t.doc_id)

    @staticmethod
    def _unit(X):
        return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)

    def lexical(self, q):
        return (self.W @ self.word.transform([expand(q)]).T).toarray().ravel()

    def vector(self, q):
        return self.V @ self._unit(self.svd.transform(self.char.transform([expand(q)])))[0]

    def kg(self, q):
        ents = link_entities(q)
        score = np.zeros(len(self.ids))
        for e in ents:
            for i, d in enumerate(self.ids):
                score[i] += d in self.edge_docs[e]
        return score / max(1, len(ents))

    def search(self, q, k=5, mode="hybrid", rrf_k=30):
        scores = {"lexical": self.lexical(q), "vector": self.vector(q), "kg": self.kg(q)}
        if mode != "hybrid":
            rank = np.argsort(-scores[mode], kind="stable")
            return [(self.ids[i], float(scores[mode][i])) for i in rank[:k] if scores[mode][i] > 0]
        fused = np.zeros(len(self.ids))
        for name, weight in (("lexical", 1.0), ("vector", .7), ("kg", .5)):
            order = np.argsort(-scores[name], kind="stable")
            for r, i in enumerate(order):
                if scores[name][i] > 0:
                    fused[i] += weight / (rrf_k + r + 1)
        rank = np.argsort(-fused, kind="stable")
        return [(self.ids[i], float(scores["lexical"][i])) for i in rank[:k] if fused[i] > 0]

retriever = HybridRetriever(docs)

# ---- Q-table và Rule shield (Mục 5) ----
ACTIONS = ["ORDER_HBA1C", "ORDER_GLUCOSE", "ADVISE_ROUTINE", "REFER_DOCTOR"]
A_H, A_G, A_ROUTINE, A_REFER = range(4)
EDGES = [.02, .05, .10, .30, .60]
NB = len(EDGES) + 1
N_STATES = 4 * NB

Q = np.load(RES / "q_table.npy")
PI = Q.argmax(1)

H_DIAG = float(thresholds.query("test=='HbA1c' and category=='Ngưỡng đái tháo đường'").lower.iloc[0])
G_FAST = float(thresholds.query("test=='Đường huyết đói' and category=='Ngưỡng đái tháo đường'").lower.iloc[0])
G_RAND = float(thresholds.query("test=='Đường huyết ngẫu nhiên'").lower.iloc[0])
AGE_SCREEN = 35

def lab_flags(known, h, g, fasting=False):
    hk, gk = (known & 1) > 0, (known & 2) > 0
    g_cut = G_FAST if fasting else G_RAND
    diag = (hk & (h >= H_DIAG)) | (gk & (g >= g_cut))
    unclear = gk & ~hk & (g >= G_FAST) & (g < G_RAND) & (not fasting)
    return hk, gk, diag, unclear

def apply_shield(a, known, h, g, age, fasting=False):
    """Trả về hành động sau luật và mã luật đã kích hoạt cho từng bệnh nhân."""
    a, fired = a.copy(), np.full(len(a), "", dtype=object)
    hk, gk, diag, unclear = lab_flags(known, h, g, fasting)
    r2 = unclear & (a == A_ROUTINE); a[r2] = A_H; fired[r2] = "R2"
    r3 = ~hk & ~gk & (age >= AGE_SCREEN) & (a == A_ROUTINE); a[r3] = A_H; fired[r3] = "R3"
    r1 = diag & (a != A_REFER); a[r1] = A_REFER; fired[r1] = "R1"
    return a, fired

def rule_action(known, h, g, age, bmi, fasting=False):
    """Chính sách chỉ dùng luật, làm mốc so sánh với RL."""
    hk, gk, diag, _ = lab_flags(known, h, g, fasting)
    a = np.full(len(known), A_ROUTINE)
    a[~hk & ((age >= AGE_SCREEN) | (bmi >= 25))] = A_H
    a[hk & ~gk & (h >= 5.7)] = A_G
    a[diag] = A_REFER
    return a

# ---- Giao thức tin nhắn (Mục 6) ----
SCHEMA = {
 "PARSE": ["text", "memory"], "PARSED": ["profile", "intent", "errors", "missing", "new_fields"],
 "PRECHECK": ["text", "profile"], "SAFETY": ["level", "reason"],
 "PREDICT": ["profile"], "PREDICTION": ["p", "label", "scenario", "threshold", "score", "tier", "drivers", "what_if"],
 "DECIDE": ["p", "profile"], "DECISION": ["action", "rl_action", "rules", "lab_checks"],
 "RETRIEVE": ["queries"], "EVIDENCE": ["items", "facts", "stale", "expanded", "tools"],
 "COMPOSE": ["state"], "DRAFT": ["text"], "CHECKLIST": ["state"], "PLAN": ["text"],
 "REVIEW": ["state"], "VERDICT": ["passed", "issues"],
}

@dataclass
class Message:
    sender: str
    receiver: str
    kind: str
    payload: dict
    conversation: str = ""
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)

def validate_message(m):
    """Hợp đồng giữa các agent: sai kiểu hoặc thiếu trường thì coi là lỗi."""
    if not isinstance(m, Message):
        raise TypeError("Phản hồi không phải Message")
    if m.kind not in SCHEMA:
        raise ValueError(f"Kiểu tin nhắn lạ: {m.kind}")
    missing = [k for k in SCHEMA[m.kind] if k not in m.payload]
    if missing:
        raise ValueError(f"{m.kind} thiếu trường {missing}")
    return m

class Agent:
    tools = ()
    def __init__(self, name):
        self.name = name
    def reply(self, msg, kind, **payload):
        return Message(self.name, msg.sender, kind, payload, msg.conversation)
    def handle(self, msg):
        raise NotImplementedError

def vn(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")

def pct(p):
    """Không hiển thị 0% hoặc 100% để tránh hiểu xác suất mô hình là chắc chắn."""
    return "trên 99,9%" if p > .999 else ("dưới 0,1%" if p < .001 else f"{vn(p * 100)}%")

# ---- Intake và Safety ----
NUMBER = r"(\d+(?:[.,]\d+)?)"
to_f = lambda s: float(s.replace(",", "."))
LIMITS = {"age": (1, 120), "bmi": (10, 70), "hba1c": (3, 20), "glucose": (20, 800)}

def parse_profile(text):
    s, p = norm(text), {}
    if m := re.search(rf"{NUMBER}\s*tuoi", s): p["age"] = to_f(m[1])
    # Khác notebook: chỉ nhận giới tính khi đi kèm tuổi hoặc nêu rõ, vì "chỉ", "ảnh", "năm" bỏ dấu trùng "chi", "anh", "nam"
    head = str(text).strip().lower()
    if re.search(r"\b(nam|ong|anh)[\s,]+\d+\s*tuoi|\bbo toi\b|\b(gioi tinh\W*|toi la )nam\b", s) or re.match(r"nam\b", head): p["gender"] = "Male"
    if re.search(r"\b(nu|ba|chi)[\s,]+\d+\s*tuoi|\bme toi\b|\b(gioi tinh\W*|toi la )nu\b", s) or re.match(r"nữ\b", head): p["gender"] = "Female"
    if m := re.search(rf"bmi\D{{0,6}}{NUMBER}", s): p["bmi"] = to_f(m[1])
    hgt = re.search(r"cao\s*(\d)m(\d{1,2})", s) or re.search(r"cao\s*(\d)[.,](\d{1,2})\s*m", s)
    cm = re.search(r"cao\s*(\d{3})\s*cm", s)
    wgt = re.search(rf"nang\s*{NUMBER}\s*(?:kg|can)", s)
    height = int(hgt[1]) + int(hgt[2].ljust(2, "0")) / 100 if hgt else (int(cm[1]) / 100 if cm else None)
    if height and wgt and "bmi" not in p:
        p["bmi"] = round(to_f(wgt[1]) / height ** 2, 1)
    if m := re.search(rf"(?:hba1c|a1c)\D{{0,12}}{NUMBER}", s): p["hba1c"] = to_f(m[1])
    if m := re.search(rf"(?:duong huyet|glucose)(\s+luc doi|\s+doi|\s+ngau nhien|\s+bat ky)?\D{{0,16}}{NUMBER}\s*(mmol)?", s):
        p["glucose"] = round(to_f(m[2]) * 18, 0) if m[3] else to_f(m[2])
        kind = (m[1] or "").strip()
        p["glucose_kind"] = "fasting" if "doi" in kind else ("random" if kind else "unknown")
    if re.search(r"(?<!khong )(tang huyet ap|cao huyet ap)", s): p["hypertension"] = 1
    if re.search(r"khong (bi )?(tang huyet ap|cao huyet ap)", s): p["hypertension"] = 0
    if re.search(r"(?<!khong )(benh tim|suy tim)", s): p["heart_disease"] = 1
    if re.search(r"khong (hut thuoc|co hut thuoc)", s): p["smoking"] = "never"
    elif re.search(r"(bo|cai|da ngung) thuoc", s): p["smoking"] = "former"
    elif re.search(r"(dang )?hut thuoc", s) and "anh huong" not in s and "?" not in text: p["smoking"] = "current"
    elif re.search(r"dang hut thuoc", s): p["smoking"] = "current"
    if re.search(r"(bo|me|cha|anh|chi|em)( toi)? (bi|mac) (tieu duong|dai thao duong)", s): p["family_history"] = 1
    if re.search(r"it van dong|luoi van dong|ngoi nhieu", s): p["activity"] = "Low"
    return p

KW_SUPPORT = ["chuan bi", "checklist", "danh sach", "mang gi", "hoi bac si gi", "tai kham"]
KW_ASSESS = ["nguy co", "danh gia", "kiem tra giup", "chi so cua toi", "ket qua cua toi"]  # hai cụm cuối thêm cho web app
KW_DOMAIN = ["dai thao duong", "tieu duong", "duong huyet", "insulin", "hba1c", "a1c", "van dong", "metformin", "sglt2", "cgm", "tuyp", "thuoc tieu duong"]

class IntakeAgent(Agent):
    tools = ("parse_profile",)
    def handle(self, msg):
        text, memory = msg.payload["text"], msg.payload["memory"]
        found, errors = parse_profile(text), []
        for key, (lo, hi) in LIMITS.items():
            if key in found and not lo <= found[key] <= hi:
                errors.append(f"{key}={found[key]:g} nằm ngoài khoảng hợp lệ {lo}–{hi}")
                found.pop(key)
        profile = {**memory.get("profile", {}), **found}
        s = norm(text)
        measured = {k for k in found if k in ("age", "bmi", "hba1c", "glucose")}
        domain = bool(link_entities(text)) or any(k in s for k in KW_DOMAIN)
        if any(k in s for k in KW_SUPPORT) and (domain or memory.get("profile")):
            intent = "support"
        elif any(k in s for k in KW_ASSESS) or measured or errors or memory.get("pending") == "assess":
            intent = "assess"
        elif domain:
            intent = "knowledge"
        else:
            intent = "out_of_scope"
        missing = [k for k in ("age", "bmi") if k not in profile] if intent == "assess" else []
        return self.reply(msg, "PARSED", profile=profile, intent=intent, errors=errors, missing=missing, new_fields=sorted(found))

EMERGENCY = ["bat tinh", "hon me", "co giat", "kho tho", "dau nguc", "khong danh thuc", "ngat xiu", "non lien tuc"]
BOUNDARY = [r"\blieu\b", "ke don", "ke cho", "bao nhieu mg", "may vien", "ngung insulin", "ngung thuoc", "bo thuoc", "doi thuoc", "tang lieu", "giam lieu"]
DOSE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:mg(?!/dL)|mcg|viên|đơn vị|UI\b|IU\b)(?!/)", re.I)
DIAGNOSIS = re.compile(r"(bạn|anh|chị|ông|bà)\s+(đã\s+)?(bị|mắc)\s+(bệnh\s+)?(đái tháo đường|tiểu đường)", re.I)
UNIT_NUMBER = re.compile(rf"{NUMBER}\s*(%|mg/dL|mmol/L)")
DISCLAIMER = "Thông tin chỉ để tham khảo, không thay thế chẩn đoán của bác sĩ."

def allowed_numbers(state):
    vals = set()
    for e in state.get("evidence", []):
        vals |= {to_f(x) for x in re.findall(NUMBER, e["text"])}
    prof = state.get("profile", {})
    vals |= {float(prof[k]) for k in ("hba1c", "glucose", "bmi", "age") if k in prof}
    if state.get("prediction"):
        pr = state["prediction"]
        vals |= {round(pr["p"] * 100, 1), round(pr["threshold"] * 100, 1), round(pr["score"], 1), 99.9, .1}
        vals |= {round(abs(d) * 100, 1) for _, d in pr["drivers"]}
    for c in (state.get("decision") or {}).get("lab_checks", []):
        vals.add(float(c["value"]))
    return vals

class SafetyAgent(Agent):
    tools = ("red_flags", "critic")
    def handle(self, msg):
        return self.precheck(msg) if msg.kind == "PRECHECK" else self.review(msg)

    def precheck(self, msg):
        s, p = norm(msg.payload["text"]), msg.payload["profile"]
        if hit := [k for k in EMERGENCY if k in s]:
            return self.reply(msg, "SAFETY", level="emergency", reason=hit[0])
        if p.get("glucose", 999) < 70:
            return self.reply(msg, "SAFETY", level="urgent", reason="đường huyết dưới 70 mg/dL")
        if any(re.search(k, s) for k in BOUNDARY):
            return self.reply(msg, "SAFETY", level="boundary", reason="kê đơn hoặc thay đổi thuốc")
        return self.reply(msg, "SAFETY", level="ok", reason="")

    def review(self, msg):
        st = msg.payload["state"]; draft = st["draft"]; ev = st.get("evidence", []); issues = []
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", draft)}
        if bad := sorted(n for n in cited if not 1 <= n <= len(ev)):
            issues.append(("CITATION", f"Trích dẫn {bad} không tồn tại"))
        if ev and not cited and st.get("needs_citation", True):
            issues.append(("CITATION", "Thiếu trích dẫn cho nội dung y khoa"))
        if DOSE.search(draft):
            issues.append(("DOSE", "Câu trả lời không được nêu liều thuốc"))
        if DIAGNOSIS.search(draft):
            issues.append(("DIAGNOSIS", "Không được khẳng định người dùng mắc bệnh"))
        if DISCLAIMER not in draft and st.get("needs_citation", True):
            issues.append(("DISCLAIMER", "Thiếu câu nhắc tham khảo"))
        ok = allowed_numbers(st)
        for raw, unit in UNIT_NUMBER.findall(draft):
            if not any(abs(to_f(raw) - a) < .051 for a in ok):
                issues.append(("NUMBER", f"Số {raw}{unit} không có nguồn"))
        have = {e["doc_id"] for e in ev}
        for c in (st.get("decision") or {}).get("lab_checks", []):
            if c["doc_id"] not in have:
                issues.append(("NEED_EVIDENCE", c["doc_id"]))
            if c["diagnostic"] and "ngưỡng" not in draft:
                issues.append(("REASSURANCE", f"Phải nêu {c['test']} đạt ngưỡng"))
        return self.reply(msg, "VERDICT", passed=not issues, issues=issues)

# ---- Prediction, Policy và Knowledge ----
def lab_checks(profile):
    """Công cụ threshold_lookup: đối chiếu chỉ số người dùng với bảng ngưỡng trong KB."""
    out = []
    def row(test, value, unit, category, doc_id, diagnostic=False):
        out.append(dict(test=test, value=value, unit=unit, category=category, doc_id=doc_id, diagnostic=diagnostic))
    if "hba1c" in profile:
        v = profile["hba1c"]
        for t in thresholds[thresholds.test == "HbA1c"].itertuples():
            if (pd.isna(t.lower) or v >= t.lower) and (pd.isna(t.upper) or v < t.upper):
                row("HbA1c", v, "%", t.category, t.doc_id, "Ngưỡng đái" in t.category)
    if "glucose" in profile:
        v, kind = profile["glucose"], profile.get("glucose_kind", "unknown")
        if kind == "fasting":
            for t in thresholds[thresholds.test == "Đường huyết đói"].itertuples():
                if (pd.isna(t.lower) or v >= t.lower) and (pd.isna(t.upper) or v < t.upper):
                    row("Đường huyết lúc đói", v, "mg/dL", t.category, t.doc_id, "Ngưỡng đái" in t.category)
        elif v >= G_RAND:
            row("Đường huyết", v, "mg/dL", "Ngưỡng đường huyết ngẫu nhiên, cần xác nhận", "N04", True)
        elif v >= G_FAST:
            row("Đường huyết", v, "mg/dL", "Chưa rõ đo lúc đói hay không", "N02")
    return out

def dia_vector(p):
    smoke = {"current": "current", "former": "former", "never": "never"}.get(p.get("smoking"), "No Info")
    gender = {"Male": 1., "Female": 0.}.get(p.get("gender"), .5)
    base = np.array([[float(z(p["age"], "age")), float(z(p["bmi"], "bmi")), p.get("hypertension", 0), p.get("heart_disease", 0), gender]
                     + [float(smoke == s) for s in SMOKE]], np.float32)
    h, g = np.array([z(p.get("hba1c", 0), "h")], np.float32), np.array([z(p.get("glucose", 0), "g")], np.float32)
    mh, mg = np.array([float("hba1c" in p)], np.float32), np.array([float("glucose" in p)], np.float32)
    return dia_input(base, h, g, mh, mg)

def dia_prob(p):
    dianet.eval()
    with torch.no_grad():
        return float(torch.sigmoid(dianet(torch.tensor(dia_vector(p)))["cls"]))

def risk_row(p):
    row = {c: np.nan for c in RISK_COLS}
    row.update(Age=p["age"], BMI=p["bmi"])
    if "hba1c" in p: row["HbA1c"] = p["hba1c"]
    if p.get("glucose_kind") == "fasting": row["Fasting_Blood_Sugar"] = p["glucose"]
    if "hypertension" in p: row["Hypertension"] = float(p["hypertension"])
    if "heart_disease" in p: row["Heart_Disease"] = float(p["heart_disease"])
    if "family_history" in p: row["Family_History_Diabetes"] = float(p["family_history"])
    if "smoking" in p: row["Smoking_Status"] = {"never": 0., "former": 1., "current": 2.}[p["smoking"]]
    if "activity" in p: row["Physical_Activity_Level"] = float(ORD["Physical_Activity_Level"].index(p["activity"]))
    return pd.DataFrame([row])[RISK_COLS]

class PredictionAgent(Agent):
    tools = ("DiaNet", "RiskNet")
    def handle(self, msg):
        p = msg.payload["profile"]
        scen = SCEN[int("hba1c" in p) + 2 * int("glucose" in p)]
        prob, thr = dia_prob(p), THRESH[scen]
        score, tier = risk_predict(risk_row(p))
        refs = {"Xét nghiệm HbA1c": ("hba1c", None), "Đường huyết": ("glucose", None), "BMI": ("bmi", 23.), "Tuổi": ("age", 30.),
                "Tăng huyết áp": ("hypertension", 0), "Bệnh tim": ("heart_disease", 0), "Hút thuốc": ("smoking", "never")}
        drivers = []
        for name, (key, ref) in refs.items():
            if key in p and p[key] != ref:
                alt = {k: v for k, v in p.items() if k != key} if ref is None else {**p, key: ref}
                drivers.append((name, prob - dia_prob(alt)))
        drivers = sorted([d for d in drivers if abs(d[1]) >= .005], key=lambda d: -abs(d[1]))[:3]
        what_if = []
        # Chỉ mô phỏng yếu tố người dùng đã khai, vì không biết mức hiện tại thì không có mốc so sánh
        changes = {"giảm BMI về 24": {"BMI": min(24., p["bmi"])}, "ngừng hút thuốc": {"Smoking_Status": 0.}, "vận động mức cao": {"Physical_Activity_Level": 2.}}
        ref_row, ref_score = risk_row(p), float(score[0])
        for name, ch in changes.items():
            if name == "giảm BMI về 24" and p["bmi"] <= 24: continue
            if name == "ngừng hút thuốc" and p.get("smoking") != "current": continue
            if name == "vận động mức cao" and p.get("activity") != "Low": continue
            alt = ref_row.copy()
            for k, v in ch.items(): alt[k] = v
            what_if.append((name, float(risk_predict(alt)[0][0]) - ref_score))
        return self.reply(msg, "PREDICTION", p=prob, label=int(prob >= thr), scenario=scen, threshold=thr, score=float(score[0]),
                          tier=TIERS[int(tier.argmax())], drivers=drivers, what_if=what_if)

class PolicyAgent(Agent):
    tools = ("Q-table", "threshold_lookup")
    def __init__(self, name, mode="rl+rule"):
        super().__init__(name); self.mode = mode
    def handle(self, msg):
        pr, p = msg.payload["p"], msg.payload["profile"]
        known = np.array([int("hba1c" in p) + 2 * int("glucose" in p)])
        h, g, age, bmi = (np.array([float(p.get(k, 0))]) for k in ("hba1c", "glucose", "age", "bmi"))
        fasting = p.get("glucose_kind") == "fasting"
        rl = PI[known * NB + np.digitize([pr], EDGES)]
        rules = ""
        if self.mode == "rule":
            a = rule_action(known, h, g, age, bmi, fasting)
        elif self.mode == "rl":
            a = rl
        else:
            a, fired = apply_shield(rl, known, h, g, age, fasting); rules = fired[0]
        return self.reply(msg, "DECISION", action=ACTIONS[int(a[0])], rl_action=ACTIONS[int(rl[0])], rules=rules, lab_checks=lab_checks(p))

class KnowledgeAgent(Agent):
    tools = ("search_kb", "kg_neighbors")
    MIN_SCORE, MAX_AGE_DAYS = .04, 730
    def __init__(self, name, today=None):
        super().__init__(name); self.today = pd.Timestamp(today or "2026-11-05")

    def facts_for(self, text, limit=6):
        facts = []
        for e in link_entities(text)[:3]:
            facts += [f"({s}, {r}, {o}) [{d}]" for s, r, o, d in (out_of(e) + into(e))[:3]]
        return list(dict.fromkeys(facts))[:limit]

    def handle(self, msg):
        items, tools, expanded = {}, [], False
        k = msg.payload.get("k", 3)
        for q in msg.payload["queries"]:
            tools.append("search_kb")
            hits = retriever.search(q, k=k)
            if not hits or hits[0][1] < self.MIN_SCORE:  # bằng chứng yếu: tự mở rộng truy vấn bằng KG
                ents = link_entities(q)
                if ents:
                    tools.append("kg_neighbors"); expanded = True
                    neigh = [o for _, _, o, _ in out_of(ents[0])[:4]] + [s for s, _, _, _ in into(ents[0])[:4]]
                    hits = retriever.search(q + " " + " ".join(neigh), k=k)
            for doc_id, score in hits:
                if score >= self.MIN_SCORE / 2 and doc_id not in items:
                    items[doc_id] = score
        for doc_id in msg.payload.get("doc_ids", []):
            items.setdefault(doc_id, 1.0)
        out = [{**DOC[d], "score": s} for d, s in items.items()]
        stale = [d["doc_id"] for d in out if (self.today - pd.Timestamp(d["accessed"])).days > self.MAX_AGE_DAYS]
        return self.reply(msg, "EVIDENCE", items=out, facts=self.facts_for(" ".join(msg.payload["queries"])), stale=stale, expanded=expanded, tools=tools)

# ---- Advisor, Support và Coordinator ----
ACTION_VI = {"ORDER_HBA1C": "nên làm xét nghiệm HbA1c", "ORDER_GLUCOSE": "nên đo đường huyết lúc đói",
             "ADVISE_ROUTINE": "duy trì lối sống lành mạnh và khám định kỳ", "REFER_DOCTOR": "nên gặp bác sĩ để được đánh giá và xác nhận"}

class AdvisorAgent(Agent):
    tools = ("compose",)
    def cite(self, st, doc_id):
        for i, e in enumerate(st["evidence"], 1):
            if e["doc_id"] == doc_id:
                return f"[{i}]"
        return ""

    def facts(self, st, limit=3):
        return [f"- {e['text']} [{i}]" for i, e in enumerate(st["evidence"], 1)][:limit]

    def handle(self, msg):
        st = msg.payload["state"]; L = []
        level = st.get("safety", {}).get("level", "ok")
        if st.get("oos"):
            L = ["Câu hỏi này nằm ngoài phạm vi hệ thống. Hệ thống chỉ hỗ trợ thông tin về đái tháo đường."]
        elif st.get("ask"):
            L = [st["ask"]]
        elif level == "emergency":
            L = ["**Đây có thể là tình huống khẩn cấp. Hãy gọi cấp cứu 115 hoặc đến cơ sở y tế gần nhất ngay.** Không chờ hệ thống tư vấn thêm."] + self.facts(st, 2)
        elif level == "urgent":
            L = ["**Chỉ số này có thể là hạ đường huyết, cần xử trí ngay.** Nếu không tự xử trí được hoặc mất ý thức, hãy gọi cấp cứu 115."] + self.facts(st, 3)
        elif level == "boundary":
            L = ["Hệ thống không kê đơn, không tính liều và không hướng dẫn tự thay đổi thuốc. Hãy hỏi bác sĩ điều trị trước khi thay đổi."] + self.facts(st, 2)
        elif st.get("prediction") or st.get("decision"):
            L = self.assessment(st)
        elif st.get("plan"):
            L = []
        elif st["evidence"]:
            L = ["Theo các nguồn đã truy xuất:"] + self.facts(st, 3)
        else:
            L = ["Chưa truy xuất được bằng chứng phù hợp nên hệ thống không trả lời nội dung y khoa này. Hãy hỏi nhân viên y tế."]
        if st.get("kb_failed"):
            L.append("Lưu ý: chưa truy xuất được nguồn tri thức nên phần giải thích chưa được kiểm chứng.")
        if st.get("stale"):
            L.append(f"Lưu ý: nguồn {', '.join(st['stale'])} đã cũ, cần kiểm tra lại.")
        if st.get("plan"):
            L.append(st["plan"])
        if not st.get("oos") and not st.get("ask"):
            L.append(DISCLAIMER)
        return self.reply(msg, "DRAFT", text="\n\n".join(L))

    def assessment(self, st):
        pr, de, p = st.get("prediction"), st["decision"], st["profile"]
        L = []
        if pr:
            L.append(f"**Phân loại (DiaNet, {SCEN_VI[pr['scenario']].lower()}):** xác suất ước lượng {pct(pr['p'])}, ngưỡng phân loại {vn(pr['threshold'] * 100)}%, nhóm {'nguy cơ cao' if pr['label'] else 'dưới ngưỡng phân loại'}.")
            L.append(f"**Dự đoán (RiskNet):** điểm nguy cơ tham khảo {vn(pr['score'])}/100, mức {pr['tier']}. Điểm này không phải xác suất mắc bệnh.")
            if pr["drivers"]:
                L.append("Yếu tố ảnh hưởng lớn trong mô hình: " + "; ".join(f"{n} ({'+' if d >= 0 else '−'}{vn(abs(d) * 100)} điểm phần trăm)" for n, d in pr["drivers"]) + ".")
        for c in de["lab_checks"]:
            c_ref = self.cite(st, c["doc_id"])
            word = "đạt ngưỡng:" if c["diagnostic"] else "thuộc nhóm:"
            L.append(f"**Đối chiếu:** {c['test']} {vn(c['value'], 1 if c['unit'] == '%' else 0)}{'' if c['unit'] == '%' else ' '}{c['unit']} {word} {c['category']} {c_ref}.")
        if pr and not pr["label"] and any(c["diagnostic"] for c in de["lab_checks"]):
            L.append("Dữ liệu huấn luyện có nhãn không khớp ngưỡng y khoa ở nhóm này, nên xác suất mô hình thấp không có nghĩa là an toàn. Hệ thống ưu tiên ngưỡng từ tài liệu.")
        rule_note = f" (luật {de['rules']} ghi đè đề xuất {de['rl_action']} của RL)" if de["rules"] else ""
        L.append(f"**Bước tiếp theo:** {ACTION_VI[de['action']]}{rule_note}.")
        tail = {"REFER_DOCTOR": "N05", "ORDER_HBA1C": "N06", "ORDER_GLUCOSE": "N06", "ADVISE_ROUTINE": "N07"}[de["action"]]
        if ref := self.cite(st, tail):
            L.append(f"{DOC[tail]['text']} {ref}")
        for doc_id, cond in (("N28", p.get("smoking") == "current"), ("N10", p.get("bmi", 0) >= 25), ("N26", p.get("activity") == "Low")):
            if cond and (ref := self.cite(st, doc_id)):
                L.append(f"{DOC[doc_id]['text']} {ref}")
        if pr and pr["what_if"]:
            L.append("Mô phỏng RiskNet nếu thay đổi: " + "; ".join(f"{n}: {'+' if d >= 0 else '−'}{vn(abs(d))} điểm" for n, d in pr["what_if"]) + ". Đây là tương quan trong dữ liệu, không phải hiệu quả điều trị.")
        return L

class SupportAgent(Agent):
    tools = ("make_checklist",)
    def handle(self, msg):
        st = msg.payload["state"]; p = st.get("profile", {})
        cite = lambda d: next((f" [{i}]" for i, e in enumerate(st["evidence"], 1) if e["doc_id"] == d), "")
        labs = []
        if "hba1c" in p: labs.append(f"HbA1c {vn(p['hba1c'])}%")
        if "glucose" in p: labs.append(f"đường huyết {vn(p['glucose'], 0)} mg/dL")
        L = ["**Danh sách chuẩn bị đi khám**",
             f"- Mang kết quả đã có: {', '.join(labs) if labs else 'các xét nghiệm gần nhất nếu có'}; toa thuốc đang dùng.",
             f"- Hỏi bác sĩ có cần xét nghiệm xác nhận không{cite('N05')}.",
             f"- Hỏi mục tiêu HbA1c và huyết áp của bạn{cite('N20')}.",
             f"- Hỏi lịch khám mắt và kiểm tra bàn chân{cite('N30')}{cite('N31')}.",
             "- Ghi lại triệu chứng và câu hỏi của bạn. Hệ thống không tự đặt lịch hẹn."]
        return self.reply(msg, "PLAN", text="\n".join(L))

class Coordinator(Agent):
    """Điều phối bằng trạng thái dùng chung; mọi lời gọi agent đều đi qua call()."""
    def __init__(self, policy_mode="rl+rule", safety=True, use_dl=True):
        super().__init__("coordinator")
        self.team = {"intake": IntakeAgent("intake"), "safety": SafetyAgent("safety"), "prediction": PredictionAgent("prediction"),
                     "policy": PolicyAgent("policy", policy_mode), "knowledge": KnowledgeAgent("knowledge"),
                     "advisor": AdvisorAgent("advisor"), "support": SupportAgent("support")}
        self.safety_on, self.use_dl = safety, use_dl
        self.sessions = defaultdict(dict)

    def call(self, st, target, kind, payload, fallback=None, retries=1):
        msg = Message(self.name, target, kind, payload, st["conversation"])
        validate_message(msg)
        for attempt in range(retries + 1):
            t = time.perf_counter()
            try:
                out = validate_message(self.team[target].handle(msg))
                st["trace"].append(dict(agent=target, kind=kind, reply=out.kind, ms=round((time.perf_counter() - t) * 1000, 1), status="ok"))
                st["messages"] += 2
                st["path"].append(target)
                st["tools"] += list(self.team[target].tools) if target != "knowledge" else out.payload["tools"]
                return out.payload
            except (TimeoutError, TypeError, ValueError, KeyError) as err:
                st["trace"].append(dict(agent=target, kind=kind, reply="ERROR", ms=round((time.perf_counter() - t) * 1000, 1), status=f"{type(err).__name__}: {err}"))
                st["messages"] += 1; st["failures"] += 1
        return fallback

    def retrieve(self, st, queries, doc_ids=()):
        res = self.call(st, "knowledge", "RETRIEVE", {"queries": list(queries), "doc_ids": list(doc_ids)})
        if res is None:
            st["kb_failed"] = True
            return
        have = {e["doc_id"] for e in st["evidence"]}
        st["evidence"] += [e for e in res["items"] if e["doc_id"] not in have]
        st["facts"] = list(dict.fromkeys(st["facts"] + res["facts"]))
        st["stale"] = sorted(set(st["stale"]) | set(res["stale"]))
        st["expanded"] = st["expanded"] or res["expanded"]

    def ask(self, text, session="default"):
        t0 = time.perf_counter()
        memory = self.sessions[session]
        st = dict(text=text, conversation=session, trace=[], path=[], tools=[], messages=0, failures=0, evidence=[], facts=[], stale=[], expanded=False,
                  prediction=None, decision=None, ask=None, oos=False, plan=None, safety={"level": "ok", "reason": ""}, critic_rounds=0, issues=[])
        parsed = self.call(st, "intake", "PARSE", {"text": text, "memory": memory},
                           fallback=dict(profile=memory.get("profile", {}), intent="knowledge", errors=[], missing=[], new_fields=[]))
        st["profile"], st["intent"] = parsed["profile"], parsed["intent"]
        memory["profile"] = dict(parsed["profile"])
        if self.safety_on:
            st["safety"] = self.call(st, "safety", "PRECHECK", {"text": text, "profile": st["profile"]}, fallback=dict(level="boundary", reason="safety unavailable"))
        level = st["safety"]["level"]

        if level == "emergency":
            self.retrieve(st, ["hạ đường huyết nặng mất ý thức cấp cứu", "nhiễm toan ceton cấp cứu"])
        elif level == "urgent":
            self.retrieve(st, ["xử trí hạ đường huyết nhẹ đo lại sau 15 phút", "hạ đường huyết nặng"], doc_ids=["N18"])
        elif level == "boundary":
            self.retrieve(st, [text, "dùng thuốc đúng chỉ định không tự đổi thuốc"])
        elif st["intent"] == "out_of_scope":
            st["oos"] = True
        elif st["intent"] == "assess":
            self.assess(st, parsed)
        elif st["intent"] == "support":
            self.checklist(st)
        else:
            self.retrieve(st, [with_topic_hints(text)])
        self.write(st)
        memory["pending"] = "assess" if st["ask"] else None
        st["latency_ms"] = (time.perf_counter() - t0) * 1000
        st["answer"] = st["draft"]
        return st

    def checklist(self, st):
        self.retrieve(st, ["xét nghiệm xác nhận chẩn đoán", "mục tiêu HbA1c huyết áp"], doc_ids=["N05", "N20", "N30", "N31"])
        st["plan"] = (self.call(st, "support", "CHECKLIST", {"state": st}) or {}).get("text")

    def assess(self, st, parsed):
        if parsed["errors"] or parsed["missing"]:
            need = {"age": "tuổi", "bmi": "BMI hoặc chiều cao và cân nặng"}
            st["ask"] = ("; ".join(parsed["errors"]) + ". Hãy kiểm tra và nhập lại.") if parsed["errors"] else "Để đánh giá, hãy cho biết thêm: " + ", ".join(need[m] for m in parsed["missing"]) + "."
            return
        p = st["profile"]
        pred = self.call(st, "prediction", "PREDICT", {"profile": p}) if self.use_dl else None
        st["prediction"] = pred
        st["decision"] = self.call(st, "policy", "DECIDE", {"p": pred["p"] if pred else .5, "profile": p},
                                   fallback=dict(action="REFER_DOCTOR", rl_action="REFER_DOCTOR", rules="FALLBACK", lab_checks=lab_checks(p)))
        action = st["decision"]["action"]
        topic = {"REFER_DOCTOR": "cần xét nghiệm xác nhận chẩn đoán", "ADVISE_ROUTINE": "xét nghiệm lại sau mỗi 3 năm tần suất",
                 "ORDER_HBA1C": "ai nên xét nghiệm từ 35 tuổi", "ORDER_GLUCOSE": "ai nên xét nghiệm từ 35 tuổi"}[action]
        queries = [topic]
        if p.get("smoking") == "current": queries.append("hút thuốc đái tháo đường")
        if p.get("bmi", 0) >= 25: queries.append("giảm 5 đến 7 phần trăm cân nặng phòng type 2")
        if p.get("activity") == "Low": queries.append("150 phút hoạt động thể lực mỗi tuần")
        self.retrieve(st, queries)
        if action == "REFER_DOCTOR":
            self.checklist(st)

    def write(self, st):
        st["needs_citation"] = not (st["oos"] or st["ask"] or st.get("kb_failed"))
        st["draft"] = self.call(st, "advisor", "COMPOSE", {"state": st}, fallback=dict(text=DISCLAIMER))["text"]
        if not self.safety_on:
            return
        for round_ in range(3):
            verdict = self.call(st, "safety", "REVIEW", {"state": st}, fallback=dict(passed=False, issues=[("SAFETY", "unavailable")]))
            st["issues"] += verdict["issues"]
            if verdict["passed"]:
                return
            if round_ == 2:
                break
            st["critic_rounds"] += 1
            need = [v for k, v in verdict["issues"] if k == "NEED_EVIDENCE"]
            if need:
                self.retrieve(st, [DOC[d]["title"] for d in need], doc_ids=need)
            st["draft"] = self.call(st, "advisor", "COMPOSE", {"state": st}, fallback=dict(text=DISCLAIMER))["text"]
        st["draft"] = "Hệ thống không thể kiểm chứng câu trả lời nên không hiển thị nội dung chi tiết. Hãy hỏi nhân viên y tế. " + DISCLAIMER
        st["blocked"] = True
