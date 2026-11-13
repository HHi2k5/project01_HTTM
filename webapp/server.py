"""API và trang web của demo. Chạy bằng start_webapp.bat rồi mở http://127.0.0.1:8000."""
import math
import re
import sys
import threading
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine as E

app = FastAPI(title="Trợ lý tiểu đường", docs_url=None, redoc_url=None)
coordinator = E.Coordinator()
lock = threading.Lock()  # Coordinator giữ bộ nhớ phiên dùng chung, xử lý từng lượt một

ACTION_LABEL = {"ORDER_HBA1C": "Xét nghiệm HbA1c", "ORDER_GLUCOSE": "Đo đường huyết lúc đói",
                "ADVISE_ROUTINE": "Tư vấn thường, khám định kỳ", "REFER_DOCTOR": "Chuyển khám bác sĩ"}
RULE_LABEL = {"R1": "Xét nghiệm đạt ngưỡng nên bắt buộc chuyển khám",
              "R2": "Đường huyết 126–199 mg/dL chưa rõ lúc đói nên yêu cầu HbA1c",
              "R3": "Từ 35 tuổi chưa có xét nghiệm nên không tư vấn thường",
              "FALLBACK": "Policy lỗi nên chuyển khám để an toàn"}
FACT = re.compile(r"^\((.+), ([^,]+), (.+)\) \[(\w+)\]$")


class Profile(BaseModel):
    gender: Optional[Literal["Male", "Female"]] = None
    age: Optional[float] = None
    bmi: Optional[float] = None
    hba1c: Optional[float] = None
    glucose: Optional[float] = None
    glucose_kind: Optional[Literal["fasting", "random", "unknown"]] = None
    hypertension: Optional[Literal[0, 1]] = None
    heart_disease: Optional[Literal[0, 1]] = None
    family_history: Optional[Literal[0, 1]] = None
    smoking: Optional[Literal["never", "former", "current"]] = None
    activity: Optional[Literal["Low"]] = None


class ProfileRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    profile: Profile


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=1000)


def clean_profile(profile):
    """Giữ các trường đã nhập và áp cùng giới hạn hợp lệ với Intake Agent."""
    data = profile.model_dump(exclude_none=True)
    for key, (lo, hi) in E.LIMITS.items():
        if key in data and not lo <= data[key] <= hi:
            raise HTTPException(422, f"{key} phải nằm trong khoảng {lo}–{hi}")
    if "glucose" not in data:
        data.pop("glucose_kind", None)
    return data


def fact_view(text):
    m = FACT.match(text)
    return dict(subject=m[1], relation=m[2].replace("_", " "), object=m[3], doc_id=m[4]) if m else dict(raw=text)


def turn_view(r):
    """Chuyển trạng thái của Coordinator thành JSON cho giao diện."""
    pr, de = r.get("prediction"), r.get("decision")
    prediction = decision = None
    if pr:
        prediction = dict(p=float(pr["p"]), p_text=E.pct(pr["p"]), threshold=float(pr["threshold"]), above=bool(pr["label"]),
                          scenario=E.SCEN_VI[pr["scenario"]], score=float(pr["score"]), tier=pr["tier"],
                          drivers=[dict(name=n, delta=float(d)) for n, d in pr["drivers"]],
                          what_if=[dict(name=n, delta=float(d)) for n, d in pr["what_if"]])
    if de:
        rules = str(de["rules"])
        decision = dict(action=de["action"], action_label=ACTION_LABEL[de["action"]], rl_action=de["rl_action"],
                        rl_label=ACTION_LABEL[de["rl_action"]], rule=rules, rule_label=RULE_LABEL.get(rules, ""),
                        lab_checks=[{**c, "value": float(c["value"]), "diagnostic": bool(c["diagnostic"])} for c in de["lab_checks"]])
    evidence = [dict(n=i, doc_id=e["doc_id"], title=e["title"], text=e["text"], source_type=e["source_type"], publisher=e["publisher"],
                     url=e["url"], locator=e.get("locator") or "", accessed=e["accessed"]) for i, e in enumerate(r["evidence"], 1)]
    return dict(text=r["text"], answer=r["answer"], intent=r["intent"], safety=r["safety"], path=r["path"], trace=r["trace"],
                tools=list(dict.fromkeys(r["tools"])), prediction=prediction, decision=decision, evidence=evidence,
                facts=[fact_view(f) for f in r["facts"][:6]], stale=r["stale"], expanded=bool(r["expanded"]), asked=bool(r["ask"]),
                critic_rounds=r["critic_rounds"], issues=[list(i) for i in r["issues"]], blocked=bool(r.get("blocked")),
                messages=r["messages"], failures=r["failures"], latency_ms=round(r["latency_ms"], 1), profile=r["profile"])


def records(name, **kwargs):
    frame = pd.read_csv(E.RES / f"{name}.csv", **kwargs)
    rows = frame.to_dict("records")
    return [{k: None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()} for row in rows]


@app.get("/api/status")
def status():
    return dict(documents=len(E.docs), entities=E.G.number_of_nodes(), relations=E.G.number_of_edges(), limits=E.LIMITS)


@app.post("/api/chat")
def chat(req: ChatRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "Câu hỏi đang trống")
    with lock:
        return turn_view(coordinator.ask(text, req.session_id))


@app.post("/api/profile")
def save_profile(req: ProfileRequest):
    data = clean_profile(req.profile)
    with lock:
        coordinator.sessions[req.session_id]["profile"] = data
    return dict(profile=data)


@app.delete("/api/session/{session_id}")
def drop_session(session_id: str):
    with lock:
        coordinator.sessions.pop(session_id, None)
    return dict(ok=True)


@app.get("/api/results")
def results():
    tests = records("tests")
    return dict(system=records("system_eval"), groups=records("system_eval_groups"), policy=records("rl_policy_results"),
                dianet=[r for r in records("dianet_results") if r["Mô hình"].startswith("DiaNet")], risknet=records("risknet_results"),
                retrieval=[r for r in records("retrieval_results") if r["k"] == 5], kb=records("kb_stats"),
                tests=dict(total=len(tests), passed=sum(t["Kết quả"] == "PASS" for t in tests)))


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/figures", StaticFiles(directory=E.RES / "figures"), name="figures")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    if "--open" in sys.argv:  # mô hình đã nạp xong ở đây, nên mở trình duyệt sau khi máy chủ lắng nghe
        threading.Timer(1, webbrowser.open, ["http://127.0.0.1:8000"]).start()
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định không in được tiếng Việt
    print("Web app: http://127.0.0.1:8000  (Ctrl+C để dừng)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
