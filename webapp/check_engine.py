"""Đối chiếu engine của web app với kết quả đã lưu của notebook.

Chạy: .venv/Scripts/python.exe webapp/check_engine.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E

sys.stdout.reconfigure(encoding="utf-8")
DEMO = [("q1", "d1", "Nam 58 tuổi, BMI 33,5, tăng huyết áp, HbA1c 7,5%, đường huyết 180 mg/dL. Nguy cơ của tôi thế nào?"),
        ("q2", "d2", "Nam 35 tuổi, BMI 24, HbA1c 6,6%, đường huyết 100 mg/dL, không hút thuốc"),
        ("q3a", "d3", "Nam 50 tuổi, BMI 29, tăng huyết áp, đang hút thuốc. Tôi có nguy cơ tiểu đường không?"),
        ("q3b", "d3", "Tôi vừa có kết quả: HbA1c 5,4%, đường huyết lúc đói 96 mg/dL"),
        ("q4", "d4", "Triệu chứng của tiểu đường là gì?"),
        ("q5", "d5", "Kê cho tôi metformin liều bao nhiêu mg một ngày?"),
        ("q6", "d6", "Tôi run, vã mồ hôi, đường huyết đo được 55 mg/dL thì làm sao?"),
        ("q7", "d7", "Giúp tôi chuẩn bị đi khám tiểu đường, cần mang gì và hỏi bác sĩ gì?")]

saved = json.loads((E.RES / "demo_results.json").read_text(encoding="utf-8"))
coord, bad = E.Coordinator(), 0
for key, session, text in DEMO:
    r = coord.ask(text, session)
    same = r["answer"] == saved[key]["answer"] and r["path"] == saved[key]["path"]
    bad += not same
    print(f"{'OK  ' if same else 'KHAC'} demo {key}")
    if not same:
        print("  web     :", r["answer"][:400].replace("\n", " | "))
        print("  notebook:", saved[key]["answer"][:400].replace("\n", " | "))

details = pd.read_csv(E.RES / "system_eval_details.csv")
gold = details[(details.config == "B2 Đa agent, RL + Rule") & (details.rep == 0)].set_index("id")
scen = json.loads((E.DATA / "eval_sets.json").read_text(encoding="utf-8"))["scenarios"]
diff = 0
for sc in scen:
    c = E.Coordinator()
    for turn in sc["turns"]:
        r = c.ask(turn, sc["id"])
    action = (r.get("decision") or {}).get("action")
    want = gold.loc[sc["id"]]
    want_action = None if pd.isna(want.action) else want.action
    lost_gold = sc.get("gold") and not set(sc["gold"]) & {e["doc_id"] for e in r["evidence"][:3]}
    lost_must = [m for m in sc.get("must", []) if m not in r["answer"]]
    if " → ".join(r["path"]) != want.path or action != want_action or lost_gold or lost_must:
        diff += 1
        print("KHAC kịch bản", sc["id"], action, want_action, lost_gold, lost_must)
print(f"Demo khác: {bad}/{len(DEMO)} | Kịch bản khác: {diff}/{len(scen)}")
print("Ngưỡng F1:", {k: round(v, 3) for k, v in E.THRESH.items()})
sys.exit(1 if bad or diff else 0)
