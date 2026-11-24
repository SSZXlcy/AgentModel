"""Run report (W8): one Markdown file assembled from a run's own records."""
from __future__ import annotations

import json
from pathlib import Path

from .recorder import read_jsonl

KIND_TEXT = {"tightening": "收紧，立即生效", "meta": "节奏参数，立即生效", "mode": "切换节奏，观察期判定",
             "improvement": "验证通过"}


def _fmt(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def build(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    transitions = read_jsonl(run_dir / "transitions.jsonl")
    journal = read_jsonl(run_dir / "journal.jsonl")
    rounds = {r["round"]: r for r in journal if r.get("type") == "round"}
    for row in journal:
        if row.get("type") == "backfill" and row["round"] in rounds:
            rounds[row["round"]].setdefault("outcome", {}).update(row["outcome"])
    sv = summary.get("supervisor") or {}
    headline = "Macro-F1（越高越好）" if summary["task"] == "A" or summary["mean_utility"] >= 0 else "RMSE（越低越好，表中为负 RMSE）"
    lines = [f"# 运行报告：任务 {summary['task']}", "",
             f"- 运行目录：`{summary['run_dir']}`", f"- 答案协议：`{summary.get('label_protocol')}`",
             f"- 最终代码版本：`{summary['code_version']}`；策略版本数：{summary.get('strategy_versions', 0)}",
             f"- 大模型：{(sv.get('llm') or summary.get('agent') or {}).get('provider', 'local')}", "",
             "## 一、结果", "", f"指标：{headline}", "",
             "| | 本系统 | 官方模型不更新 |", "|---|---|---|",
             f"| 平均 | {_fmt(summary['mean_utility'])} | {_fmt(summary['static_mean_utility'])} |",
             f"| 最差一批 | {_fmt(summary['worst_utility'])} | {_fmt(summary['static_worst_utility'])} |", "",
             f"与不更新相比：{_fmt(summary['gain_vs_static'])}；模型晋升 {summary['promotions']} 次，回退 "
             f"{summary['rollbacks']} 次，退回基线 {summary['resets_to_static']} 次，报错 {summary['errors']} 次。", "",
             "## 二、成本", "",
             f"- 训练量（行×轮）：{summary.get('train_row_epochs')}",
             f"- 训练秒数：{_fmt(summary['cost']['train_seconds'], 2)}",
             f"- 标注：{summary['cost']['labels']}", f"- 大模型 Token：{summary['cost']['llm_tokens']}", ""]
    if sv:
        tiers = sv.get("tiers", {})
        lines += ["## 三、何时延续、何时介入", "",
                  f"延续 {tiers.get('continue', 0)} 次，复查 {tiers.get('review', 0)} 次，更新 {tiers.get('update', 0)} 次。"
                  f"触发次数：{json.dumps(sv.get('triggers', {}), ensure_ascii=False)}。"
                  f"观察期结果：{json.dumps(sv.get('outcomes', {}), ensure_ascii=False)}。", "",
                  "| 批次 | 档位 | 触发 | 来源 | 结果 | Token |", "|---|---|---|---|---|---|"]
        for item in sv.get("interventions", []):
            result = item.get("decision") or ("生效 " + ",".join(item.get("applied", [])) if item.get("applied")
                                              else "无修正生效")
            lines.append(f"| {item.get('window')} | {item.get('tier')} | {item.get('trigger') or ','.join(item.get('triggers', []))}"
                         f" | {item.get('source')} | {result} | {item.get('tokens', 0)} |")
        lines.append("")
    updates = [t for t in transitions if t["type"] == "strategy_update"]
    lines += ["## 四、策略更新时间线", ""]
    if not updates:
        lines.append("本次运行没有策略更新。")
    for u in updates:
        outcome = (rounds.get(u.get("round")) or {}).get("outcome", {}).get("probation")
        changes = ", ".join(f"`{k}`={json.dumps(v, ensure_ascii=False)}" for k, v in (u.get("policy_changes") or {}).items())
        code = "" if u["code_version_before"] == u["code_version_after"] else \
            f"；代码 `{u['code_version_before']}` → `{u['code_version_after']}`"
        lines.append(f"- **版本 {u['version']}**（第 {u['effective_from_window']} 批起，来源 {u['source']}）："
                     f"{changes or '无参数变化'}{code}。理由：{u.get('reason')}。观察期：{outcome or '未结束/不适用'}")
    lines += ["", "## 五、每轮分析与总结", ""]
    for r in sorted(rounds):
        row = rounds[r]
        paragraph = (row.get("summary") or {}).get("report_paragraph")
        if paragraph:
            lines += [f"### 第 {r} 轮（{row.get('trigger_meaning') or row.get('trigger')}，来源 {row.get('source')}）", "",
                      paragraph, ""]
            for f in (row.get("plan") or {}).get("findings") or []:
                lines.append(f"- 发现：{f.get('claim')}（依据：{f.get('evidence')}）")
            for e in row.get("experiments") or []:
                res = e["result"]
                what = e.get("operator") or json.dumps(e.get("change") or e.get("edits"), ensure_ascii=False)
                params = json.dumps({k: v for k, v in (e.get("params") or {}).items() if k != "source"}, ensure_ascii=False)
                lines.append(f"- 实验 `{what}` {params}：{res.get('mode')}，差值 {_fmt(res.get('mean_diff'))}，"
                             f"波动范围 {_fmt(res.get('band'))}，n={res.get('n')} → {'通过' if e.get('passed') else '未通过'}"
                             f"{'（规则补充）' if e.get('by') == 'rule' else ''}")
            for a in row.get("applied") or []:
                lines.append(f"- 生效：`{a.get('operator')}` {json.dumps(a.get('change') or {}, ensure_ascii=False)}"
                             f"（{KIND_TEXT.get(a.get('kind'), a.get('kind'))}{'，依据经验 ' + a['experience'] if a.get('experience') else ''}）")
            for x in row.get("rejected") or []:
                lines.append(f"- 未采纳：`{x.get('operator')}`：{x.get('reason')}")
            lines.append("")
    lessons = [l for r in rounds.values() for l in (r.get("summary") or {}).get("lessons") or []]
    lines += ["## 六、经验", ""] + ([f"- **{l.get('cue')}**：{l.get('lesson')}（{l.get('evidence')}；把握：{l.get('confidence')}）"
                                    for l in lessons] or ["本次运行没有沉淀出有充分证据的经验。"])
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
