from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
OUTPUT = PROJECT_ROOT / "index.html"


def records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict(orient="records")


def table_preview(path: Path, n: int = 12) -> list[dict]:
    return pd.read_csv(path).head(n).to_dict(orient="records")


def js_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_data() -> dict:
    clinical = records(RAW_DIR / "clinical_baseline.csv")
    qc = records(REPORT_DIR / "data_quality_report.csv")
    metrics = records(REPORT_DIR / "model_metrics.csv")
    predictions = records(REPORT_DIR / "sample_patient_predictions.csv")
    features = records(PROCESSED_DIR / "patient_features.csv")
    dictionary = records(REPORT_DIR / "data_dictionary.csv")
    experiments = records(REPORT_DIR / "model_experiments.csv")
    model_references = records(REPORT_DIR / "model_references.csv")
    model_design = json.loads((REPORT_DIR / "model_design.json").read_text(encoding="utf-8"))
    audit = json.loads((REPORT_DIR / "audit_manifest.json").read_text(encoding="utf-8"))
    cgm = pd.read_csv(RAW_DIR / "cgm_timeseries.csv")

    eligible = [row for row in features if bool(row.get("cgm_qc_pass"))]
    event_rate = sum(int(row["remission_12m"]) for row in eligible) / max(len(eligible), 1)
    center_counts = pd.DataFrame(clinical)["center"].value_counts().reset_index()
    center_counts.columns = ["center", "n"]

    sample_ids = [row["patient_id"] for row in predictions[:6]]
    cgm_sample = cgm[(cgm["patient_id"].isin(sample_ids)) & (cgm["sensor_day"] <= 3)].copy()
    cgm_sample = cgm_sample.fillna("")

    return {
        "summary": {
            "patients_total": len(features),
            "raw_cgm_rows": int(len(cgm)),
            "interval_minutes": int(audit.get("cgm_interval_minutes", 5)),
            "readings_per_day": int(audit.get("readings_per_day", 288)),
            "expected_effective_readings": int(audit.get("expected_effective_readings_per_patient", 3456)),
            "qc_pass": int(sum(bool(row.get("cgm_qc_pass")) for row in features)),
            "qc_excluded": int(sum(not bool(row.get("cgm_qc_pass")) for row in features)),
            "event_rate": round(event_rate * 100, 1),
            "train": int(sum(row["analysis_split"] == "train" and bool(row.get("cgm_qc_pass")) for row in features)),
            "internal": int(
                sum(row["analysis_split"] == "internal_validation" and bool(row.get("cgm_qc_pass")) for row in features)
            ),
            "external": int(
                sum(row["analysis_split"] == "external_validation" and bool(row.get("cgm_qc_pass")) for row in features)
            ),
        },
        "clinicalPreview": table_preview(RAW_DIR / "clinical_baseline.csv"),
        "cgmPreview": table_preview(RAW_DIR / "cgm_timeseries.csv"),
        "features": features,
        "qc": qc,
        "metrics": metrics,
        "predictions": predictions,
        "dictionary": dictionary,
        "experiments": experiments,
        "modelReferences": model_references,
        "modelDesign": model_design,
        "audit": audit,
        "centerCounts": center_counts.to_dict(orient="records"),
        "cgmSample": cgm_sample.to_dict(orient="records"),
    }


def write_dashboard(data: dict) -> None:
    payload = js_json(data)
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>三中心CGM糖尿病缓解预测PoC看板</title>
  <style>
    :root {{
      --ink: #14213d;
      --muted: #5d6778;
      --line: #d9e1ec;
      --panel: #ffffff;
      --bg: #f4f7fb;
      --blue: #2266d8;
      --teal: #0d9488;
      --orange: #d97706;
      --red: #b42318;
      --green: #15803d;
      --purple: #7c3aed;
      --shadow: 0 16px 40px rgba(20, 33, 61, .10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
    }}
    header {{
      background: linear-gradient(135deg, #10213f 0%, #164e63 58%, #0f766e 100%);
      color: white;
      padding: 32px 28px 26px;
    }}
    .wrap {{ max-width: 1240px; margin: 0 auto; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(300px, .8fr);
      gap: 24px;
      align-items: end;
    }}
    h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 44px); line-height: 1.12; letter-spacing: 0; }}
    .subtitle {{ max-width: 900px; margin: 0; color: rgba(255,255,255,.84); font-size: 17px; }}
    .hero-note {{
      border: 1px solid rgba(255,255,255,.22);
      background: rgba(255,255,255,.10);
      padding: 16px;
      border-radius: 8px;
      color: rgba(255,255,255,.90);
    }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(244,247,251,.94);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
    }}
    .tabs {{ display: flex; gap: 8px; overflow-x: auto; padding: 10px 28px; max-width: 1240px; margin: 0 auto; }}
    .tab {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      padding: 9px 13px;
      white-space: nowrap;
      cursor: pointer;
      font-size: 14px;
    }}
    .tab.active {{ background: var(--ink); color: white; border-color: var(--ink); }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 24px 28px 42px; }}
    section {{ display: none; }}
    section.active {{ display: block; }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .span-3 {{ grid-column: span 3; }}
    .span-4 {{ grid-column: span 4; }}
    .span-5 {{ grid-column: span 5; }}
    .span-6 {{ grid-column: span 6; }}
    .span-7 {{ grid-column: span 7; }}
    .span-8 {{ grid-column: span 8; }}
    .span-12 {{ grid-column: span 12; }}
    .kpi-label {{ color: var(--muted); font-size: 13px; }}
    .kpi-value {{ font-size: 31px; font-weight: 760; margin-top: 4px; letter-spacing: 0; }}
    .kpi-foot {{ color: var(--muted); font-size: 12px; margin-top: 5px; }}
    h2 {{ margin: 0 0 14px; font-size: 24px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 0 0 12px; }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 5px 9px;
      background: #eef6ff;
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 650;
    }}
    .pill.green {{ background: #e9f8ef; color: var(--green); }}
    .pill.orange {{ background: #fff4e5; color: var(--orange); }}
    .pill.red {{ background: #fff0ed; color: var(--red); }}
    .flow {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
    .flow-step {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fbfdff; min-height: 116px; }}
    .flow-step strong {{ display:block; font-size: 18px; margin-bottom: 6px; }}
    .pipeline {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }}
    .pipeline-step {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfdff; min-height: 140px; }}
    .pipeline-step b {{ display:block; margin-bottom: 6px; }}
    .bar-list {{ display: grid; gap: 10px; }}
    .bar-row {{ display: grid; grid-template-columns: 170px 1fr 64px; gap: 10px; align-items: center; font-size: 13px; }}
    .bar-track {{ height: 12px; background: #e8eef6; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--blue); border-radius: 999px; }}
    .bar-fill.teal {{ background: var(--teal); }}
    .bar-fill.orange {{ background: var(--orange); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: white; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf2f7; text-align: left; vertical-align: top; white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: #f7faff; color: #334155; font-weight: 720; }}
    tr:hover td {{ background: #fbfdff; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }}
    input, select {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      min-height: 38px;
      background: white;
      color: var(--ink);
      font: inherit;
    }}
    button, .link-btn {{
      border: 1px solid var(--ink);
      background: var(--ink);
      color: white;
      border-radius: 8px;
      padding: 9px 12px;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      font-size: 14px;
    }}
    .link-btn.secondary {{ background: white; color: var(--ink); border-color: var(--line); }}
    .chart {{ width: 100%; min-height: 260px; }}
    .chart svg {{ width: 100%; height: auto; display: block; }}
    .patient-grid {{ display: grid; grid-template-columns: minmax(260px, .7fr) minmax(0, 1.3fr); gap: 16px; }}
    .metric-strip {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
    .mini {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfdff; }}
    .mini b {{ display:block; font-size: 18px; margin-top: 3px; }}
    .audit-grid {{ display: grid; grid-template-columns: 210px 1fr; gap: 8px 14px; font-size: 13px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 5px; }}
    .warn {{ border-left: 4px solid var(--orange); padding: 12px 14px; background: #fff8ed; border-radius: 8px; }}
    .schema {{ background:#0f172a; color:#e2e8f0; border-radius:8px; padding:14px; overflow:auto; font-size:13px; }}
    @media (max-width: 1000px) {{
      .hero, .patient-grid {{ grid-template-columns: 1fr; }}
      .span-3, .span-4, .span-5, .span-6, .span-7, .span-8 {{ grid-column: span 12; }}
      .flow, .metric-strip, .pipeline {{ grid-template-columns: 1fr 1fr; }}
      .bar-row {{ grid-template-columns: 120px 1fr 48px; }}
    }}
    @media (max-width: 560px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .tabs {{ padding-left: 16px; padding-right: 16px; }}
      .flow, .metric-strip, .pipeline {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap hero">
      <div>
        <div class="pill green">5分钟CGM · 时序大模型PoC · 可审计</div>
        <h1>三中心CGM糖尿病缓解预测模型工程看板</h1>
        <p class="subtitle">新版PoC把CGM原始数据改为5分钟粒度，并把传统Logistic、传统ML、CGM人工统计特征、本课题CGM时序大模型和架构消融作为实验对比，而不是混合成一个模型。</p>
      </div>
      <div class="hero-note">
        <strong>给专家看的重点</strong>
        <p style="margin-top:8px">本页展示完整模型设计思路：输入格式、特征工程、神经网络架构、输出格式、相关CGM大模型借鉴点和实验验证路径。</p>
      </div>
    </div>
  </header>

  <nav aria-label="Dashboard tabs">
    <div class="tabs">
      <button class="tab active" data-tab="overview">总览</button>
      <button class="tab" data-tab="sources">数据源</button>
      <button class="tab" data-tab="quality">CGM质控</button>
      <button class="tab" data-tab="architecture">时序大模型设计</button>
      <button class="tab" data-tab="models">模型验证</button>
      <button class="tab" data-tab="patients">个体样例</button>
      <button class="tab" data-tab="audit">审计链</button>
    </div>
  </nav>

  <main>
    <section id="overview" class="active">
      <div class="grid">
        <div class="card span-3"><div class="kpi-label">模拟受试者</div><div class="kpi-value" id="kpiPatients"></div><div class="kpi-foot">三中心均衡分布</div></div>
        <div class="card span-3"><div class="kpi-label">CGM原始记录</div><div class="kpi-value" id="kpiRows"></div><div class="kpi-foot" id="kpiInterval"></div></div>
        <div class="card span-3"><div class="kpi-label">质控通过</div><div class="kpi-value" id="kpiQc"></div><div class="kpi-foot" id="kpiExpected"></div></div>
        <div class="card span-3"><div class="kpi-label">模拟12月缓解率</div><div class="kpi-value" id="kpiEvent"></div><div class="kpi-foot">仅用于PoC链路演示</div></div>
        <div class="card span-8">
          <h2>新版实验设计</h2>
          <div class="flow">
            <div class="flow-step"><strong>单指标基线</strong><span class="muted">HbA1c单指标，确认传统静态指标天花板。</span></div>
            <div class="flow-step"><strong>传统统计</strong><span class="muted">Logistic只用静态临床变量，作为可解释基线。</span></div>
            <div class="flow-step"><strong>传统ML</strong><span class="muted">表格人工特征与交互项，不读取原始时序。</span></div>
            <div class="flow-step"><strong>CGM时序模型</strong><span class="muted">本课题模型读取5分钟原始序列嵌入，单独对比。</span></div>
          </div>
        </div>
        <div class="card span-4">
          <h2>样本流转</h2>
          <div id="splitBars" class="bar-list"></div>
        </div>
        <div class="card span-6">
          <h2>三中心分布</h2>
          <div id="centerBars" class="bar-list"></div>
        </div>
        <div class="card span-6">
          <h2>现场可讲的一句话</h2>
          <p>本项目不是“把Logistic、机器学习和大模型混在一起”，而是先定义本课题CGM时序大模型，再用传统统计、传统ML、CGM摘要特征和架构消融证明它的增量价值。</p>
          <p class="warn">本页所有患者级数据均为模拟脱敏样例，不包含真实患者信息，不能用于临床诊疗。</p>
        </div>
      </div>
    </section>

    <section id="sources">
      <div class="grid">
        <div class="card span-12">
          <h2>数据源与下载</h2>
          <p class="muted">CGM原始时序表已改为5分钟粒度：每天288点，14天共4032点/人；剔除前2天后，每例理论有效读数3456点。</p>
          <div class="toolbar">
            <a class="link-btn" href="poc_cgm_remission/data/raw/clinical_baseline.csv">下载基线临床表</a>
            <a class="link-btn" href="poc_cgm_remission/data/raw/cgm_timeseries.csv">下载5分钟CGM原始时序表</a>
            <a class="link-btn" href="poc_cgm_remission/data/processed/patient_features.csv">下载分析特征表</a>
            <a class="link-btn secondary" href="poc_cgm_remission/reports/data_dictionary.csv">下载数据字典</a>
          </div>
        </div>
        <div class="card span-6"><h3>基线临床表预览</h3><div id="clinicalTable" class="table-wrap"></div></div>
        <div class="card span-6"><h3>5分钟CGM原始时序表预览</h3><div id="cgmTable" class="table-wrap"></div></div>
        <div class="card span-12"><h3>核心字段字典</h3><div id="dictionaryTable" class="table-wrap"></div></div>
      </div>
    </section>

    <section id="quality">
      <div class="grid">
        <div class="card span-5">
          <h2>CGM质控规则</h2>
          <p>PoC按申报书逻辑剔除佩戴前2天适应期，只分析后12天有效数据。新版粒度为5分钟，因此质量门禁更接近真实CGM工程。</p>
          <div class="flow">
            <div class="flow-step"><strong>5分钟原始导出</strong><span class="muted">14天连续血糖，每天288点。</span></div>
            <div class="flow-step"><strong>窗口锁定</strong><span class="muted">剔除前2天，保留12天。</span></div>
            <div class="flow-step"><strong>质量门禁</strong><span class="muted">有效读数≥90%，即约3111点以上。</span></div>
            <div class="flow-step"><strong>序列张量</strong><span class="muted">保留mask，不把缺失静默填平。</span></div>
          </div>
        </div>
        <div class="card span-7"><h2>每例CGM有效率</h2><div id="qcChart" class="chart"></div></div>
        <div class="card span-12">
          <div class="toolbar"><h3 style="margin:0">质控明细</h3><input id="qcSearch" placeholder="搜索 patient_id / center / reason" /></div>
          <div id="qcTable" class="table-wrap"></div>
        </div>
      </div>
    </section>

    <section id="architecture">
      <div class="grid">
        <div class="card span-12">
          <h2>本课题CGM时序大模型：具体技术方案</h2>
          <div id="architecturePipeline" class="pipeline"></div>
        </div>
        <div class="card span-6">
          <h3>输入数据格式</h3>
          <div id="inputFormatTable" class="table-wrap"></div>
        </div>
        <div class="card span-6">
          <h3>输出数据格式</h3>
          <pre class="schema" id="outputSchema"></pre>
        </div>
        <div class="card span-12">
          <h3>相关CGM时序大模型：关系与借鉴点</h3>
          <div id="referenceCards" class="grid"></div>
        </div>
        <div class="card span-12">
          <h3>训练与验证路径</h3>
          <div id="trainingPlan" class="flow"></div>
        </div>
      </div>
    </section>

    <section id="models">
      <div class="grid">
        <div class="card span-7">
          <h2>实验对比：AUC</h2>
          <div id="aucChart" class="chart"></div>
        </div>
        <div class="card span-5">
          <h2>模型验证页现在回答什么</h2>
          <p>这里展示的是实验对比：传统Logistic、传统ML、CGM摘要模型、本课题CGM时序大模型和架构消融。它们不是组合模型，而是为了证明“5分钟原始CGM时序表示”相对传统方案的增量。</p>
          <p class="warn">PoC用可复现Logistic头模拟不同输入/架构组；正式项目应替换为真实深度模型训练、预训练和外部验证。</p>
        </div>
        <div class="card span-12"><h3>实验组定义</h3><div id="experimentTable" class="table-wrap"></div></div>
        <div class="card span-12"><h3>模型指标明细</h3><div id="metricsTable" class="table-wrap"></div></div>
      </div>
    </section>

    <section id="patients">
      <div class="grid">
        <div class="card span-12">
          <div class="toolbar"><h2 style="margin:0">个体预测样例</h2><select id="patientSelect" aria-label="选择患者"></select></div>
          <div class="patient-grid">
            <div><div id="patientSummary" class="card" style="box-shadow:none"></div></div>
            <div>
              <div class="metric-strip" id="patientMetrics"></div>
              <div class="card" style="box-shadow:none; margin-top:16px"><h3>72小时5分钟CGM曲线样例</h3><div id="patientCgmChart" class="chart"></div></div>
            </div>
          </div>
        </div>
        <div class="card span-12"><h3>样例个体预测表</h3><div id="predictionTable" class="table-wrap"></div></div>
      </div>
    </section>

    <section id="audit">
      <div class="grid">
        <div class="card span-6"><h2>审计Manifest</h2><div id="auditSummary" class="audit-grid"></div></div>
        <div class="card span-6">
          <h2>可复跑命令</h2>
          <p><code>./run_demo.sh</code></p>
          <p><code>/Users/xuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 poc_cgm_remission/src/build_dashboard.py</code></p>
          <div class="toolbar"><a class="link-btn" href="poc_cgm_remission/reports/audit_manifest.json">查看完整manifest</a><a class="link-btn secondary" href="poc_cgm_remission/src/run_poc.py">查看PoC脚本</a></div>
        </div>
        <div class="card span-12"><h3>输出文件hash</h3><div id="hashTable" class="table-wrap"></div></div>
      </div>
    </section>
  </main>

  <script>
    const DATA = {payload};
    const fmt = new Intl.NumberFormat("zh-CN");
    const esc = value => String(value ?? "").replace(/[&<>"']/g, m => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#039;"}}[m]));
    const percent = value => `${{Number(value).toFixed(1)}}%`;
    function setText(id, text) {{ document.getElementById(id).textContent = text; }}
    function renderTable(el, rows, columns = null, maxRows = 80) {{
      const target = typeof el === "string" ? document.getElementById(el) : el;
      const view = rows.slice(0, maxRows);
      const cols = columns || Object.keys(view[0] || {{}});
      target.innerHTML = `<table><thead><tr>${{cols.map(c => `<th>${{esc(c)}}</th>`).join("")}}</tr></thead><tbody>${{
        view.map(r => `<tr>${{cols.map(c => `<td>${{esc(r[c])}}</td>`).join("")}}</tr>`).join("")
      }}</tbody></table>`;
    }}
    function renderBars(id, rows, labelKey, valueKey, colorClass = "") {{
      const max = Math.max(...rows.map(r => Number(r[valueKey]) || 0), 1);
      document.getElementById(id).innerHTML = rows.map(r => {{
        const value = Number(r[valueKey]) || 0;
        return `<div class="bar-row"><div>${{esc(r[labelKey])}}</div><div class="bar-track"><div class="bar-fill ${{colorClass}}" style="width:${{value / max * 100}}%"></div></div><b>${{fmt.format(value)}}</b></div>`;
      }}).join("");
    }}
    function renderQcChart() {{
      const rows = DATA.qc.map(r => ({{ id: r.patient_id, valid: Number(r.valid_pct), pass: String(r.qc_pass) === "True" || r.qc_pass === true }}));
      const width = 860, height = 280, pad = 34;
      const barW = Math.max(2, (width - pad * 2) / rows.length - 1);
      const bars = rows.map((r, i) => {{
        const h = Math.max(1, r.valid * (height - pad * 2));
        const x = pad + i * ((width - pad * 2) / rows.length);
        const y = height - pad - h;
        return `<rect x="${{x.toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{barW.toFixed(1)}}" height="${{h.toFixed(1)}}" fill="${{r.pass ? "#0d9488" : "#b42318"}}"><title>${{r.id}} · ${{percent(r.valid*100)}}</title></rect>`;
      }}).join("");
      const thresholdY = height - pad - .9 * (height - pad * 2);
      document.getElementById("qcChart").innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="CGM有效率图">
        <line x1="${{pad}}" y1="${{thresholdY}}" x2="${{width-pad}}" y2="${{thresholdY}}" stroke="#d97706" stroke-dasharray="5 5"/>
        <text x="${{pad}}" y="${{thresholdY-8}}" fill="#d97706" font-size="12">90%质控线</text>${{bars}}
        <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#94a3b8"/>
        <text x="${{pad}}" y="20" fill="#5d6778" font-size="12">绿色=通过，红色=剔除</text>
      </svg>`;
    }}
    function renderAucChart() {{
      const rows = DATA.metrics.filter(r => r.split !== "train");
      const models = DATA.experiments.map(e => e.model);
      const splits = ["internal_validation", "external_validation"];
      const labels = {{ internal_validation: "内部验证", external_validation: "外部验证" }};
      const width = 980, height = 360, pad = 52;
      const groupW = (width - pad * 2) / models.length;
      const colors = ["#2266d8", "#0d9488"];
      const bars = models.map((m, i) => splits.map((s, j) => {{
        const row = rows.find(r => r.model === m && r.split === s);
        const auc = Number(row?.auc || 0);
        const barW = 24;
        const x = pad + i * groupW + groupW / 2 - 30 + j * 32;
        const h = auc * (height - pad * 2);
        const y = height - pad - h;
        return `<rect x="${{x}}" y="${{y}}" width="${{barW}}" height="${{h}}" rx="4" fill="${{colors[j]}}"><title>${{m}} · ${{labels[s]}} AUC=${{auc}}</title></rect><text x="${{x+barW/2}}" y="${{y-6}}" text-anchor="middle" font-size="10" fill="#334155">${{auc.toFixed(3)}}</text>`;
      }}).join("") + `<text x="${{pad+i*groupW+groupW/2}}" y="${{height-19}}" text-anchor="middle" font-size="10" fill="#334155">${{DATA.experiments[i].experiment_id}}</text><text x="${{pad+i*groupW+groupW/2}}" y="${{height-6}}" text-anchor="middle" font-size="10" fill="#334155">${{m.slice(0,12)}}</text>`).join("");
      const yTicks = [0, .25, .5, .75, 1].map(v => {{
        const y = height - pad - v * (height - pad * 2);
        return `<line x1="${{pad}}" y1="${{y}}" x2="${{width-pad}}" y2="${{y}}" stroke="#edf2f7"/><text x="10" y="${{y+4}}" font-size="11" fill="#64748b">${{v.toFixed(2)}}</text>`;
      }}).join("");
      document.getElementById("aucChart").innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="AUC实验对照图">${{yTicks}}${{bars}}
        <rect x="${{width-178}}" y="16" width="12" height="12" fill="${{colors[0]}}"/><text x="${{width-160}}" y="27" font-size="12">内部验证</text>
        <rect x="${{width-96}}" y="16" width="12" height="12" fill="${{colors[1]}}"/><text x="${{width-78}}" y="27" font-size="12">外部验证</text>
      </svg>`;
    }}
    function renderArchitecture() {{
      const d = DATA.modelDesign;
      document.getElementById("architecturePipeline").innerHTML = d.architecture.map(step => `<div class="pipeline-step"><b>${{esc(step.stage)}}</b><div>${{esc(step.design)}}</div><p class="muted" style="margin-top:8px">${{esc(step.output)}}</p></div>`).join("");
      renderTable("inputFormatTable", d.input_format, ["component","shape","description"], 20);
      document.getElementById("outputSchema").textContent = JSON.stringify(d.output_schema, null, 2);
      document.getElementById("referenceCards").innerHTML = DATA.modelReferences.map(r => `<div class="card span-4" style="box-shadow:none"><h3>${{esc(r.model)}}</h3><p><strong>谁提出：</strong>${{esc(r.proposed_by)}}</p><p><strong>干什么：</strong>${{esc(r.does_what)}}</p><p><strong>和本PoC关系：</strong>${{esc(r.relation_to_poc)}}</p><p><strong>可借鉴：</strong>${{esc(r.borrowable)}}</p><a class="link-btn secondary" href="${{esc(r.source_url)}}" target="_blank" rel="noreferrer">来源</a></div>`).join("");
      document.getElementById("trainingPlan").innerHTML = d.training_plan.map((p, i) => `<div class="flow-step"><strong>阶段${{i+1}}</strong><span class="muted">${{esc(p)}}</span></div>`).join("");
    }}
    function renderPatient() {{
      const id = document.getElementById("patientSelect").value;
      const p = DATA.predictions.find(r => r.patient_id === id) || DATA.predictions[0];
      const feature = DATA.features.find(r => r.patient_id === id) || {{}};
      const riskClass = p.risk_band === "高缓解潜力" ? "green" : p.risk_band === "低缓解潜力" ? "red" : "orange";
      document.getElementById("patientSummary").innerHTML = `<h3>${{esc(p.patient_id)}} <span class="pill ${{riskClass}}">${{esc(p.risk_band)}}</span></h3>
        <p class="muted">${{esc(p.center)}} · ${{esc(p.analysis_split)}}</p>
        <p><strong>PoC干预提示：</strong>${{esc(p.poc_intervention_hint)}}</p>
        <p class="warn">这里只是模拟样例，不代表真实患者建议。</p>`;
      const metrics = [["HbA1c", feature.baseline_hba1c_pct, "%"],["BMI", feature.bmi, "kg/m²"],["TIR", feature.tir_3p9_10_pct, "%"],["餐后峰值", feature.seq_postprandial_peak_mean, "mmol/L"]];
      document.getElementById("patientMetrics").innerHTML = metrics.map(m => `<div class="mini">${{m[0]}}<b>${{esc(m[1])}}</b><span class="muted">${{m[2]}}</span></div>`).join("");
      renderCgmLine(id);
    }}
    function renderCgmLine(id) {{
      const rows = DATA.cgmSample.filter(r => r.patient_id === id && r.glucose_mmol_l !== "");
      const width = 840, height = 280, pad = 42;
      if (!rows.length) {{ document.getElementById("patientCgmChart").innerHTML = `<p class="muted">该患者未嵌入72小时曲线样例，可在原始CSV中查看。</p>`; return; }}
      const ys = rows.map(r => Number(r.glucose_mmol_l)).filter(Number.isFinite);
      const minY = Math.max(2.5, Math.min(...ys) - .5), maxY = Math.min(18, Math.max(...ys) + .5);
      const points = rows.map((r, i) => {{
        const x = pad + i / Math.max(rows.length - 1, 1) * (width - pad * 2);
        const y = height - pad - (Number(r.glucose_mmol_l) - minY) / (maxY - minY) * (height - pad * 2);
        return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
      }}).join(" ");
      const yFor = v => height - pad - (v - minY) / (maxY - minY) * (height - pad * 2);
      document.getElementById("patientCgmChart").innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}">
        <rect x="${{pad}}" y="${{yFor(10).toFixed(1)}}" width="${{width-pad*2}}" height="${{Math.max(0, yFor(3.9)-yFor(10)).toFixed(1)}}" fill="#e9f8ef"/>
        <line x1="${{pad}}" x2="${{width-pad}}" y1="${{yFor(3.9)}}" y2="${{yFor(3.9)}}" stroke="#15803d" stroke-dasharray="4 4"/>
        <line x1="${{pad}}" x2="${{width-pad}}" y1="${{yFor(10)}}" y2="${{yFor(10)}}" stroke="#d97706" stroke-dasharray="4 4"/>
        <polyline fill="none" stroke="#2266d8" stroke-width="1.8" points="${{points}}"/>
        <line x1="${{pad}}" y1="${{height-pad}}" x2="${{width-pad}}" y2="${{height-pad}}" stroke="#94a3b8"/>
        <text x="${{pad}}" y="22" font-size="12" fill="#64748b">绿色区间：3.9-10.0 mmol/L · 5分钟粒度</text>
      </svg>`;
    }}
    function init() {{
      const s = DATA.summary;
      setText("kpiPatients", fmt.format(s.patients_total));
      setText("kpiRows", fmt.format(s.raw_cgm_rows));
      setText("kpiInterval", `14天，${{s.interval_minutes}}分钟粒度，每天${{s.readings_per_day}}点`);
      setText("kpiQc", `${{fmt.format(s.qc_pass)}} / ${{fmt.format(s.patients_total)}}`);
      setText("kpiExpected", `后12天理论有效读数${{fmt.format(s.expected_effective_readings)}}点`);
      setText("kpiEvent", `${{s.event_rate}}%`);
      renderBars("splitBars", [{{ label: "训练集", n: s.train }},{{ label: "内部验证", n: s.internal }},{{ label: "外部验证", n: s.external }},{{ label: "质控剔除", n: s.qc_excluded }}], "label", "n", "teal");
      renderBars("centerBars", DATA.centerCounts, "center", "n", "orange");
      renderTable("clinicalTable", DATA.clinicalPreview);
      renderTable("cgmTable", DATA.cgmPreview);
      renderTable("dictionaryTable", DATA.dictionary);
      renderTable("experimentTable", DATA.experiments, ["experiment_id","model","method_type","data_input","answers","uses_raw_sequence"], 20);
      renderTable("metricsTable", DATA.metrics, ["experiment_id","model","method_type","data_input","split","n","event_rate","auc","threshold","sensitivity","specificity","accuracy","brier","auc_bootstrap_95ci"], 80);
      renderTable("predictionTable", DATA.predictions, ["patient_id","center","analysis_split","remission_12m","prob_E0","prob_E1","prob_E2","prob_E3","prob_E4","prob_E5","risk_band","poc_intervention_hint"], 80);
      renderTable("qcTable", DATA.qc, ["patient_id","center","analysis_split","valid_readings_after_day2","expected_readings_after_day2","valid_pct","qc_pass","exclusion_reason"], 140);
      renderQcChart();
      renderAucChart();
      renderArchitecture();
      const select = document.getElementById("patientSelect");
      select.innerHTML = DATA.predictions.slice(0, 30).map(p => `<option value="${{esc(p.patient_id)}}">${{esc(p.patient_id)}} · ${{esc(p.risk_band)}}</option>`).join("");
      select.addEventListener("change", renderPatient);
      renderPatient();
      const audit = DATA.audit;
      document.getElementById("auditSummary").innerHTML = [["运行时间", audit.run_at],["随机种子", audit.random_seed],["Python版本", audit.python],["CGM粒度", `${{audit.cgm_interval_minutes}}分钟`],["脚本hash", audit.script.sha256],["样本总数", audit.sample_flow.patients_total],["质控通过", audit.sample_flow.patients_qc_pass]].map(([k,v]) => `<strong>${{esc(k)}}</strong><span>${{esc(v)}}</span>`).join("");
      renderTable("hashTable", audit.outputs || [], ["path","sha256","bytes"], 30);
      document.getElementById("qcSearch").addEventListener("input", e => {{
        const q = e.target.value.trim().toLowerCase();
        const rows = DATA.qc.filter(r => Object.values(r).join(" ").toLowerCase().includes(q));
        renderTable("qcTable", rows, ["patient_id","center","analysis_split","valid_readings_after_day2","expected_readings_after_day2","valid_pct","qc_pass","exclusion_reason"], 140);
      }});
      document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {{
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll("main section").forEach(s => s.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.tab).classList.add("active");
      }}));
    }}
    init();
  </script>
</body>
</html>
"""
    OUTPUT.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


def main() -> None:
    write_dashboard(load_data())


if __name__ == "__main__":
    main()
