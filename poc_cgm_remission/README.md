# CGM糖尿病缓解预测小样例PoC

本目录是给三中心CGM真实世界队列项目准备的可执行小样例。它使用模拟脱敏数据，演示从原始CGM时序数据到可审计模型报告的最小闭环。

## 运行方式

推荐在项目根目录运行：

```bash
./run_demo.sh
```

也可以直接运行脚本：

```bash
/Users/xuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 poc_cgm_remission/src/run_poc.py
```

注意：macOS 自带的 `python3` 通常没有安装本PoC需要的 `numpy`、`pandas`，所以建议使用上面的 `./run_demo.sh` 或 bundled Python 命令。

## 生成内容

- `../../index.html`：可直接打开或部署到 GitHub Pages 的专家交互看板。
- `data/raw/clinical_baseline.csv`：模拟脱敏基线临床表。
- `data/raw/cgm_timeseries.csv`：模拟脱敏14天CGM原始时序表。
- `data/processed/patient_features.csv`：按受试者聚合后的临床+CGM特征表。
- `reports/data_quality_report.csv`：CGM有效率、剔除原因、中心分布等质控结果。
- `reports/model_metrics.csv`：静态临床模型、CGM模型、融合模型在训练/内部验证/外部验证中的指标。
- `reports/sample_patient_predictions.csv`：样例个体预测概率、风险分层和干预建议。
- `reports/audit_manifest.json`：脚本、输入、输出的hash、运行时间、随机种子、样本流转记录。
- `reports/poc_summary.md`：一页式中文汇报摘要。

## PoC边界

这个PoC用于快速建立工程信任，不用于临床诊疗。生产项目应替换为真实三中心脱敏数据、正式SAP、EDC/数据库权限控制、CRF/数据字典、统计程序锁定、模型版本管理、伦理与数据处理协议。
