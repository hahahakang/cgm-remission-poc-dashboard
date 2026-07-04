# 三中心CGM糖尿病缓解预测PoC

这是一个用于专家沟通的静态交互看板和可复跑PoC。数据均为模拟脱敏样例，不含真实患者信息。

## 在线看板

打开 `index.html` 即可查看：

- 数据源预览与CSV下载
- CGM有效率质控
- 训练/内部验证/外部验证模型对照
- 个体预测样例与CGM曲线
- 审计manifest和输出hash

## 本地重跑

```bash
./run_demo.sh
```

重跑后会刷新 `poc_cgm_remission/reports/` 下的结果，并重新生成 `index.html`。

## 发布到 GitHub Pages

先在 GitHub 创建一个空仓库，然后运行：

```bash
./publish_to_github.sh https://github.com/USER/REPO.git
```

推送成功后，在仓库 `Settings -> Pages` 里选择 `Deploy from a branch`，分支选 `main`，目录选 `/root`。

## 注意

本PoC只证明工程链路可跑通，不能作为真实临床疗效或预测能力证据。正式研究应使用三中心脱敏真实数据、锁定统计分析计划和伦理/数据合规协议后重跑。
