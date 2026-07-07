# Trial-to-Paid Conversion Skill

This repository contains a composable conversion-prediction skill bundle for
Lab 5.

The pipeline predicts whether a SaaS free-trial user converts to paid from six
standardized behavior features. It combines:

- full second-order feature engineering
- low-weight business-rule synthetic augmentation
- L2 logistic regression
- threshold calibration

2026年7月7日，分数 92.5%，Top-1。second-order features, low-weight
synthetic augmentation, L2 logistic training, and threshold calibration.

Run the pipeline through `pipeline.py` or the bundled MCP server.
