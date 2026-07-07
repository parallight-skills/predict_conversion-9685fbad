# Trial-to-Paid Conversion Skill

This repository contains a composable conversion-prediction skill bundle for
Lab 5.

The pipeline predicts whether a SaaS free-trial user converts to paid from six
standardized behavior features. It combines:

- full second-order feature engineering
- low-weight business-rule synthetic augmentation
- L2 logistic regression
- threshold calibration

The final submitted version reached 92.5% on the private leaderboard and ranked
#1 at submission time.

Run the pipeline through `pipeline.py` or the bundled MCP server.
