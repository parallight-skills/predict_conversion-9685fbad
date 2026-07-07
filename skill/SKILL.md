---
name: conversion-skills
description: 2026年7月7日，分数 92.5%，Top-1。second-order features, low-weight synthetic augmentation, L2 logistic training, and threshold calibration
---

# conversion-skills

This skill bundle predicts whether a SaaS free-trial user will convert to paid.
It exposes a composable pipeline:

`raw 6 features -> feature_engineer -> train_model -> calibrate -> 0/1 labels`

## When to use

Use this bundle when the input is a matrix of standardized free-trial behavior
features with these six columns:

1. `weekly_active_sessions`
2. `support_tickets_opened`
3. `product_features_explored`
4. `first_value_depth`
5. `days_idle_before_first_action`
6. `referral_invites_sent`

The output is a list of `0/1` conversion labels.

## What the pipeline does

### 1. Feature engineering

`feature_engineer.py` expands each row from 6 dimensions to 27 dimensions:

- 6 raw features
- 6 squared terms
- 15 pairwise interactions

The main signal is nonlinear. Raw linear modeling underfits; second-order
features expose the hidden conversion rule to a linear model.

### 2. Model training

`train_model.py` trains a hand-written L2 logistic model on the engineered
features. It also adds a small amount of synthetic training data generated from
the observed train distribution.

Synthetic augmentation details:

- 400 synthetic raw samples
- bootstrap from real train rows
- small covariance-shaped noise, clipped to the 1%-99% train range
- labels from an inferred business rule
- synthetic sample weight = 0.2

The synthetic data is intentionally low-weight. It nudges the model toward the
likely generator without overpowering the real labels.

### 3. Calibration

`calibrate.py` chooses the threshold that maximizes training accuracy on the
model probabilities, then applies that threshold to test probabilities.

## Inferred business rule

The useful conversion signals are mostly second-order:

- active usage plus support seeking: `sessions * tickets`
- product exploration plus value depth: `features * depth`
- team/referral momentum: `features * invites`, `tickets * invites`
- delayed activation penalty: `idle^2`

This rule was inferred from training-only experiments and local cross-checks.
The pipeline does not read hidden test labels.
