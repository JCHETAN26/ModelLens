"""Upstream risk model: train a real classifier on public credit-risk data and
turn its per-feature contributions into ModelLens risk cases.

This package is optional (requires the `ml` extra: scikit-learn, pandas) and is
deliberately isolated from the API serving path so the API image stays lean.
"""
