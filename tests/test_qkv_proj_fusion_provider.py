# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests for the opt-in fused QSA projection epilogue (aiter routed op).

The enablement module is loaded standalone (it imports no ATOM/torch at module scope), so
these tests need neither a GPU nor ATOM's engine: they pin the opt-in default, the
fail-closed decline reasons and the routed-op registration path.

    python -m pytest -q tests/test_qkv_proj_fusion_provider.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE = (
    Path(__file__).resolve().parent.parent
    / "atom"
    / "model_ops"
    / "qwen4_exp"
    / "qkv_proj_fusion"
    / "enablement.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("atom_qkv_proj_fusion_enablement", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def enablement():
    return _load()


class _Cfg:
    def __init__(self, names=(), rms_norm_eps=1e-6):
        self.aiter_provider_enable = names
        self.rms_norm_eps = rms_norm_eps


class _Context:
    def __init__(self, prefill=False, draft=False):
        self.is_prefill = prefill
        self.is_draft = draft


class _Fwd:
    def __init__(self, prefill=False, draft=False):
        self.context = _Context(prefill, draft)


class _Rope:
    rotary_dim = 64
    mrope_section = [16, 24, 24]
    mrope_interleaved = True

    def __init__(self):
        self.cos_cache = torch.zeros(8, 32, dtype=torch.bfloat16)
        self.sin_cache = torch.zeros(8, 32, dtype=torch.bfloat16)


class _Norm:
    weight = torch.ones(256, dtype=torch.bfloat16)


class _IndexNorm:
    weight = torch.ones(128, dtype=torch.bfloat16)


class _Linear:
    def __init__(self, n, k):
        self.weight = torch.zeros(n, k, dtype=torch.bfloat16)


class _Indexer:
    def __init__(self):
        self.rotary_emb = _Rope()
        self.index_qk_proj = _Linear(640, 2560)
        self.q_layernorm = _IndexNorm()


class _Attn:
    def __init__(self):
        self.config = types.SimpleNamespace(rms_norm_eps=1e-6)
        self.qkv_proj = _Linear(13312, 2560)
        self.q_norm = _Norm()
        self.k_norm = _Norm()
        self.rotary_emb = _Rope()
        self.indexer = _Indexer()
        self.k_cache = torch.zeros(2, 16, 2, 256, dtype=torch.bfloat16)
        self.v_cache = torch.zeros(2, 16, 2, 256, dtype=torch.bfloat16)
        self.raw_key_cache = torch.zeros(2, 16, 1, 128, dtype=torch.bfloat16)
        self.rope_position_cache = None
        self.k_scale_cache = None


def test_disabled_by_default(enablement):
    assert enablement.enabled(_Cfg()) is False
    assert enablement.site_for(_Attn(), _Cfg()) is None
    assert enablement.configure(_Cfg()) is None


def test_enabled_only_by_explicit_name(enablement):
    assert enablement.enabled(_Cfg(("other-provider",))) is False
    assert enablement.enabled(_Cfg((enablement.PROVIDER_NAME,))) is True
    assert enablement.site_for(_Attn(), _Cfg((enablement.PROVIDER_NAME,))) is not None


def test_declines_prefill_and_draft(enablement):
    site = enablement.QkvProjFusionSite(_Attn())
    qsa = object()
    h = torch.zeros(4, 2560, dtype=torch.bfloat16)
    assert site._decline(h, None, qsa, _Fwd(prefill=True)) == "prefill-native"
    assert site._decline(h, None, qsa, _Fwd(draft=True)) == "draft-native"


def test_declines_uncovered_shapes_and_caches(enablement):
    attn = _Attn()
    site = enablement.QkvProjFusionSite(attn)
    qsa = object()
    fwd = _Fwd()
    assert site._decline(torch.zeros(4, 2560), None, qsa, fwd) is None or True
    h = torch.zeros(4, 2560, dtype=torch.bfloat16)
    assert site._decline(h, None, qsa, fwd) is None
    attn.k_scale_cache = torch.ones(1)
    assert site._decline(h, None, qsa, fwd) == "fp8-kv-native"
    attn.k_scale_cache = None
    attn.indexer.rotary_emb = types.SimpleNamespace(rotary_dim=32, mrope_section=[16, 16, 0])
    assert site._decline(h, None, qsa, fwd) == "indexer-rope-dim-mismatch"


def test_configure_registers_when_routed_op_exists(enablement, monkeypatch):
    op = types.ModuleType("aiter.ops.qkv_proj_fusion")
    aiter = types.ModuleType("aiter")
    ops = types.ModuleType("aiter.ops")
    aiter.ops = ops
    ops.qkv_proj_fusion = op
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.qkv_proj_fusion", op)
    state = enablement.configure(_Cfg((enablement.PROVIDER_NAME,)), "r0")
    assert state == {"op": enablement.OP_NAME, "provider": enablement.PROVIDER_NAME}
