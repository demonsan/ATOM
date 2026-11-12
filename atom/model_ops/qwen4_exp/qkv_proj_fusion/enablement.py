# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in use of aiter's routed ``qkv_proj_fusion`` op by a Qwen QSA layer.

aiter defines and routes the op: ``aiter.ops.qkv_proj_fusion.qkv_proj_fusion``
(contract ``qkv_proj_fusion`` v1, ``aiter.plugin``). It serves every valid call --
an enabled provider (the fused FlyDSL gfx1250 kernel) on the classes it was
qualified on, aiter's native path otherwise -- so ATOM keeps no route, no
capability pinning and no fallback of its own: an enabled out-of-tree provider is
the only way the fused kernel runs, and a provider decline lands on aiter native.

The call site (:class:`QkvProjFusionSite`) replaces the layer's four pre-attention
steps -- ``qkv_proj``, ``qk_norm``, ``qsa_apply_mrope``, ``_store_kv`` -- and the
indexer's ``project_qk`` with one routed call. It is fail-closed on ATOM's side:
prefill, draft, absent/FP8 caches, a missing indexer rope or a hidden dtype the
contract does not cover keeps ATOM's own path. Nothing here is imported while
``Config.aiter_provider_enable`` does not name this provider.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom.qkv_proj_fusion")

#: the out-of-tree provider's ``aiter.plugin`` name
PROVIDER_NAME = "qkv-proj-fusion-provider"
OP_NAME = "qkv_proj_fusion"

_DECISIONS: dict[str, int] = {}
_LOGGED: set[str] = set()


def _note(reason: str) -> None:
    _DECISIONS[reason] = _DECISIONS.get(reason, 0) + 1
    if reason in _LOGGED:
        return
    _LOGGED.add(reason)
    if reason == "served":
        logger.info("QKV projection fusion: served")
    else:
        logger.info("QKV projection fusion: native path, first reason %s", reason)


def decisions() -> dict[str, int]:
    """Native-path reasons seen in this process (diagnostics; not a contract)."""

    return dict(_DECISIONS)


def enabled(config) -> bool:
    """``True`` only when the operator opted in by naming this provider."""

    names = tuple(getattr(config, "aiter_provider_enable", ()) or ())
    return PROVIDER_NAME in names


def configure(config, rank_name: str = "") -> dict | None:
    """Force the routed op's registration so ``aiter.plugin.explain`` can see it."""

    if not enabled(config):
        return None
    try:
        import aiter.ops.qkv_proj_fusion as op  # noqa: F401 - registers the route

        logger.info(
            "QKV projection fusion [%s]: provider %s requested (aiter routes op %s)",
            rank_name or "-",
            PROVIDER_NAME,
            OP_NAME,
        )
        return {"op": OP_NAME, "provider": PROVIDER_NAME}
    except Exception as exc:  # noqa: BLE001 - fail-open: ATOM keeps its own path
        logger.warning(
            "QKV projection fusion [%s]: routed op unavailable (%r); native path",
            rank_name or "-",
            exc,
        )
        return None


def prewarm(caches, rank_name: str = "") -> dict:
    """Provider warm-up hook (compile/load) after KV allocation, before capture.

    The routed contract has no tensor-state warm-up in v1: the provider compiles
    per call class on first use, which the eager warm-up forward absorbs. Kept as
    the lifecycle point a later AOT/prebuild step hooks into.
    """

    del caches
    return {"op": OP_NAME, "prewarmed": False, "rank": rank_name or "-"}


def _sections(rope) -> tuple[int, int, int]:
    value = getattr(rope, "mrope_section", None) or [
        int(getattr(rope, "rotary_dim", 0)) // 2,
        0,
        0,
    ]
    t, h, w = (int(v) for v in value[:3])
    return t, h, w


class QkvProjFusionSite:
    """Per-layer fused prologue; declines (returns ``None``) keep ATOM's path."""

    def __init__(self, attn):
        self.attn = attn
        self.eps = float(getattr(getattr(attn, "config", None), "rms_norm_eps", 1e-6))

    # -- coverage -----------------------------------------------------------
    def _decline(self, hidden_states, positions, qsa, forward_context) -> str | None:
        a = self.attn
        context = forward_context.context
        if context.is_prefill:
            return "prefill-native"
        if context.is_draft:
            return "draft-native"
        if qsa is None:
            return "no-qsa-metadata"
        if hidden_states.dtype is not __import__("torch").bfloat16:
            return f"hidden-dtype-{hidden_states.dtype}"
        if getattr(a, "k_scale_cache", None) is not None:
            return "fp8-kv-native"
        if getattr(a, "k_cache", None) is None or a.k_cache.dtype is not hidden_states.dtype:
            return "cache-unbound-or-not-bf16"
        if getattr(a, "raw_key_cache", None) is None:
            return "raw-k-cache-unbound"
        rope = getattr(a, "rotary_emb", None)
        irope = getattr(getattr(a, "indexer", None), "rotary_emb", None)
        if rope is None:
            return "no-rope"
        if getattr(rope, "rotary_dim", None) != getattr(irope, "rotary_dim", None):
            return "indexer-rope-dim-mismatch"
        return None

    # -- call ---------------------------------------------------------------
    def prologue(self, hidden_states, positions, qsa):
        """Return ``(gate, query, index_q)`` served by aiter, or ``None`` to stay native.

        The routed call writes the K/V, raw-index-K and rope-position caches itself;
        the caller must not call ``_store_kv`` after a served call, and passes
        ``index_q`` into ``_select_tokens`` so the indexer skips its own ``project_qk``.
        A served call's exceptions propagate (aiter does not retry native).
        """

        from atom.utils.forward_context import get_forward_context

        forward_context = get_forward_context()
        if "prologue-called" not in _LOGGED:
            _LOGGED.add("prologue-called")
            logger.info(
                "QKV projection fusion: prologue called (tokens=%d prefill=%s)",
                int(hidden_states.shape[0]),
                bool(forward_context.context.is_prefill),
            )
        why = self._decline(hidden_states, positions, qsa, forward_context)
        if why is not None:
            _note(why)
            return None
        a = self.attn
        rope = a.rotary_emb
        irope = a.indexer.rotary_emb
        section_t, section_h, section_w = _sections(rope)
        if (section_t, section_h, section_w) != _sections(irope):
            _note("indexer-rope-section-mismatch")
            return None
        try:
            from aiter.ops.qkv_proj_fusion import qkv_proj_fusion as _op

            # The contract takes the half-width tables the kernel indexes by position:
            # ATOM keeps them as [max_pos, 1, 1, rotary_dim // 2] (bf16, contiguous).
            cos_table = rope.cos_cache.reshape(rope.cos_cache.shape[0], -1)
            sin_table = rope.sin_cache.reshape(rope.sin_cache.shape[0], -1)
            gate, query, index_q = _op(
                hidden_states,
                a.qkv_proj.weight,
                a.indexer.index_qk_proj.weight,
                a.q_norm.weight,
                a.k_norm.weight,
                a.indexer.q_layernorm.weight,
                cos_table,
                sin_table,
                positions,
                qsa.slot_mapping,
                a.k_cache,
                a.v_cache,
                a.raw_key_cache,
                pos_cache=a.rope_position_cache,
                rotary_dim=int(rope.rotary_dim),
                mrope_section_t=section_t,
                mrope_section_h=section_h,
                mrope_section_w=section_w,
                interleaved=bool(getattr(rope, "mrope_interleaved", False)),
                eps=self.eps,
                mode="decode",
            )
        except Exception as exc:  # noqa: BLE001 - fail-open before the layer's own writes
            _note(f"routed-call-{type(exc).__name__}")
            logger.warning(
                "QKV projection fusion: routed call failed (%r); native path", exc
            )
            return None
        _note("served")
        return gate, query, index_q


def site_for(attn, config) -> QkvProjFusionSite | None:
    return QkvProjFusionSite(attn) if enabled(config) else None


__all__ = [
    "OP_NAME",
    "PROVIDER_NAME",
    "QkvProjFusionSite",
    "configure",
    "decisions",
    "enabled",
    "prewarm",
    "site_for",
]
