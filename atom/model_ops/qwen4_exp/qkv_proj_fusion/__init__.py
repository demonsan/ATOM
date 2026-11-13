# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in use of aiter's routed ``qkv_proj_fusion`` op (contract v1).

One routed call replaces a Qwen QSA layer's pre-attention prologue -- the ``qkv_proj``
GEMM, ``qk_norm``, ``qsa_apply_mrope`` and ``_store_kv`` -- plus the indexer's
``project_qk``. aiter owns the op and routes it through ``aiter.plugin``; an enabled
out-of-tree provider (the fused FlyDSL gfx1250 kernel) serves the call classes it was
qualified on and aiter's native path serves everything else, so ATOM keeps no route or
fallback of its own.

Enablement follows the other provider-backed ops: name this provider in
``--aiter-provider-enable`` / ``ATOM_AITER_PROVIDER_ENABLE`` (``Config.aiter_provider_enable``);
while it is not named nothing here is imported and the layer keeps its own path.
"""

from .enablement import (  # noqa: F401
    OP_NAME,
    PROVIDER_NAME,
    QkvProjFusionSite,
    configure,
    decisions,
    enabled,
    prewarm,
    site_for,
)
