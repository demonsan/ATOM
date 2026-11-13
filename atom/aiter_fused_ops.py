# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-process lifecycle of the opt-in aiter ops served through providers.

aiter owns each op and routes it (``aiter.plugin``); an enabled out-of-tree provider
serves the classes it was qualified on and aiter's native path serves the rest, so ATOM
keeps no route and no fallback of its own. This module only times the setup:

* :func:`configure_aiter_fused_ops` -- called by ``ModelRunner.__init__`` after
  ``_setup_device_and_distributed()``, before the model is built, warmed up or captured.
* :func:`prewarm_aiter_fused_ops` -- before CUDA-graph capture: providers compile / load
  what the bound caches can select.

Both never raise: an aiter without the interface, or a missing provider, is a log line
and the caller keeps ATOM's own path.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom")

OP_NAME = "qkv_proj_fusion"
PROVIDER_NAME = "qkv-proj-fusion-provider"


def _active(config) -> bool:
    return bool(getattr(config, "aiter_provider_enable", ()))


def configure_aiter_fused_ops(config, rank_name: str = "") -> dict:
    """Configure every enabled opt-in op in this process. Never raises."""

    out: dict = {}
    if not _active(config):
        return out
    from atom.aiter_provider import configure_aiter_providers

    out["aiter_providers"] = configure_aiter_providers(config, rank_name)
    try:
        from atom.model_ops.qwen4_exp.qkv_proj_fusion import configure as configure_op

        out[OP_NAME] = configure_op(config, rank_name)
    except Exception as exc:  # noqa: BLE001 - fail-open: ATOM's prologue
        logger.warning(
            "QKV projection fusion [%s]: configure failed (%r); native path",
            rank_name or "-",
            exc,
        )
        out[OP_NAME] = None
    return out


def prewarm_aiter_fused_ops(config, models, rank_name: str = "") -> dict:
    """Compile every class the CUDA-graph capture will replay. Never raises."""

    out: dict = {}
    if not _active(config):
        return out
    try:
        sizes = list(
            getattr(getattr(config, "compilation_config", None), "cudagraph_capture_sizes", ())
            or ()
        )
        sites = []
        for model in models:
            for module in model.modules():
                site = getattr(module, "qkv_proj_fusion", None)
                if site is not None:
                    sites.append(site)
        warmed = 0
        for site in sites:
            warmed += int(site.warmup(sizes))
        out[OP_NAME] = {"sites": len(sites), "sizes_warmed": warmed}
        logger.info(
            "QKV projection fusion [%s]: pre-capture warmup %s",
            rank_name or "-",
            out[OP_NAME],
        )
    except Exception as exc:  # noqa: BLE001 - first capture class then stays native
        logger.warning(
            "QKV projection fusion [%s]: pre-capture warmup failed (%r)",
            rank_name or "-",
            exc,
        )
        out[OP_NAME] = None
    return out
