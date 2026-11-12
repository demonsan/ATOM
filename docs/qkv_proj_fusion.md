# Fused QSA projection epilogue (`qkv_proj_fusion`)

Opt-in use of aiter's routed op `qkv_proj_fusion` v1 by a Qwen3.8-Flash-Next QSA layer
(`atom/model_ops/qwen4_exp/qsa_attention.py`). One routed call replaces the four steps the
layer performs before its sparse attention — the `qkv_proj` GEMM, `qk_norm`, the partial
NeoX mRoPE and `_store_kv` — plus the indexer's `project_qk`.

## How it is enabled

Name the provider in the existing provider knob:

```bash
python -m atom.entrypoints.openai_server --model <model> \
  --aiter-provider-enable qkv-proj-fusion-provider
# or: ATOM_AITER_PROVIDER_ENABLE=qkv-proj-fusion-provider
```

`Config.aiter_provider_enable` is the only switch; the default (empty) imports nothing and
the layer keeps its own path. ATOM never enables a provider by environment and never sets
`AITER_PLUGINS`.

## Call flow

```
ATOM Qwen4ExpAttention.forward  (decode / verify, not prefill / draft)
  └─ qkv_proj_fusion.QkvProjFusionSite.prologue
       └─ aiter.ops.qkv_proj_fusion.qkv_proj_fusion(...)      AITER owns contract v1
            └─ aiter.plugin dispatch
                 ├─ enabled provider (fused FlyDSL gfx1250 kernel) → served
                 └─ decline / no provider / AITER_PLUGINS=0     → aiter native
```

A served call returns `(gate, query, index_q)` and has already written the K/V,
raw-index-K and optional rope-position caches, so the layer skips `_store_kv` and passes
`index_q` into `_select_tokens` (which then skips its own `project_qk` / raw-K store).
A decline returns `None` and the layer runs its stock path.

## What stays native

* prefill and the draft layer (the contract's `mode` has no prefill),
* FP8 KV caches (`k_scale_cache is not None`) — contract v1 is bf16,
* an indexer whose mRoPE shape differs from the attention's,
* any layer whose caches are not bound yet.

Provider-side failures are aiter's contract, not ATOM's: ATOM checks the conditions above
first, and a routed `run()` error propagates (aiter does not retry native, side effects must
not be repeated).

## Files

| path | role |
|---|---|
| `atom/model_ops/qwen4_exp/qkv_proj_fusion/enablement.py` | opt-in check, decline reasons, the routed call, `decisions()` diagnostics |
| `atom/model_ops/qwen4_exp/qkv_proj_fusion/__init__.py` | public surface (`site_for`, `enabled`, `configure`, `prewarm`) |
| `atom/aiter_fused_ops.py` | per-process `configure` / `prewarm` lifecycle for the enabled provider |
| `atom/model_ops/qwen4_exp/qsa_attention.py` | the call site (`forward` / `_select_tokens(index_q=...)`) |
| `tests/test_qkv_proj_fusion_provider.py` | CPU tests: opt-in default, decline reasons, registration path |

## Not claimed here

Accuracy and performance of the fused path (the provider's own evidence covers those; ATOM
records which path served through `decisions()`), capture-time behaviour of the provider's
first launch, and any non-gfx1250 arch.
