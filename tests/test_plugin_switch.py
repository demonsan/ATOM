# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""In-place plugin recipe switch of a switchable boot (CPU only).

``plugin_host.stage_switch`` / ``commit_switch`` / ``rollback_switch``: what a
switch refuses, what it restores, what it releases and reloads; the engine's
drain (new requests held, the switch run once drained, answered late); the
recipe source a switch carries.
"""

import sys
import types

import pytest
import torch

from atom import plugin_host as host
from atom import plugin_recipe as recipes

ARCH = "DeepseekV41ForCausalLM"

_TOY = {
    "name": "toy.block.mix",
    "contract": 1,
    "kind": "region",
    "form": "bind",
    "inputs": ("x",),
    "kwargs": (),
    "returns": ("y",),
    "key": ("x",),
    "graph": "capture_safe",
    "target": "toy/model.py",
}


class _EntryPoint:
    def __init__(self, obj, module=None):
        self._obj, self.module = obj, module

    def load(self):
        return self._obj


@pytest.fixture(autouse=True)
def fresh_host(monkeypatch):
    saved = (dict(host._SEAM_SPECS), dict(host._SEAM_SOURCES))
    for name in [n for n in host._SEAM_SPECS if n.startswith("toy.")]:
        # toy seams other test files declared (and left) are not this model's
        del host._SEAM_SPECS[name]
        host._SEAM_SOURCES.pop(name, None)
    monkeypatch.setattr(host, "_aiter_entry_points", dict)
    monkeypatch.setattr(host, "_configure_aiter", lambda names: None)
    monkeypatch.setattr(host, "_graph_state", lambda: (False, None))
    monkeypatch.setattr(host, "_valid_rows", lambda rows: None)
    # A test provider's module is this file: never unload the tests (a reload
    # test names its own roots).
    monkeypatch.setattr(host, "_provider_roots", lambda name: ())
    host._reset_for_testing()
    yield
    host._reset_for_testing()
    for live, old in zip((host._SEAM_SPECS, host._SEAM_SOURCES), saved):
        live.clear()
        live.update(old)


def _plan(**fields):
    return recipes.resolve([{"schema": recipes.RECIPE_SCHEMA, "arch": ARCH, **fields}])


def _owner(name, value):
    def register(h):
        h.register(
            seam="toy.block.mix", contract=1, owner=name, impl=lambda x, **k: x + value
        )

    return register


def _providers(monkeypatch, **registers):
    monkeypatch.setattr(
        host,
        "_entry_points",
        lambda: {name: _EntryPoint(reg) for name, reg in registers.items()},
    )


def _boot(monkeypatch, plan, **registers):
    _providers(monkeypatch, **registers)
    host.configure(plan, kill_switch=False, switchable=True)
    return host.bind("toy.block.mix", lambda x: x, spec=_TOY)


# -- a switchable boot binds every seam ---------------------------------------------


def test_a_switchable_boot_routes_a_seam_no_recipe_owns_yet(monkeypatch):
    host.configure(_plan(), kill_switch=False, switchable=False)
    native = lambda x: x  # noqa: E731
    assert host.bind("toy.block.mix", native, spec=_TOY) is native
    host._reset_for_testing()
    site = _boot(monkeypatch, _plan(), a=_owner("a", 10))
    assert isinstance(site, host.Router) and site(1) == 1  # native, routed
    host.stage_switch(_plan(seams={"toy.block.mix": ["a"]}), device="cpu")
    host.commit_switch()
    assert site(1) == 11  # the call site built before the switch now routes


def test_switch_to_another_owner_and_back(monkeypatch):
    site = _boot(
        monkeypatch,
        _plan(seams={"toy.block.mix": ["a"]}),
        a=_owner("a", 1),
        b=_owner("b", 100),
    )
    assert site(1) == 2
    report = host.stage_switch(_plan(seams={"toy.block.mix": ["b"]}), device="cpu")
    assert report["added"] == ["b"] and report["removed"] == ["a"]
    assert not report["recompile"]
    assert site(1) == 101  # staged serves (the engine is drained meanwhile)
    host.commit_switch()
    assert host.status()["plan_id"] == report["plan_id"]
    host.stage_switch(_plan(seams={"toy.block.mix": ["a"]}), device="cpu")
    host.commit_switch()
    assert site(1) == 2


def test_rollback_restores_the_recipe_that_served(monkeypatch):
    plan_a = _plan(seams={"toy.block.mix": ["a"]})
    site = _boot(monkeypatch, plan_a, a=_owner("a", 1), b=_owner("b", 100))
    host.stage_switch(_plan(seams={"toy.block.mix": ["b"]}), device="cpu")
    host.rollback_switch()
    assert site(1) == 2 and host.status()["plan_id"] == plan_a.plan_id
    assert host._staged is None


@pytest.mark.parametrize(
    "plan, reason",
    [
        (
            dict(model_options={"target_model_cls": "pkg.Other"}),
            "model-differs",
        ),
        (dict(env={"ATOM_X": "1"}), "env-differs"),
        (dict(seams={"model.install": ["a"]}), "boot-hook-differs"),
    ],
)
def test_what_a_switch_refuses_changes_nothing(monkeypatch, plan, reason):
    plan_a = _plan(seams={"toy.block.mix": ["a"]})
    site = _boot(monkeypatch, plan_a, a=_owner("a", 1))
    with pytest.raises(host.SwitchRefused, match=reason):
        host.stage_switch(_plan(**plan), device="cpu")
    assert site(1) == 2 and host._staged is None


def test_a_server_not_booted_switchable_refuses():
    host.configure(_plan(), kill_switch=False, switchable=False)
    with pytest.raises(host.SwitchRefused, match="not-switchable"):
        host.stage_switch(_plan(), device="cpu")


def test_an_unknown_arm_and_a_second_stage_are_refused(monkeypatch):
    _boot(monkeypatch, _plan(seams={"toy.block.mix": ["a"]}), a=_owner("a", 1))
    with pytest.raises(host.SwitchRefused, match="recipe-unknown-arm"):
        host.stage_switch(_plan(), arm="nope", device="cpu")
    host.stage_switch(_plan(), device="cpu")
    with pytest.raises(host.SwitchRefused, match="switch-in-progress"):
        host.stage_switch(_plan(), device="cpu")


def test_a_provider_that_fails_rolls_back(monkeypatch):
    def broken(h):
        raise RuntimeError("bad build")

    site = _boot(
        monkeypatch, _plan(seams={"toy.block.mix": ["a"]}), a=_owner("a", 1), x=broken
    )
    with pytest.raises(host.SwitchRefused, match="recipe-plugin-drift"):
        host.stage_switch(_plan(seams={"toy.block.mix": ["x"]}), device="cpu")
    assert site(1) == 2


def test_a_prepare_that_raises_or_a_changed_weight_rolls_back(monkeypatch):
    module = types.ModuleType("toy_prep")
    module.prepare = lambda device: (_ for _ in ()).throw(RuntimeError("no kernel"))
    points = {
        "a": _EntryPoint(_owner("a", 1)),
        "p": _EntryPoint(_owner("p", 7), module="toy_prep"),
    }
    monkeypatch.setattr(host, "_entry_points", lambda: points)
    monkeypatch.setitem(sys.modules, "toy_prep", module)
    host.configure(_plan(seams={"toy.block.mix": ["a"]}), kill_switch=False, switchable=True)
    site = host.bind("toy.block.mix", lambda x: x, spec=_TOY)
    with pytest.raises(host.SwitchRefused, match="prepare"):
        host.stage_switch(_plan(seams={"toy.block.mix": ["p"]}), device="cpu")
    assert site(1) == 2
    module.prepare = lambda device: "ok"
    weight = torch.nn.Linear(2, 2)
    host.freeze_weights(weight)
    module.prepare = lambda device: weight.weight.data.add_(1.0)
    with pytest.raises(host.SwitchRefused, match="weights-changed"):
        host.stage_switch(_plan(seams={"toy.block.mix": ["p"]}), device="cpu")
    assert site(1) == 2


def test_commit_releases_the_outgoing_providers(monkeypatch):
    module = types.ModuleType("toy_rel")
    released = []
    module.release = lambda device: released.append(device)
    points = {
        "a": _EntryPoint(_owner("a", 1), module="toy_rel"),
        "b": _EntryPoint(_owner("b", 2)),
    }
    monkeypatch.setattr(host, "_entry_points", lambda: points)
    monkeypatch.setitem(sys.modules, "toy_rel", module)
    host.configure(_plan(seams={"toy.block.mix": ["a"]}), kill_switch=False, switchable=True)
    host.bind("toy.block.mix", lambda x: x, spec=_TOY)
    host.stage_switch(_plan(seams={"toy.block.mix": ["b"]}), device="gpu7")
    assert released == []  # nothing released before the ranks agreed
    report = host.commit_switch()
    assert released == ["gpu7"] and report["released"] == ["a"]


class _Mix(torch.nn.Module):
    def forward(self, x):
        return x * 2


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Mix(), _Mix()])


def _takeover(name, value):
    def register(h):
        h.register_module(
            slot="blocks.*", owner=name, factory=lambda m, cap: (lambda x: x + value)
        )

    return register


def test_module_takeovers_are_reinstalled_and_a_changed_set_recompiles(monkeypatch):
    _providers(monkeypatch, m=_takeover("m", 5), n=_takeover("n", 9))
    host.configure(_plan(), kill_switch=False, switchable=True)
    net = _Net()
    host.install_modules(net)
    x = torch.ones(1)
    assert torch.equal(net.blocks[0](x), x * 2) and host.compile_factor() == ()
    report = host.stage_switch(_plan(modules={"blocks.*": ["m"]}), device="cpu")
    assert report["recompile"] and report["modules_installed"] == ["blocks.0", "blocks.1"]
    host.commit_switch()
    assert torch.equal(net.blocks[1](x), x + 5)
    report = host.stage_switch(_plan(modules={"blocks.*": ["n"]}), device="cpu")
    assert not report["recompile"]  # same modules taken over: graphs re-record only
    host.rollback_switch()
    assert torch.equal(net.blocks[1](x), x + 5)  # the old router itself is back
    host.stage_switch(_plan(), device="cpu")
    host.commit_switch()
    assert "forward" not in net.blocks[0].__dict__ and torch.equal(net.blocks[0](x), x * 2)


def test_reload_imports_the_providers_code_again(tmp_path, monkeypatch):
    pkg = tmp_path / "toy_reload_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")

    def write(value):
        (pkg / "hook.py").write_text(
            "def register(h):\n"
            "    h.register(seam='toy.block.mix', contract=1, owner='r',\n"
            f"               impl=lambda x, **k: x + {value})\n"
        )

    write(3)
    monkeypatch.syspath_prepend(str(tmp_path))

    class _Point:
        module = "toy_reload_pkg.hook"

        def load(self):
            import importlib

            return importlib.import_module("toy_reload_pkg.hook").register

    monkeypatch.setattr(host, "_entry_points", lambda: {"r": _Point()})
    plan = _plan(seams={"toy.block.mix": ["r"]})
    host.configure(plan, kill_switch=False, switchable=True)
    site = host.bind("toy.block.mix", lambda x: x, spec=_TOY)
    assert site(1) == 4
    write(30)
    report = host.stage_switch(plan, device="cpu", reload=True, reload_roots=[str(pkg)])
    assert report["reloaded"] and report["unloaded_modules"] >= 1
    host.commit_switch()
    assert site(1) == 31
    for name in [n for n in sys.modules if n.startswith("toy_reload_pkg")]:
        del sys.modules[name]


def test_persistent_state_outlives_a_reload_and_a_rollback(tmp_path, monkeypatch):
    """``plugin_host.persistent(key)``: the same dict before a reloading switch (the provider's modules gone,
    a changed version imported), after a rolled-back one, and after a committed one."""
    pkg = tmp_path / "toy_persist_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")

    def write(value):
        (pkg / "hook.py").write_text(
            "from atom import plugin_host\n"
            "STORE = plugin_host.persistent('toy.persist')\n"
            f"STORE.setdefault('entries', []).append({value})\n"
            "def register(h):\n"
            "    h.register(seam='toy.block.mix', contract=1, owner='r',\n"
            f"               impl=lambda x, **k: x + {value})\n"
        )

    write(3)
    monkeypatch.syspath_prepend(str(tmp_path))

    class _Point:
        module = "toy_persist_pkg.hook"

        def load(self):
            import importlib

            return importlib.import_module("toy_persist_pkg.hook").register

    monkeypatch.setattr(host, "_entry_points", lambda: {"r": _Point()})
    plan = _plan(seams={"toy.block.mix": ["r"]})
    host._persistent.pop("toy.persist", None)
    host.configure(plan, kill_switch=False, switchable=True)
    site = host.bind("toy.block.mix", lambda x: x, spec=_TOY)
    store = host.persistent("toy.persist")
    assert store == {"entries": [3]} and sys.modules["toy_persist_pkg.hook"].STORE is store
    write(30)
    host.stage_switch(plan, device="cpu", reload=True, reload_roots=[str(pkg)])
    host.rollback_switch()  # the staged version is dropped, the persistent dict is not
    assert host.persistent("toy.persist") is store and store["entries"] == [3, 30]
    host.stage_switch(plan, device="cpu", reload=True, reload_roots=[str(pkg)])
    host.commit_switch()
    assert site(1) == 31
    assert host.persistent("toy.persist") is store and store["entries"] == [3, 30, 30]
    assert sys.modules["toy_persist_pkg.hook"].STORE is store
    for name in [n for n in sys.modules if n.startswith("toy_persist_pkg")]:
        del sys.modules[name]
    host._persistent.pop("toy.persist", None)


def test_persistent_keys_are_separate_and_must_be_named():
    a, b = host.persistent("toy.a"), host.persistent("toy.b")
    assert a is not b and host.persistent("toy.a") is a and a == {}
    with pytest.raises(ValueError):
        host.persistent("")
    host._persistent.pop("toy.a")
    host._persistent.pop("toy.b")


def test_a_switchable_compile_key_follows_the_takeovers_not_the_plan(monkeypatch):
    host.configure(_plan(seams={"toy.block.mix": []}), kill_switch=False, switchable=True)
    assert host.switchable() and host.compile_factor() == ()


# -- the recipe a switch carries --------------------------------------------------


def test_source_of_documents_sets_no_environment(monkeypatch):
    monkeypatch.delenv("ATOM_TOY_SWITCH", raising=False)
    doc = {"schema": recipes.RECIPE_SCHEMA, "arch": ARCH, "env": {"ATOM_TOY_SWITCH": "1"}}
    source, plan = recipes.source_of(documents=[doc])
    assert source.plan_id == plan.plan_id and source.resolve().plan_id == plan.plan_id
    import os

    assert "ATOM_TOY_SWITCH" not in os.environ
    with pytest.raises(recipes.RecipeRefused, match="recipe-unknown-arm"):
        recipes.source_of(documents=[doc], arm="nope")


# -- the switch on a running server: ATOM's own pause / resume, utility command, graph release ------


class _Core:
    """``core_mgr``: records the utility commands the switch sends; ``fail`` makes one raise."""

    def __init__(self, fail=None, resume_fails=False):
        self.calls, self.fail, self.resume_fails = [], fail or {}, resume_fails

    def broadcast_utility_command_sync(self, cmd, timeout=300.0, **kwargs):
        self.calls.append((cmd, kwargs))
        if cmd == "resume_generation" and self.resume_fails:
            raise RuntimeError("resume lost")
        if cmd in self.fail:
            raise self.fail[cmd]
        if cmd == "switch_plugin_recipe":
            return [{"cmd": cmd, "result": {"plan_id": kwargs["plan_id"], "recompile": False}}]
        return [{"cmd": cmd}]


def _engine(monkeypatch, core, *, dp=1, pp=1, switchable=True):
    if switchable:
        monkeypatch.setenv("ATOM_PLUGIN_SWITCHABLE", "1")
    else:
        monkeypatch.delenv("ATOM_PLUGIN_SWITCHABLE", raising=False)
    config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(data_parallel_size=dp), pipeline_parallel_size=pp
    )
    return types.SimpleNamespace(config=config, core_mgr=core)


def _source(**fields):
    source, _ = recipes.source_of(documents=[{"schema": recipes.RECIPE_SCHEMA, "arch": ARCH, **fields}])
    return source


def test_a_deployment_that_cannot_switch_refuses_before_pausing_anything(monkeypatch):
    core = _Core()
    for dp, pp, switchable, reason in ((1, 1, False, "not-switchable"), (2, 1, True, "dp-or-pp"), (1, 2, True, "dp-or-pp")):
        with pytest.raises(host.SwitchRefused) as refused:
            host.switch_serving_recipe(_engine(monkeypatch, core, dp=dp, pp=pp, switchable=switchable), _source())
        assert refused.value.reason == reason
    assert core.calls == []
    assert host.switch_unavailable(_engine(monkeypatch, core).config) is None


def test_the_engine_pauses_generation_switches_every_rank_then_resumes(monkeypatch):
    core = _Core()
    source = _source(seams={"toy.block.mix": []})
    reply = host.switch_serving_recipe(
        _engine(monkeypatch, core), source, arm="plugin", reload=True, reload_roots=["/x"], timeout=100.0
    )
    assert [c[0] for c in core.calls] == ["pause_generation", "switch_plugin_recipe", "resume_generation"]
    assert core.calls[0][1] == {"mode": "keep"}  # what runs keeps its KV and waits
    assert core.calls[1][1] == {
        "documents": list(source.documents),
        "plan_id": source.plan_id,
        "arm": "plugin",
        "reload": True,
        "reload_roots": ["/x"],
    }
    assert reply["result"]["plan_id"] == source.plan_id and reply["pause_seconds"] >= 0.0


@pytest.mark.parametrize(
    "error, expected",
    [
        (RuntimeError("TP rank 1: SwitchRefused: model-differs: the recipe builds another model"), "model-differs"),
        (RuntimeError("refused: drain-timeout: still running"), "drain-timeout"),
    ],
)
def test_a_refusal_is_a_switch_refused_and_generation_resumes(monkeypatch, error, expected):
    core = _Core(fail={"switch_plugin_recipe": error})
    with pytest.raises(host.SwitchRefused) as refused:
        host.switch_serving_recipe(_engine(monkeypatch, core), _source())
    assert refused.value.reason == expected
    assert [c[0] for c in core.calls][-1] == "resume_generation"


def test_a_timeout_or_other_failure_propagates_and_generation_still_resumes(monkeypatch):
    for error, raises in ((TimeoutError("no answer"), TimeoutError), (RuntimeError("boom"), RuntimeError)):
        core = _Core(fail={"switch_plugin_recipe": error})
        with pytest.raises(raises, match=str(error)):
            host.switch_serving_recipe(_engine(monkeypatch, core), _source())
        assert [c[0] for c in core.calls][-1] == "resume_generation"


def test_a_resume_that_fails_is_logged_loudly_and_does_not_mask_the_switch(monkeypatch, caplog):
    core = _Core(resume_fails=True)
    with caplog.at_level("CRITICAL", logger="atom"):
        reply = host.switch_serving_recipe(_engine(monkeypatch, core), _source())
    assert reply["result"]["recompile"] is False
    assert any("could not be resumed" in r.getMessage() for r in caplog.records)


class _Out:
    def __init__(self):
        self.sent = []

    def put_nowait(self, item):
        self.sent.append(item)


def test_the_engine_utility_command_runs_the_runner_method_on_every_rank():
    from atom.model_engine import engine_utility
    from atom.model_engine.engine_utility import PLUGIN_SWITCH_CMD, EngineUtilityHandler

    asked = []

    class _Mgr:
        proc_num = 4

        def utility_rpc(self, method, payload, timeout):
            asked.append((method, payload.args, timeout))
            return [types.SimpleNamespace(ok=True, value={"plan_id": "p"}, tp_rank=r) for r in range(4)]

    handler = object.__new__(EngineUtilityHandler)
    handler.runner_mgr, handler.output_queue, handler.label = _Mgr(), _Out(), "test"
    reply = handler._handle_switch_plugin_recipe(
        {"documents": ["{}"], "plan_id": "p", "arm": "plugin", "reload": 1, "reload_roots": ["/r"], "timeout": 50.0}
    )
    assert PLUGIN_SWITCH_CMD == "switch_plugin_recipe"
    assert EngineUtilityHandler._UTILITY_HANDLERS[PLUGIN_SWITCH_CMD] == "_handle_switch_plugin_recipe"
    assert asked == [("switch_plugin_recipe", (["{}"], "p", "plugin", True, ["/r"]), 50.0)]
    assert reply == {"cmd": PLUGIN_SWITCH_CMD, "result": {"plan_id": "p"}}
    assert handler.output_queue.sent == [("UTILITY_RESPONSE", reply)]
    assert engine_utility.PAUSE_GENERATION_CMD == "pause_generation"  # ATOM's own, unchanged


def test_the_runner_method_is_plugin_hosts_switch():
    """``ModelRunner`` cannot be imported without a GPU build, so its one method is read from the source: it
    only hands over to ``plugin_host.switch_runner`` (the switch has no engine code of its own)."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "atom/model_engine/model_runner.py").read_text()
    runner = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner")
    method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "switch_plugin_recipe")
    body = [n for n in method.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    assert ast.unparse(body[0].value) == "plugin_host.switch_runner(self, *args, **kwargs)"


class _Runner:
    """The runner ``switch_runner`` drives: what it needs of ModelRunner, and what it was asked."""

    def __init__(self, rollout=False, enforce_eager=False, capture_fails=False):
        self.config, self.device, self.label = types.SimpleNamespace(), "cuda:0", "ModelRunner0/4"
        self.enforce_eager, self.req_graph, self.log = enforce_eager, object(), []
        self._graphs_released_for_sleep = False
        self.capture_fails = capture_fails
        if rollout:
            self._recapture_cudagraphs_if_needed = lambda: self.log.append("recapture-rollout")

    def capture_cudagraph(self):
        self.log.append("capture")
        if self.capture_fails:
            raise RuntimeError("capture failed")


@pytest.fixture
def switch_world(monkeypatch):
    """Everything below ``switch_runner`` that needs a GPU, a TP group or the engine, as a log."""
    from atom.rollout import memory_manager

    log = []
    world = types.SimpleNamespace(log=log, outcomes=None, recompile=False, staged_error=None)
    monkeypatch.setattr(memory_manager, "release_cudagraphs", lambda r: (log.append("release"), setattr(r, "_graphs_released_for_sleep", True)))

    def stage(source, **kw):
        log.append(("stage", source.plan_id, kw["reload"]))
        if world.staged_error:
            raise world.staged_error

    monkeypatch.setattr(host, "stage_switch", stage)
    monkeypatch.setattr(host, "commit_switch", lambda: (log.append("commit"), {"recompile": world.recompile, "plan_id": "p2"})[1])
    monkeypatch.setattr(host, "rollback_switch", lambda: log.append("rollback"))
    monkeypatch.setattr(host, "_tp_gather", lambda v: world.outcomes if world.outcomes is not None else [v])
    monkeypatch.setattr(host, "_reset_compilation", lambda r: log.append("reset"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: log.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (10 << 30, 40 << 30))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 3 << 30)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 5 << 30)
    return world


def _switch(runner, **kw):
    return host.switch_runner(runner, ["{}"], "p2", "plugin", kw.get("reload", False), ())


def test_a_runner_switches_in_the_order_stage_release_commit_recapture(switch_world):
    runner = _Runner()
    report = _switch(runner, reload=True)
    assert switch_world.log == [("stage", "p2", True), "release", "commit", "empty_cache"]
    assert runner.log == ["capture"]  # the capture the engine runs at start, no copy of it
    assert report["plan_id"] == "p2" and report["graphs"] is True
    assert report["device_used_gib"] == 30.0 and report["memory_reserved_gib"] == 5.0
    assert report["switch_seconds"] >= report["recapture_seconds"] >= 0.0
    assert runner._graphs_released_for_sleep is False


def test_a_rollout_runner_recaptures_through_its_own_wake_up_path(switch_world):
    runner = _Runner(rollout=True)
    _switch(runner)
    assert runner.log == ["recapture-rollout"]


def test_changed_takeovers_reset_the_compiled_model_before_the_recapture(switch_world):
    switch_world.recompile = True
    runner = _Runner()
    _switch(runner)
    assert switch_world.log.index("commit") < switch_world.log.index("reset") < switch_world.log.index("empty_cache")


def test_a_failed_capture_serves_eagerly_and_an_eager_runner_captures_nothing(switch_world):
    runner = _Runner(capture_fails=True)
    report = _switch(runner)
    assert runner.enforce_eager is True and runner.req_graph is None and report["graphs"] is False
    eager = _Runner(enforce_eager=True)
    _switch(eager)
    assert eager.log == []


def test_a_rank_that_refused_rolls_the_others_back_and_nothing_is_released(switch_world):
    switch_world.outcomes = [None, ("model-differs", "another model"), None]
    runner = _Runner()
    with pytest.raises(host.SwitchRefused, match="TP rank 1: another model") as refused:
        _switch(runner)
    assert refused.value.reason == "model-differs"
    assert switch_world.log == [("stage", "p2", False), "rollback"]  # this rank staged: it rolls back; graphs untouched
    assert runner.log == []


def test_this_rank_refusing_has_nothing_to_roll_back(switch_world):
    switch_world.staged_error = host.SwitchRefused("recipe-unknown-arm", "no such arm")
    switch_world.outcomes = [("recipe-unknown-arm", "no such arm"), None]
    with pytest.raises(host.SwitchRefused):
        _switch(_Runner())
    assert "rollback" not in switch_world.log and "release" not in switch_world.log


def test_a_boot_hook_owner_is_never_released_or_reloaded(monkeypatch):
    boot = types.ModuleType("toy_boot")
    released = []
    boot.release = lambda device: released.append(device)

    def register_boot(h):
        h.register(
            seam=host.MODEL_INSTALL,
            contract=1,
            owner="boot",
            graph="eager",
            impl=lambda model, cfg, drafter, **k: model,
        )

    points = {
        "boot": _EntryPoint(register_boot, module="toy_boot"),
        "a": _EntryPoint(_owner("a", 1)),
    }
    monkeypatch.setattr(host, "_entry_points", lambda: points)
    monkeypatch.setitem(sys.modules, "toy_boot", boot)
    plan = _plan(seams={host.MODEL_INSTALL: ["boot"], "toy.block.mix": ["a"]})
    host.configure(plan, kill_switch=False, switchable=True)
    host.bind("toy.block.mix", lambda x: x, spec=_TOY)
    host.stage_switch(plan, device="gpu0", reload=True)
    report = host.commit_switch()
    assert released == [] and "boot" not in report["released"]
