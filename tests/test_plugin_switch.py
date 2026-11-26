# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""In-place plugin recipe switch of a switchable boot (CPU only).

``plugin_host.stage_switch`` / ``commit_switch`` / ``rollback_switch``: what a
switch refuses, what it restores, what it releases and reloads; the engine's
drain (new requests held, the switch run once drained, answered late); the
recipe source a switch carries.
"""

import sys
import time
import types

import pytest
import torch

from atom import plugin_host as host
from atom import plugin_recipe as recipes
from atom.model_engine import engine_core

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


# -- the engine drains, then switches ---------------------------------------------


class _Scheduler:
    def __init__(self):
        self.running, self.got = 1, []

    def is_finished(self):
        return self.running == 0

    def extend(self, seqs):
        self.got.extend(seqs)


class _Handler:
    def __init__(self):
        self.answers, self.ran = [], []

    def run_plugin_switch(self, args, *, drain_seconds):
        self.ran.append(args["plan_id"])
        return {"cmd": "switch_plugin_recipe", "result": {"plan_id": args["plan_id"]}}

    def answer(self, request_id, reply):
        self.answers.append((request_id, reply))


def _engine(unavailable=None):
    engine = object.__new__(engine_core.EngineCore)
    engine.label = "test"
    engine.scheduler = _Scheduler()
    engine.utility_handler = _Handler()
    engine._plugin_switch = None
    engine._held_requests = []
    engine._ingested_seqs = 0
    engine._plugin_switch_unavailable = unavailable
    engine.has_pending_kv_work = lambda: False
    return engine


def test_the_engine_holds_new_requests_until_drained_then_switches():
    engine = _engine()
    args = {"documents": ["{}"], "plan_id": "p2", "request_id": "r1"}
    assert engine.begin_plugin_switch(args) is None
    assert "pending" in engine.begin_plugin_switch(args)
    engine._held_requests.append("seq-new")  # what pull_and_process_input_queue does
    engine.run_pending_plugin_switch()
    assert engine.utility_handler.ran == []  # still running: nothing switched
    engine.scheduler.running = 0
    engine.run_pending_plugin_switch()
    assert engine.utility_handler.ran == ["p2"] and engine._plugin_switch is None
    assert engine.scheduler.got == ["seq-new"] and engine._ingested_seqs == 1
    (rid, reply), = engine.utility_handler.answers
    assert rid == "r1" and reply["held_requests"] == 1


def test_a_drain_past_its_deadline_is_refused_and_lets_requests_in():
    engine = _engine()
    args = {"documents": ["{}"], "plan_id": "p2", "request_id": "r1", "drain_timeout_s": 0}
    engine.begin_plugin_switch(args)
    engine._held_requests.append("seq-new")
    time.sleep(0.01)
    engine.run_pending_plugin_switch()
    (rid, reply), = engine.utility_handler.answers
    assert "drain-timeout" in reply["error"] and engine.scheduler.got == ["seq-new"]
    assert engine.utility_handler.ran == []


def test_a_deployment_that_cannot_switch_refuses_before_holding_anything():
    engine = _engine(unavailable="the server was not started with --plugin-switchable")
    assert "--plugin-switchable" in engine.begin_plugin_switch(
        {"documents": ["{}"], "plan_id": "p"}
    )
    assert engine._plugin_switch is None
    dp = types.SimpleNamespace(
        plugin_switchable=True,
        parallel_config=types.SimpleNamespace(data_parallel_size=2),
        pipeline_parallel_size=1,
    )
    assert "DP or PP" in engine_core.plugin_switch_unavailable(dp)


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
