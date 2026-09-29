"""The CTXCTL model, and the context switch it exists to perform.

FECS is PGRAPH's context-switch controller. Booting it and servicing its
interrupts is not the same as it doing its job; this exercises the job.

Skipped unless pyghidra, the Falcon processor module and a linux-firmware
checkout are all present -- the microcode is not vendored.
"""
from __future__ import annotations

import os
import pathlib

import pytest

pytest.importorskip("pyghidra")

FW_ENV = "FALCON_FIRMWARE_DIR"
CHANNEL = 0x1234


def _fecs():
    root = os.environ.get(FW_ENV)
    if not root:
        pytest.skip(f"set {FW_ENV} to a linux-firmware nvidia/ tree")
    for chip in ("gp102", "gp104", "gp100", "tu104", "tu102"):
        p = pathlib.Path(root) / chip / "gr" / "fecs_inst.bin"
        if p.is_file():
            return p
    pytest.skip("no FECS microcode under FALCON_FIRMWARE_DIR")


def _boot(steps=150_000):
    from halucinator.backends.ghidra_backend import GhidraBackend
    from halucinator.backends.hal_backend import MemoryRegion
    from halucinator.backends.irq.delivery import DeliveryPlan
    from halucinator.peripheral_models.falcon_ctxctl import FalconCtxctl

    fw = _fecs()
    if not os.environ.get("GHIDRA_INSTALL_DIR"):
        pytest.skip("GHIDRA_INSTALL_DIR not set")
    eng = FalconCtxctl("ctxctl", 0x0, 0x40000)
    be = GhidraBackend(arch="falcon", cpu_model="fuc5")
    be.add_memory_region(MemoryRegion("imem", 0x0, 0x8000,
                                      permissions="rwx", file=str(fw)))
    be.add_memory_region(MemoryRegion("dmem", 0x0, 0x4000,
                                      permissions="rw", space="dmem"))
    be.add_memory_region(MemoryRegion("io", 0x0, 0x40000, permissions="rw",
                                      space="io", emulate=eng))
    be.init()
    be.write_register("sp", 0x4000)
    be.write_register("pc", 0x0)
    be.auto_deliver_peripheral_irqs = True
    be.peripheral_irq_plan = DeliveryPlan(falcon_vector=0, stack_space="dmem")
    seen = set()
    for _ in range(steps):
        seen.add(be.read_register("pc"))
        be.step()
        if be._halted or be._step_fault_pc is not None:
            break
    return be, eng, seen


def _run(be, steps=150_000):
    seen = set()
    for _ in range(steps):
        seen.add(be.read_register("pc"))
        be.step()
        if be._halted or be._step_fault_pc is not None:
            break
    return seen


@pytest.fixture(scope="module")
def booted():
    return _boot()


def test_it_boots_without_wedging(booted):
    be, _, _ = booted
    assert be._step_fault_pc is None


def test_it_reaches_its_idle_loop(booted):
    """Initialisation completes and the firmware settles into `sleep`.

    Before the MMIO bus was modelled it spent its whole run polling one BAR0
    register and reached 247 distinct instructions; it now reaches several
    hundred and stops at the dispatcher.
    """
    _, _, seen = booted
    assert len(seen) > 400, f"only {len(seen)} distinct PCs -- still stuck early"


def test_it_used_the_mmio_bus(booted):
    _, eng, _ = booted
    assert eng.mmio_reads > 0 and eng.mmio_writes > 0


def test_it_performs_a_context_switch():
    """The thing FECS exists to do.

    PFIFO publishes a channel in NEW_CTX and raises the switch interrupt; the
    microcode does its work and publishes the channel it has loaded in
    CURRENT_CTX. Asserting on that value -- not merely that code ran -- is what
    makes this a result rather than an observation.
    """
    be, eng, before = _boot()
    assert eng.current_ctx == 0, "CURRENT_CTX set before any request"
    eng.request_context_switch(CHANNEL)
    after = _run(be)
    from halucinator.peripheral_models.falcon_ctxctl import CTX_VALID
    assert eng.current_ctx == (CHANNEL | CTX_VALID), (
        f"firmware published CURRENT_CTX=0x{eng.current_ctx:08x}, "
        f"expected 0x{CHANNEL | CTX_VALID:08x}")
    assert len(after - before) > 100, "no new code path was taken"


def test_without_a_request_it_never_switches():
    """The control. Idle for the same budget and CURRENT_CTX stays empty."""
    be, eng, before = _boot()
    after = _run(be)
    assert eng.current_ctx == 0
    assert eng.switch_requests == 0
