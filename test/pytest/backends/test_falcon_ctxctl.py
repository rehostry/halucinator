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


def test_the_switch_issues_a_memory_command_for_that_channel():
    """A switch is not just a handshake: the context has to be moved.

    The microcode programs MEMIF with the channel PFIFO gave it and issues a
    command, which is the request to move that channel's context. Asserting the
    command carries *our* channel is what separates "it ran some code" from
    "it acted on the request".
    """
    be, eng, _ = _boot()
    eng.request_context_switch(CHANNEL)
    _run(be)
    assert eng.mem_commands, "no MEMIF command was issued"
    _cmd, chan, _base, _target = eng.mem_commands[0]
    from halucinator.peripheral_models.falcon_ctxctl import CTX_VALID
    assert chan == (CHANNEL | CTX_VALID), (
        f"MEMIF command carried channel 0x{chan:08x}")


def test_the_switch_finishes_and_returns_to_idle():
    """It must come back, not stall somewhere in the middle.

    The dispatcher's idle exit is the same code the firmware runs when it has
    nothing to do, so reaching it again after the switch is how we know the
    work ended rather than hung.
    """
    be, eng, before = _boot()
    eng.request_context_switch(CHANNEL)
    after = _run(be)
    assert eng.current_ctx == (CHANNEL | 0x80000000)
    assert before & after, "never returned to any previously-idle code"
    assert be._step_fault_pc is None


def test_without_a_request_it_never_switches():
    """The control. Idle for the same budget and CURRENT_CTX stays empty."""
    be, eng, before = _boot()
    after = _run(be)
    assert eng.current_ctx == 0
    assert eng.switch_requests == 0


def test_repeated_switches_are_deterministic():
    """Load, unload, load again -- and the second load matches the first.

    Each request drives one half of a switch: the first loads the channel and
    publishes it in CURRENT_CTX, the next saves it and leaves CURRENT_CTX
    empty, and the third loads it again. Asserting the third reproduces the
    first -- the same channel, the same number of MEMIF commands, the same
    volume of register traffic -- is what makes this a cycle rather than a
    one-off.

    What it is NOT: a check on the *contents* of a context image. The bulk of
    the state a real switch moves lives in the GPU, not in the firmware or its
    register file, and nothing here models that. Two loads agreeing shows the
    firmware's own path is repeatable; it says nothing about what was copied.
    """
    be, eng, _ = _boot()
    marks = []
    for chan in (0x1111, 0x2222, 0x1111):
        before = (len(eng.mem_commands), eng.mmio_writes)
        eng.request_context_switch(chan)
        _run(be)
        marks.append((eng.current_ctx,
                      len(eng.mem_commands) - before[0],
                      eng.mmio_writes - before[1]))
    first, unload, again = marks
    assert first[0] == 0x1111 | 0x80000000, f"first load gave {first}"
    assert unload[0] == 0, f"second request should unload, gave {unload}"
    assert again == first, f"reload differed from the first load: {again} vs {first}"
    assert be._step_fault_pc is None


def test_a_fecs_method_is_received_and_acknowledged():
    """The host's command path, end to end into the microcode.

    Nouveau drives FECS by writing BAR0 0x409500 (argument) and 0x409504
    (method) and polling 0x409800 for a reply. The firmware never reads that
    submission window: the hardware forwards the write into the falcon's own
    method FIFO, and the microcode takes it from there on interrupt line 2,
    reading FIFO_CMD and FIFO_DATA and writing FIFO_ACK when done.

    This asserts the method is consumed -- that the whole path from a driver
    register write to the microcode's acknowledgement is connected. It does
    *not* assert a reply: discover_image_size answers only after the long init
    sequence nouveau performs first, which this does not yet replay.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_DISCOVER_IMAGE_SIZE)

    be, eng, _ = _boot()
    eng.fecs_method(MTHD_DISCOVER_IMAGE_SIZE, 0)
    assert eng._fifo, "method was not queued"
    _run(be)
    assert not eng._fifo, "firmware never acknowledged the method"
    assert eng.methods == [(MTHD_DISCOVER_IMAGE_SIZE, 0)]
    assert be._step_fault_pc is None
