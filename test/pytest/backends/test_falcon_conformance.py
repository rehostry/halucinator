"""End-to-end Falcon check: exact arithmetic, driven by real interrupts.

Unlike the other Falcon suites, this one runs an actual emulator over an
actual firmware image. The image is ours (test/falcon_conformance/), assembled
from irq_compute.fuc with envytools' envyas, so nothing vendor-owned is
redistributed and the expected values can be derived from the source rather
than recorded from a previous run.

The four words it checks are chosen so that a defect changes a number instead
of merely looking odd:

  sum_sq     64 iterations of mulu + add, each term computed through a
             call/ret pair, so the return address has to survive on the DMEM
             stack alongside push/pop.
  irq_count  the firmware waits for exactly 8 interrupts before it proceeds,
             so this is 8 by construction however fast the model runs -- and
             if interrupts stop arriving, the firmware hangs rather than
             producing a plausible smaller number.
  mix        folded by the handler on every entry, in order, through a chain
             built to overflow 16 bits, which pins mulu's documented
             16x16->32 truncation.
  done       mix ^ sum_sq, written last, so it also serves as the "finished"
             marker.

Skipped unless pyghidra and the Falcon processor module are both installed.
"""
from __future__ import annotations

import os
import pathlib

import pytest

pytest.importorskip("pyghidra")

FW = (pathlib.Path(__file__).resolve().parents[2]
      / "falcon_conformance" / "irq_compute.bin")

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "falcon_oracle",
    pathlib.Path(__file__).resolve().parents[2] / "falcon_conformance" / "oracle.py")
oracle = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(oracle)

N_IRQ, TERMS = oracle.N_IRQ, oracle.TERMS
DONE = oracle.DONE


def _run(no_timer=False, cap=400_000, fw=None, cpu_model="fuc5",
         done_offset=None):
    from halucinator.backends.ghidra_backend import GhidraBackend
    from halucinator.backends.hal_backend import MemoryRegion
    from halucinator.backends.irq.delivery import DeliveryPlan
    from halucinator.peripheral_models.falcon_engine import FalconEngine

    be = GhidraBackend(arch="falcon", cpu_model=cpu_model)
    engine = FalconEngine("engine", 0x0, 0x40000)
    if no_timer:
        engine.tick = lambda steps=1: None
    be.add_memory_region(MemoryRegion("imem", 0x0, 0x8000,
                                      permissions="rwx",
                                      file=str(fw or FW)))
    be.add_memory_region(MemoryRegion("dmem", 0x0, 0x4000,
                                      permissions="rw", space="dmem"))
    be.add_memory_region(MemoryRegion("io", 0x0, 0x40000, permissions="rw",
                                      space="io", emulate=engine))
    be.init()
    be.write_register("sp", 0x1000)
    be.write_register("pc", 0x0)
    be.auto_deliver_peripheral_irqs = True
    be.peripheral_irq_plan = DeliveryPlan(falcon_vector=0, stack_space="dmem")
    # The firmware code-loads a page from external port 0; put one there.
    be.falcon_xfer.load_port(0, bytes(range(256)), oracle.CODE_PAGE_EXT_OFF)
    for _ in range(cap):
        be.step()
        if be.read_memory(DONE, 4, space="dmem"):
            break
    got = {k: be.read_memory(off, 4, space="dmem")
           for k, off in oracle.OFFSETS.items()}
    return be, engine, got


def _require_falcon(fw):
    """Skip only when the Falcon language is genuinely unavailable.

    An earlier version wrapped the whole run in `except Exception: skip`, so a
    backend that raised for any reason reported 15 skips instead of 15
    failures -- the tests looked fine while measuring nothing. Availability is
    now probed separately, and the run itself is allowed to fail.
    """
    if not os.environ.get("GHIDRA_INSTALL_DIR"):
        pytest.skip("GHIDRA_INSTALL_DIR not set")
    if not fw.exists():
        pytest.skip(f"{fw} missing")
    try:
        from halucinator.backends.ghidra_backend import GhidraBackend
        from halucinator.backends.hal_backend import MemoryRegion
        probe = GhidraBackend(arch="falcon", cpu_model="fuc5")
        probe.add_memory_region(MemoryRegion("imem", 0x0, 0x100,
                                             permissions="rwx"))
        probe.init()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Falcon language unavailable: {exc}")


@pytest.fixture(scope="module")
def run():
    _require_falcon(FW)
    return _run()


def test_the_firmware_finishes(run):
    _, _, got = run
    assert got["done"], "firmware never wrote its done marker"


def test_every_word_is_exact(run):
    _, _, got = run
    assert got == oracle.expected()


def test_the_arithmetic_is_right(run):
    """64 mulu+add terms, each through a call/ret over the DMEM stack."""
    _, _, got = run
    assert got["sum_sq"] == sum(i * i for i in range(1, TERMS + 1)) == 89440


def test_the_alu_chain_is_right(run):
    """and/or/xor/shl/sar/hswap/xbit/sext/not/neg in one dependent chain.

    Every one of these was either a `define pcodeop` -- opaque to an emulator,
    and in mulu's case not even stubbed, so it faulted -- or missing entirely.
    """
    _, _, got = run
    assert got["alu"] == oracle._alu()


def test_mpush_preserves_the_registers_below_it(run):
    """`mpush $r4` must save r0-r3 across a callee that destroys all four.

    This pins a semantic that is derived rather than documented: the operand
    is a count, not a single register. Under the single-register reading the
    callee's writes reach the caller and this word is 0xDEADBEEF-ish garbage
    instead of 0x11223344.
    """
    _, _, got = run
    assert got["regs"] == 0x11223344


def test_exactly_one_handler_entry_per_expiry(run):
    _, engine, got = run
    assert got["irq_count"] == N_IRQ
    assert engine.timer_ticks == N_IRQ


def test_no_instruction_wedged_the_emulator(run):
    be, _, _ = run
    assert getattr(be, "_step_fault_pc", None) is None


def test_without_a_timer_nothing_is_produced():
    """The control. A harness that passes here is measuring itself."""
    _require_falcon(FW)
    _, engine, got = _run(no_timer=True, cap=20_000)
    assert engine.timer_ticks == 0
    assert all(v == 0 for v in got.values())


# -- the fuc4 variant -------------------------------------------------------
#
# fuc4 and fuc5 include the same .sinc files, so a change made for one reaches
# the other. Before this, fuc4 was only ever compiled, never executed.

FW4 = (pathlib.Path(__file__).resolve().parents[2]
       / "falcon_conformance" / "irq_compute_fuc4.bin")


@pytest.fixture(scope="module")
def run4():
    _require_falcon(FW4)
    return _run(fw=FW4, cpu_model="fuc4", done_offset=oracle.DONE)


def test_fuc4_runs_and_is_exact(run4):
    be, engine, got = run4
    got4 = {k: got[k] for k in oracle.FUC4_OFFSETS}
    assert got4 == oracle.expected_fuc4()


def test_fuc4_took_its_interrupts(run4):
    _, engine, got = run4
    assert got["irq_count"] == oracle.FUC4_N_IRQ
    assert engine.timer_ticks == oracle.FUC4_N_IRQ


def test_fuc4_did_not_wedge(run4):
    be, _, _ = run4
    assert getattr(be, "_step_fault_pc", None) is None


def test_the_dma_round_trip(run):
    """DMEM -> external memory -> DMEM, through xdst and xdld.

    External memory is not in the emulator at all -- Falcon's SLEIGH declares
    code, data and I/O and no fourth space -- so it is held by the transfer
    engine and moved across by the backend. A broken path returns zero.
    """
    _, _, got = run
    assert got["dma"] == oracle.DMA_PATTERN


def test_the_tlb_operations(run):
    """ptlb, vtlb and itlb after a code load, folded so any one matters."""
    _, _, got = run
    assert got["tlb"] == oracle._tlb()


def test_the_code_load_really_moved_the_page(run):
    """xcld copies 0x100 bytes into IMEM and maps the page."""
    be, _, _ = run
    page = bytes(be.read_memory(oracle.CODE_PAGE_PHYS, 1, 8, raw=True))
    assert page == bytes(range(8))
    assert be.falcon_xfer.code_loads == 1


def test_no_crypt_was_executed(run):
    """The crypt coprocessor is not modelled, so reaching it would invalidate
    the run. Nothing in this firmware -- or in either shipped GP102 image --
    touches it."""
    be, _, _ = run
    assert not getattr(be, "falcon_crypt_ops_seen", set())
