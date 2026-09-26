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
    from halucinator.backends.ghidra_backend import GhidraBackend
    from halucinator.backends.hal_backend import MemoryRegion

    # Distinguish "not installed" from "installed and broken". Skipping the
    # second is how a half-installed module reports fifteen passes' worth of
    # green while executing nothing: removing the extension's jar made every
    # test skip, because the language is present but its declared instruction
    # state modifier cannot be constructed. That is a misconfiguration to
    # report, not an absence to tolerate.
    try:
        import pyghidra  # noqa: F401
        pyghidra.start(verbose=False)
        from ghidra.program.util import DefaultLanguageService  # type: ignore
        from ghidra.program.model.lang import LanguageID  # type: ignore
        DefaultLanguageService.getLanguageService().getLanguage(
            LanguageID("Falcon:LE:32:fuc5"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Falcon processor module not installed: {exc}")

    # The language exists. Anything that fails from here is a real problem.
    probe = GhidraBackend(arch="falcon", cpu_model="fuc5")
    probe.add_memory_region(MemoryRegion("imem", 0x0, 0x100,
                                         permissions="rwx"))
    probe.init()


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
    assert be._step_fault_pc is None


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
    assert be._step_fault_pc is None


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
    # Assert the handler exists before asserting it saw nothing: `getattr(...,
    # set())` passes just as happily when crypt handling was never installed,
    # which is a pass that means the opposite of what it looks like.
    assert hasattr(be, "falcon_crypt_ops_seen"), "crypt handler not installed"
    assert not be.falcon_crypt_ops_seen


def test_no_transfer_or_tlb_op_failed(run):
    """A DMA/TLB op that raises is logged and then hands the guest a zero.

    The log is not enough on its own -- the firmware carries on with a wrong
    value and the run still looks finished. The errors are recorded so a run
    can be rejected outright.
    """
    be, _, _ = run
    assert hasattr(be, "falcon_xfer_errors"), "xfer engine not installed"
    assert be.falcon_xfer_errors == []


def test_the_carry_chained_instructions(run):
    """adc, sbb, shlc, div and mod -- none of which anything had executed.

    adc and sbb consume a carry produced two instructions earlier, so this also
    pins that `mov` leaves the flags alone (arith.rst: mov is the one unary
    that sets none) and that the sized `add b8` sets carry from bit 7.
    """
    _, _, got = run
    assert got["carry"] == oracle._carry()


def test_rotate_through_carry_and_setp(run):
    """shrc shifts the carry into the top bit; setp writes a flag bit directly.

    setp's 0xf2 form takes its flag selector from a third byte the constructor
    used not to consume, so this is also the regression test for that.
    """
    _, _, got = run
    assert got["rotc"] == oracle._rotc()


def test_mpop_and_mpopadd(run):
    """The other half of the derived multi-register family.

    mpopadd also moves $sp by its immediate, so the value only comes back if
    the pop count and the adjustment are both right.
    """
    _, _, got = run
    assert got["multi"] == oracle._multi()


def test_a_software_trap_is_delivered_and_returns(run):
    """trap pushes $pc, sets $tstatus to `pc | reason << 20`, jumps to $tv.

    All four trap constructors were a FalconTrap() pcodeop -- a black box that
    faults an emulator -- until this session. Nothing had executed one.
    """
    _, _, got = run
    assert got["trapw"] == oracle._trapw()


def test_the_remaining_arithmetic(run):
    """sub, cmp's zero flag, muls' signed multiply, and extr's bitfield."""
    _, _, got = run
    assert got["misc"] == oracle._misc()


def test_iord_extrs_and_a_far_call(run):
    """iord reads back what iowr wrote; extrs sign-extends where extr does
    not; and a far callee returning through mpopaddret leaves r0 intact."""
    _, _, got = run
    assert got["misc2"] == oracle._misc2()


def test_signed_and_unsigned_compare_differ(run):
    """cmps and cmpu on the same operands must disagree.

    If cmps were implemented as an unsigned compare -- which it was, before the
    flag sets were brought in line with arith.rst -- both would answer the same
    and this word would change.
    """
    _, _, got = run
    assert got["signed"] == oracle._signed()


# -- coverage ratchet -------------------------------------------------------

# Instructions the module implements that nothing executes. Every entry here
# is semantics no test can catch being wrong -- which is exactly how six
# constructors briefly negated the destination instead of the source, and how
# `setp` decoded two bytes where it needed three.
#
# crypt is excluded: it is deliberately unmodelled (crypt.rst is "todo: write
# me" throughout), so executing it would prove nothing.
KNOWN_UNEXECUTED: set = set()


def test_no_new_instruction_goes_unexecuted():
    """Ratchet: the set of never-executed instructions must not grow.

    Adding a constructor without anything that runs it is how a semantic
    defect survives a green suite.
    """
    import re
    import shutil
    import subprocess

    envydis = None
    for cand in (pathlib.Path.home() / "Development/envytools/build/envydis/envydis",
                 pathlib.Path.home() / ".envytools-cache/envytools/build/envydis/envydis"):
        if cand.is_file():
            envydis = str(cand)
            break
    envydis = envydis or shutil.which("envydis")
    if not envydis:
        pytest.skip("envydis not available")
    langdir = pathlib.Path(os.environ.get("GHIDRA_INSTALL_DIR", "")) /         "Ghidra/Processors/Falcon/data/languages"
    if not langdir.is_dir():
        pytest.skip("Falcon processor module not installed")

    def mnemonics(binary):
        out = re.sub(r"\x1b\[[0-9;]*m", "", subprocess.run(
            [envydis, "-m", "falcon", "-V", "fuc5", "-i", str(binary)],
            capture_output=True, text=True).stdout)
        return {m.group(1) for m in (re.match(
            r"^[0-9a-f]{8}:\s+(?:[0-9a-f]{2} )+\s*(?:[A-Z]{1,2}\s+)?([a-z][a-z0-9]*)\b",
            l.strip()) for l in out.splitlines()) if m}

    executed = mnemonics(FW) | mnemonics(FW4)
    sinc = " ".join(f.read_text() for f in langdir.glob("*.sinc"))
    implemented = set(re.findall(r"^:([a-z][a-z0-9]*)", sinc, re.M))
    crypt = {m for m in implemented if m.startswith("ci")}
    unexecuted = implemented - executed - crypt - KNOWN_UNEXECUTED
    assert not unexecuted, (
        f"{len(unexecuted)} instruction(s) implemented but executed by no "
        f"test firmware: {sorted(unexecuted)}. Add them to the conformance "
        f"firmware, or to KNOWN_UNEXECUTED with a reason.")


def test_ins_and_the_synchronous_io_pair(run):
    """ins replaces a bitfield in place; iowrs/iords are the synchronous forms.

    These were the last three instructions with nothing executing them. `ins`
    needed envydis's own bitfield notation -- `0x4:0x7`, a position and a top
    bit, not a position and a width.
    """
    _, _, got = run
    assert got["lastthree"] == oracle._lastthree()


def test_a_line_routed_to_vector_one_enters_iv1(run):
    """Routing decides which vector a line enters, and it was ignored.

    Verified by control: with the routing write replaced by zero the line goes
    to vector 0, whose enable is clear by this point, so it is dropped and the
    marker stays 0. irq_count is *not* the discriminator -- a misrouted line is
    refused rather than counted -- so this asserts the marker.
    """
    _, _, got = run
    assert got["vec1"] == oracle._vec1()
