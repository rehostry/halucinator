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
    """The FECS code and data images, as a pair.

    Nouveau loads both -- `nvkm_falcon_load_imem` for fecs_inst.bin and
    `nvkm_falcon_load_dmem` for fecs_data.bin -- and the data image is not
    optional: it holds the tables the microcode's own init reads.
    """
    root = os.environ.get(FW_ENV)
    if not root:
        pytest.skip(f"set {FW_ENV} to a linux-firmware nvidia/ tree")
    for chip in ("gp102", "gp104", "gp100", "tu104", "tu102"):
        inst = pathlib.Path(root) / chip / "gr" / "fecs_inst.bin"
        data = inst.with_name("fecs_data.bin")
        if inst.is_file() and data.is_file():
            return inst, data
    pytest.skip("no FECS microcode under FALCON_FIRMWARE_DIR")


def _gpccs():
    """The GPCCS code and data images -- the per-GPC context controller."""
    root = os.environ.get(FW_ENV)
    if not root:
        pytest.skip(f"set {FW_ENV} to a linux-firmware nvidia/ tree")
    for chip in ("gp102", "gp104", "gp100", "tu104", "tu102"):
        inst = pathlib.Path(root) / chip / "gr" / "gpccs_inst.bin"
        data = inst.with_name("gpccs_data.bin")
        if inst.is_file() and data.is_file():
            return inst, data
    pytest.skip("no GPCCS microcode under FALCON_FIRMWARE_DIR")


def _boot(steps=150_000, images=None, **engine_kwargs):
    from halucinator.backends.ghidra_backend import GhidraBackend
    from halucinator.backends.hal_backend import MemoryRegion
    from halucinator.backends.irq.delivery import DeliveryPlan
    from halucinator.peripheral_models.falcon_ctxctl import FalconCtxctl

    inst, data = images if images is not None else _fecs()
    if not os.environ.get("GHIDRA_INSTALL_DIR"):
        pytest.skip("GHIDRA_INSTALL_DIR not set")
    eng = FalconCtxctl("ctxctl", 0x0, 0x40000, **engine_kwargs)
    be = GhidraBackend(arch="falcon", cpu_model="fuc5")
    be.add_memory_region(MemoryRegion("imem", 0x0, 0x8000,
                                      permissions="rwx", file=str(inst)))
    be.add_memory_region(MemoryRegion("dmem", 0x0, 0x4000,
                                      permissions="rw", space="dmem",
                                      file=str(data)))
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


def test_switching_between_two_channels_moves_each_channel_s_image():
    """Load, switch away, switch back -- and each step touches its own image.

    A switch request names a channel, and the work is not done until the
    context belonging to *that* channel has moved. Three things have to agree
    for that to be true, and this asserts all three: CURRENT_CTX names the
    channel that was asked for, the MEMIF command carries it, and the DMA the
    firmware issues reads from an address that is different for a different
    channel and the same for the same one.

    That last one is why the transfer log exists. Without it "a switch
    happened" and "the right context moved" are indistinguishable -- a firmware
    that ignored the channel and always read the same image would satisfy every
    other assertion here.

    What it is NOT: a check on the *contents* of a context image. The bulk of
    the state a real switch moves lives in the GPU, not in the firmware or its
    register file, and nothing here models that. The addresses are evidence
    about which image was addressed, not about what was in it.
    """
    be, eng, _ = _boot()
    xfer = be.falcon_xfer
    seen = {}
    for chan in (0x1111, 0x2222, 0x1111):
        cmds_before = len(eng.mem_commands)
        moves_before = len(xfer.transfers)
        eng.request_context_switch(chan)
        _run(be)
        assert eng.current_ctx == chan | 0x80000000, (
            f"asked for 0x{chan:x}, CURRENT_CTX is 0x{eng.current_ctx:08x}")
        new_cmds = eng.mem_commands[cmds_before:]
        assert len(new_cmds) == 1, f"{len(new_cmds)} MEMIF commands for one switch"
        assert new_cmds[0][1] == chan | 0x80000000, (
            f"MEMIF command carried 0x{new_cmds[0][1]:08x}")
        addrs = tuple(t[2] for t in xfer.transfers[moves_before:])
        assert addrs, f"no context data moved for channel 0x{chan:x}"
        # The first transfer of each switch is the one addressed from the
        # channel; the very first switch also pulls two fixed pages that the
        # later ones do not, so the sequences are not comparable as a whole.
        seen.setdefault(chan, []).append(addrs[0])
    assert seen[0x1111][0] != seen[0x2222][0], (
        "the two channels' contexts came from the same address -- the channel "
        f"was ignored: 0x{seen[0x1111][0]:x} vs 0x{seen[0x2222][0]:x}")
    assert seen[0x1111][0] == seen[0x1111][1], (
        f"switching back read a different image: "
        f"0x{seen[0x1111][0]:x} then 0x{seen[0x1111][1]:x}")
    assert be._step_fault_pc is None


def test_a_request_the_firmware_declines_moves_nothing():
    """The control for the switch tests.

    PFIFO marks a channel valid in the engine status; without that bit the
    microcode has nothing to switch to and does not act. Asserting that no
    MEMIF command is issued and CURRENT_CTX is untouched is what shows the
    switches above are driven by the request rather than by the interrupt
    alone.

    This is not a model of the unload path. Hardware unloads a context when
    PFIFO publishes an invalid next channel, and nothing here exercises that.
    """
    be, eng, _ = _boot()
    eng.request_context_switch(0x1111)
    _run(be)
    loaded = eng.current_ctx
    cmds = len(eng.mem_commands)
    moves = len(be.falcon_xfer.transfers)
    eng.request_context_switch(0x3333, valid=False)
    _run(be)
    assert eng.current_ctx == loaded, "an invalid request changed CURRENT_CTX"
    assert len(eng.mem_commands) == cmds, "an invalid request moved a context"
    assert len(be.falcon_xfer.transfers) == moves
    assert be._step_fault_pc is None


def test_a_fecs_method_is_received_and_acknowledged():
    """The host's command path, end to end into the microcode.

    Nouveau drives FECS by writing BAR0 0x409500 (argument) and 0x409504
    (method) and polling 0x409800 for a reply. The firmware never reads that
    submission window: the hardware forwards the write into the falcon's own
    method FIFO, and the microcode takes it from there on interrupt line 2,
    reading FIFO_CMD and FIFO_DATA and writing FIFO_ACK when done.

    This asserts the method is consumed -- that the whole path from a driver
    register write to the microcode's acknowledgement is connected. The reply
    itself is asserted separately, by test_discover_image_size_answers.
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


# ---------------------------------------------------------------------------
# The host control-method path: FECS answering a question the driver asks.
# ---------------------------------------------------------------------------

def _expected_image_size(strand_words, mmctx_bytes=1024):
    """The context-image size, computed from the firmware's own arithmetic.

    Written from the GP102 FECS disassembly rather than from the model, so
    agreement means two readings match rather than that the model agrees with
    itself. Three pieces:

    * The strand walk at 0xa4a runs STRAND_WORDS[0..8] and, for each strand
      that reports a non-zero count, adds ``1 + (words >> 6)`` to a running
      total which it also writes back as that strand's save and load base.
    * Those bases are stored shifted right by 8 -- rnndb marks
      STRAND_SAVE_SWBASE ``shr="8"`` -- so a unit is 256 bytes.
    * The reply at 0x2b4b rounds the MMIO-context part up to the next 4 KiB
      page and adds the strand part on top, un-rounded::

          size = ((mmctx >> 12) + 1) << 12  +  strand_bytes

    `mmctx_bytes` is the model's share, not a hardware fact: with no register
    lists behind the MMIO bus the firmware finds 1024 bytes of context, and
    the control test pins that number rather than assuming it.
    """
    units = sum(1 + (w >> 6) for w in strand_words if w > 0)
    strand_bytes = units * 256
    return (((strand_bytes + mmctx_bytes) >> 12) + 1) * 4096 + strand_bytes


def _round_up_256(value):
    """The firmware's own rounding helper at 0x4a6: ((x >> 8) + 1) << 8.

    Note it always adds at least one unit -- `round_up_256(0)` is 0x100, not 0 --
    which is why a size query answers with a whole unit even when there is
    nothing to store.
    """
    return ((value >> 8) + 1) << 8


def _expected_zcull_size():
    """The ZCULL image size with no zcull state modelled.

    The routine at 0x255d accumulates into DMEM 0x714 and then, at 0x2660, puts
    a floor under it: if the total is still zero it writes 0x100 before replying
    at 0x267b. With nothing to accumulate the floor is the answer.
    """
    return 0x100


def _expected_pm_size():
    """The PerfMon image size, which is a sum of two rounded terms.

    The routine at 0x1e23 calls the unit-size getter at 0x43f twice and adds the
    results (0x1e54). Both go through that getter's round-to-256 tail:

      * index 0xa is derived from the TPC and GPC counts in DMEM 0x7bc/0x7c0/
        0x7c4. With no units modelled those are zero, so the term rounds to one
        unit.
      * index 2 is the firmware constant 0x6f0, which rounds to 0x700.
    """
    return _round_up_256(0) + _round_up_256(0x6F0)


def test_one_method_is_dispatched_once():
    """Interrupt line 2 is level-triggered, and that is load-bearing.

    The firmware's own init writes INTR_MODE = 0x4, making line 2 -- and only
    line 2 -- level-triggered, and its handler at 0x2bb acknowledges by writing
    FIFO_ACK, never INTR_CLEAR. So the line's state has to follow the FIFO:
    latch it instead and the handler re-enters on every return, the main loop
    never runs, and the method is queued into the firmware's work ring over and
    over until the ring fills.

    The ring's write pointer is the witness. It lives at DMEM 0x828 (the read
    pointer at 0x824, the eight-entry ring at 0x7e4) and counts submissions, so
    "1" says the method was handed over exactly once. A latched line put 8
    there -- the ring's full depth -- and left the work undone.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_DISCOVER_IMAGE_SIZE, METHOD_IRQ_LINE)

    be, eng, _ = _boot()
    assert be.read_memory(0x828, 4, 1, space="dmem") == 0, "ring used at boot"
    eng.fecs_method(MTHD_DISCOVER_IMAGE_SIZE, 0)
    _run(be)
    wr = be.read_memory(0x828, 4, 1, space="dmem")
    rd = be.read_memory(0x824, 4, 1, space="dmem")
    assert wr == 1, f"method was enqueued {wr} times, not once"
    assert rd == wr, f"work ring not drained: read {rd}, write {wr}"
    assert be.read_memory(0x7E4, 4, 1, space="dmem") == MTHD_DISCOVER_IMAGE_SIZE
    assert not eng.intr_status() & (1 << METHOD_IRQ_LINE), (
        "the line is still asserted with an empty FIFO")


def test_discover_image_size_answers():
    """What FECS exists to be asked.

    Nouveau's `gf100_gr_fecs_discover_image_size` clears BAR0 0x409800, writes
    the argument and then method 0x10, and polls 0x409800 until it reads
    non-zero -- that value is the size of a graphics context image, and the
    driver cannot allocate a context without it.

    Getting an answer at all means the whole chain works: the submission
    reaches the method FIFO, the interrupt handler dispatches it into the work
    ring, the main loop drains the ring, the handler at 0x4d89 runs the sizing
    routine at 0x2876, the strand unit answers its walk, and the reply is
    published in the scratch register the driver is watching.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_DISCOVER_IMAGE_SIZE)

    be, eng, _ = _boot()
    eng.fecs_method(MTHD_DISCOVER_IMAGE_SIZE, 0)
    _run(be)
    assert eng.scratch0 == _expected_image_size([0] * 9), (
        f"FECS reported 0x{eng.scratch0:x}")
    assert be._step_fault_pc is None


def test_the_reported_size_is_computed_from_the_strand_state():
    """The falsification knob: change the hardware, the answer must move.

    A size that came out right once could be a constant. This gives the strand
    unit a different amount of state to report and requires the number to move
    by exactly what the firmware's arithmetic says -- so the reply is being
    computed from the modelled hardware, not recited.

    It also pins the strand walk itself: each non-empty strand is handed a base
    in the image, and those bases must be the running total at the point the
    walk reached it.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_DISCOVER_IMAGE_SIZE)

    words = [4096, 2048, 0, 0, 0, 0, 0, 0, 0]
    be, eng, _ = _boot(strand_words=words)
    eng.fecs_method(MTHD_DISCOVER_IMAGE_SIZE, 0)
    _run(be)
    assert eng.scratch0 == _expected_image_size(words), (
        f"FECS reported 0x{eng.scratch0:x}, expected "
        f"0x{_expected_image_size(words):x}")
    assert eng.scratch0 != _expected_image_size([0] * 9), (
        "the size did not respond to the strand state at all")
    # strand 1 starts where strand 0 ended: base0 + 1 + (4096 >> 6)
    assert eng.strand_save_base[1] - eng.strand_save_base[0] == 1 + (4096 >> 6)
    assert eng.strand_save_base == eng.strand_load_base, (
        "save and load bases must be the same image offsets")


def test_the_strand_unit_ran_the_documented_protocol():
    """The walk is a command sequence, not a register read.

    rnndb names STRAND_CMD's values; the firmware brackets its walk with
    ENABLE and DISABLE and uses SEEK and GET_INFO in between. Asserting the
    sequence is what distinguishes "the firmware drove the strand unit" from
    "the firmware read nine registers that happened to be there".
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_DISCOVER_IMAGE_SIZE, STRAND_CMD_ENABLE, STRAND_CMD_DISABLE,
        STRAND_CMD_SEEK, STRAND_CMD_GET_INFO)

    be, eng, _ = _boot()
    eng.fecs_method(MTHD_DISCOVER_IMAGE_SIZE, 0)
    _run(be)
    cmds = eng.strand_cmds
    assert STRAND_CMD_ENABLE in cmds and STRAND_CMD_DISABLE in cmds
    assert cmds.index(STRAND_CMD_ENABLE) < cmds.index(STRAND_CMD_GET_INFO)
    assert cmds.index(STRAND_CMD_GET_INFO) < cmds.index(STRAND_CMD_DISABLE)
    assert STRAND_CMD_SEEK in cmds


def test_an_unrecognised_method_is_refused_and_reported():
    """The discrimination control for every method test above.

    If the dispatcher ran the same code whatever it was handed, a reply would
    say nothing about the method. Its default case at 0x4e6a writes 0x11 --
    "method not recognised" -- into SCRATCH[6] and raises the upstream
    interrupt, so a method that is genuinely not in the table leaves a
    different, identifiable trace and no reply.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        SCRATCH_ERROR, ERR_UNKNOWN_METHOD)

    be, eng, _ = _boot()
    assert eng.scratch[SCRATCH_ERROR] != ERR_UNKNOWN_METHOD
    eng.fecs_method(0x7E, 0)                  # not in the dispatch table
    _run(be)
    assert not eng._fifo, "the method was never acknowledged"
    assert eng.scratch[SCRATCH_ERROR] == ERR_UNKNOWN_METHOD, (
        f"error register holds 0x{eng.scratch[SCRATCH_ERROR]:x}, "
        f"expected 0x{ERR_UNKNOWN_METHOD:x}")
    assert eng.scratch0 == 0, "an unknown method should not reply"
    assert be._step_fault_pc is None


def test_nouveau_s_init_sequence_runs_to_completion():
    """The driver's whole context-controller bring-up, step for step.

    `gf100_gr_init_ctxctl_ext` in nvkm/engine/gr/gf100.c is what the kernel
    does after loading FECS: start it, wait for ready, set the watchdog, then
    ask for three sizes. Each step's success criterion is nouveau's own, so
    passing here means the sequence the driver actually performs completes --
    not that a sequence invented for this test does.

      * ready is BAR0 0x409800 bit 0
      * set_watchdog_timeout returns void and never polls, so the only thing
        to check is that the timeout reached the firmware; it stores it at
        DMEM 0x24
      * each discover_* clears 0x409800, writes its argument and method, and
        polls 0x409800 until non-zero -- that value is the size

    ELPG binding is in that function too but guarded by `if (0)`, so it is not
    part of the sequence and is not replayed.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_SET_WATCHDOG_TIMEOUT, MTHD_DISCOVER_IMAGE_SIZE,
        MTHD_DISCOVER_ZCULL_IMAGE_SIZE, MTHD_DISCOVER_PM_IMAGE_SIZE)

    be, eng, _ = _boot()
    assert eng.scratch0 & 0x00000001, (
        f"FECS never reported ready; 0x409800 = 0x{eng.scratch0:08x}")

    timeout = 0x7FFFFFFF
    eng.fecs_method(MTHD_SET_WATCHDOG_TIMEOUT, timeout)
    _run(be)
    assert be.read_memory(0x24, 4, 1, space="dmem") == timeout, (
        "the watchdog timeout never reached the firmware")

    # Each expected value is computed from the routine that produces it, so a
    # wrong one is a wrong number rather than a missing reply.
    want = {"size": _expected_image_size([0] * 9),
            "size_zcull": _expected_zcull_size(),
            "size_pm": _expected_pm_size()}
    sizes = {}
    for name, mthd in (("size", MTHD_DISCOVER_IMAGE_SIZE),
                       ("size_zcull", MTHD_DISCOVER_ZCULL_IMAGE_SIZE),
                       ("size_pm", MTHD_DISCOVER_PM_IMAGE_SIZE)):
        sizes[name] = eng.fecs_call(be, mthd, 0)
        assert sizes[name], f"{name}: no reply, which is nouveau's -ETIMEDOUT"
    assert sizes == want, f"got {sizes}, expected {want}"
    # The three are different questions and must not answer the same number.
    assert len(set(sizes.values())) == 3, f"sizes are not distinct: {sizes}"
    assert be._step_fault_pc is None


def test_the_reglist_size_method_reads_its_argument():
    """Method 0x30 is conditional on its argument, and that is checkable.

    nouveau's `discover_reglist_image_size` passes 1, and the handler at 0x47d5
    tests bit 0 of the argument: set, it computes a size; clear, it replies
    zero. Both branches are asserted, which shows the argument reaches the
    microcode rather than being dropped on the way.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_DISCOVER_REGLIST_IMAGE_SIZE as MTHD)

    be, eng, _ = _boot()
    assert eng.fecs_call(be, MTHD, 1), "arg=1 must produce a size"
    with_arg = eng.scratch0
    eng.fecs_method(MTHD, 0)
    _run(be)
    assert eng.scratch0 == 0, (
        f"arg=0 replied 0x{eng.scratch0:x}; the argument was ignored")
    assert with_arg != 0
    assert be._step_fault_pc is None


def test_the_reglist_bind_methods_take_an_instance_pointer():
    """Two methods whose parameter arrives in a different register.

    nouveau's `set_reglist_bind_instance` and `set_reglist_virtual_address`
    write the pointer to BAR0 0x409810 -- SCRATCH[4], not the FIFO argument
    word -- then submit with argument 1 and wait for SCRATCH[0] to read exactly
    1. Both halves matter: the parameter goes one way and the reply comes back
    another, and the success criterion is an exact value rather than "non-zero".
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_SET_REGLIST_BIND_INSTANCE, MTHD_SET_REGLIST_VIRTUAL_ADDRESS,
        SCRATCH_INSTANCE)

    be, eng, _ = _boot()
    eng.scratch[SCRATCH_INSTANCE] = 0x00ABCD
    assert eng.fecs_call(be, MTHD_SET_REGLIST_BIND_INSTANCE, 1) == 1, (
        f"set_reglist_bind_instance replied 0x{eng.scratch0:x}, not 1")
    eng.scratch[SCRATCH_INSTANCE] = 0x1000 >> 8
    assert eng.fecs_call(be, MTHD_SET_REGLIST_VIRTUAL_ADDRESS, 1) == 1, (
        f"set_reglist_virtual_address replied 0x{eng.scratch0:x}, not 1")
    assert be._step_fault_pc is None


def test_binding_an_instance_pointer_reports_done_not_error():
    """The step that gives FECS a context to work on.

    `gf100_gr_fecs_bind_pointer` masks 0x30 out of BAR0 0x409800, submits method
    0x03 with the instance pointer, and then distinguishes two outcomes: bit
    0x10 is done and bit 0x20 is -EIO. Asserting *which* bit appears is what
    makes this a success rather than merely a response -- the firmware has a
    live error path here and the test would pass on it if it only checked that
    something changed.

    The mask also matters. SCRATCH[0] still carries the ready bit at this point,
    so clearing the whole register would be a different operation from the one
    the driver performs, and polling for "non-zero" would report done before the
    method had run.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_BIND_POINTER, BIND_DONE, BIND_ERROR)

    be, eng, _ = _boot()
    reply = eng.fecs_call(be, MTHD_BIND_POINTER, 0x00ABCD, clear=0x30,
                          until=BIND_DONE | BIND_ERROR)
    assert not reply & BIND_ERROR, "the firmware reported -EIO"
    assert reply & BIND_DONE, (
        f"no outcome bit set; SCRATCH[0] = 0x{reply:08x}")
    assert be._step_fault_pc is None


def test_saving_a_golden_context_moves_real_register_state():
    """The context image gets built, and out of the GPU's own registers.

    `gf100_gr_fecs_wfi_golden_save` is how the driver captures a pristine
    graphics context: bind an instance pointer, then submit method 0x09 and wait
    for bit 0x1 in BAR0 0x409800 (0x2 would be -EIO).

    Getting that bit needs the MMCTX engine, which is the part that moves the
    MMIO half of a context. The firmware drives it as a queue -- START_TRIGGER,
    then a descriptor per register range for as long as QFREE says there is
    room, then STOP_TRIGGER -- and each descriptor names a run of consecutive
    BAR0 registers. With QFREE reading zero the firmware waited for room
    forever and this never finished.

    The assertions are about the image, not just the bit: the engine must have
    read registers, the number of words in the image must equal the number of
    registers the descriptors named, and the addresses must be the PGRAPH
    registers the firmware's own lists hold rather than an arbitrary block.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_BIND_POINTER, MTHD_WFI_GOLDEN_SAVE, BIND_DONE, BIND_ERROR,
        GOLDEN_DONE, GOLDEN_ERROR)

    # A sentinel in a register the firmware's own list covers. If the engine
    # wrote zeros instead of reading the register file, the image would not
    # contain it -- and the assertion below refuses to pass vacuously if the
    # firmware turns out not to save this address.
    sentinel_reg, sentinel = 0x404004, 0x5EED1234

    be, eng, _ = _boot(bar0={sentinel_reg: sentinel})
    assert eng.fecs_call(be, MTHD_BIND_POINTER, 0x00ABCD, clear=0x30,
                         until=BIND_DONE | BIND_ERROR) & BIND_DONE
    reply = eng.fecs_call(be, MTHD_WFI_GOLDEN_SAVE, 0x00ABCD, clear=0x3,
                          until=GOLDEN_DONE | GOLDEN_ERROR)
    assert not reply & GOLDEN_ERROR, "the firmware reported -EIO"
    assert reply & GOLDEN_DONE, f"no outcome bit; SCRATCH[0] = 0x{reply:08x}"

    assert eng.mmctx_runs, "the MMCTX queue was never used"
    named = sum(count for _addr, count in eng.mmctx_runs)
    assert eng.mmctx_saved == named, (
        f"{eng.mmctx_saved} registers read but {named} named by descriptors")
    assert len(eng.mmctx_image) == named
    assert eng.mmctx_loaded == 0, "a save must not write registers"

    # PGRAPH lives at BAR0 0x400000-0x420000; the lists should be in there.
    inside = [a for a, _n in eng.mmctx_runs if 0x400000 <= a < 0x420000]
    assert len(inside) > len(eng.mmctx_runs) // 2, (
        f"only {len(inside)} of {len(eng.mmctx_runs)} runs address PGRAPH: "
        f"{[hex(a) for a, _ in eng.mmctx_runs[:8]]}")

    # Every word in the image must be the register the descriptors said it was.
    # Recomputed from the register file rather than from anything the engine
    # recorded while copying.
    expected = [eng.bar0_read(addr + i * 4)
                for addr, count in eng.mmctx_runs for i in range(count)]
    assert eng.mmctx_image == expected, "the image is not the register file"
    covered = [(addr, count) for addr, count in eng.mmctx_runs
               if addr <= sentinel_reg < addr + count * 4]
    assert covered, (
        f"0x{sentinel_reg:x} is not in any saved run, so this test proves "
        f"nothing about values flowing; pick one that is, from "
        f"{[hex(a) for a, _ in eng.mmctx_runs[:8]]}")
    assert sentinel in eng.mmctx_image, (
        "the sentinel register's value never reached the image")
    assert be._step_fault_pc is None


def test_the_mmctx_engine_round_trips_a_register(booted):
    """The save and load directions are inverses, driven directly.

    This does not go through the firmware: it drives MMCTX the way the firmware
    does and checks the two directions agree, which is the property a context
    switch depends on and the one a write-only model would fail. It is the unit
    test under test_saving_a_golden_context_moves_real_register_state.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        FalconCtxctl, MMCTX_CTRL, MMCTX_QUEUE, MMCTX_DIR,
        MMCTX_START_TRIGGER, MMCTX_STOP_TRIGGER, Q_CNTM1_SHIFT)

    base = 0x404000
    eng = FalconCtxctl("mmctx", 0x0, 0x40000,
                       bar0={base + i * 4: 0xC0DE0000 + i for i in range(4)})
    descriptor = (base >> 2 << 2) | (3 << Q_CNTM1_SHIFT)      # four registers

    eng.hw_write(MMCTX_CTRL, 4, 0x1000 | MMCTX_START_TRIGGER)
    assert eng.hw_read(MMCTX_CTRL, 4) & 0x1F == 0x10, "QFREE must offer room"
    eng.hw_write(MMCTX_QUEUE, 4, descriptor)
    eng.hw_write(MMCTX_CTRL, 4, 0x1000 | MMCTX_STOP_TRIGGER)
    assert not eng.hw_read(MMCTX_CTRL, 4) & MMCTX_STOP_TRIGGER, (
        "STOP_TRIGGER must clear; the firmware waits on it")
    assert eng.mmctx_image == [0xC0DE0000 + i for i in range(4)]

    for i in range(4):
        eng.bar0_write(base + i * 4, 0)
    eng.hw_write(MMCTX_CTRL, 4, 0x1000 | MMCTX_DIR | MMCTX_START_TRIGGER)
    eng.hw_write(MMCTX_QUEUE, 4, descriptor)
    eng.hw_write(MMCTX_CTRL, 4, 0x1000 | MMCTX_DIR | MMCTX_STOP_TRIGGER)
    assert [eng.bar0_read(base + i * 4) for i in range(4)] == \
        [0xC0DE0000 + i for i in range(4)], "the load did not restore the save"

    # A descriptor outside a START/STOP pair contributes nothing. The reason is
    # the backend, not the hardware: it delivers a write by comparing the guest's
    # value against a shadow read back a step later, so a descriptor still
    # sitting in memory can be offered again after the transfer has ended, and
    # taking it would append phantom registers to a finished image.
    before = (eng.mmctx_saved, len(eng.mmctx_image), len(eng.mmctx_runs))
    eng.hw_write(MMCTX_QUEUE, 4, descriptor)
    assert (eng.mmctx_saved, len(eng.mmctx_image), len(eng.mmctx_runs)) == before


# ---------------------------------------------------------------------------
# GPCCS: the same model, the other controller.
# ---------------------------------------------------------------------------

GPCCS_READY = 0x87654321


def test_the_same_model_carries_gpccs():
    """PGRAPH has two context controllers, and this one is not FECS.

    GPCCS is the per-GPC controller (BAR0 0x502000 + idx * 0x8000, against
    FECS's 0x409000). It is a different image with a different register subset --
    intro.rst marks several CTXCTL registers GPC-only or HUB-only -- so booting
    it under the same model is a check on the model rather than on one image.

    Its bring-up ends the same shape as FECS's: write 0x33333333 to MEMIF
    0x28d00, publish a ready value in SCRATCH[0], then sit in a `sleep $p2` idle
    loop draining work queues. The ready value is the thing to assert, because
    it is also how you know real firmware was loaded: a zeroed IMEM runs and
    produces plausible-looking nothing, and GPCCS's own constant is 0x87654321
    where FECS's is 1.
    """
    be, eng, seen = _boot(images=_gpccs())
    assert be._step_fault_pc is None, (
        f"wedged at 0x{be._step_fault_pc:x}" if be._step_fault_pc else "")
    assert eng.scratch0 == GPCCS_READY, (
        f"GPCCS never reported ready; SCRATCH[0] = 0x{eng.scratch0:08x}")
    # It programs its own interrupt setup, which differs from FECS's 0x8704.
    assert eng.intr_en == 0x5004, f"INTR_EN = 0x{eng.intr_en:04x}"
    assert eng.intr_mode == 0x4, (
        "line 2 must be the level-triggered one here too")
    assert be.read_register("iv0") == 0x7C, "GPCCS did not install its vector"
    # Its idle loop: sleep at 0x3381, then the queue drain it calls at 0x3384.
    assert {0x3381, 0x3384} <= seen, (
        f"never reached the idle loop; {len(seen)} distinct PCs")


def test_gpccs_dispatches_a_method_once():
    """The level-triggered line matters on this image too.

    GPCCS programs INTR_MODE = 0x4 exactly as FECS does, so the same latch bug
    would make its handler re-enter forever. Asserting the method is
    acknowledged and the line drops is the same check against the other image.
    """
    from halucinator.peripheral_models.falcon_ctxctl import METHOD_IRQ_LINE

    be, eng, _ = _boot(images=_gpccs())
    eng.fecs_method(0x10, 0)
    assert eng._fifo
    _run(be)
    assert not eng._fifo, "GPCCS never acknowledged the method"
    assert not eng.intr_status() & (1 << METHOD_IRQ_LINE)
    assert be._step_fault_pc is None


# ---------------------------------------------------------------------------
# An external oracle for the MMCTX decode.
# ---------------------------------------------------------------------------

NOUVEAU_ENV = "NOUVEAU_GR_DIR"


def _nouveau_context_registers():
    """PGRAPH context registers as nouveau records them, {addr: {counts}}.

    nvkm/engine/gr/ctx*.c holds `struct gf100_gr_init` tables -- `{ addr, count,
    pitch, data }` -- listing the registers that make up a graphics context.
    Those tables were reverse-engineered from hardware independently of
    anything inside the microcode, which is what makes them worth comparing
    against: they are a second opinion, not a restatement.

    Not vendored. Point NOUVEAU_GR_DIR at nvkm/engine/gr in a nouveau or Linux
    checkout.
    """
    import collections
    import re

    root = os.environ.get(NOUVEAU_ENV)
    if not root:
        pytest.skip(f"set {NOUVEAU_ENV} to nouveau's nvkm/engine/gr")
    entry = re.compile(r"\{\s*(0x[0-9a-f]{6})\s*,\s*(\d+)\s*,\s*0x[0-9a-f]+\s*,")
    table = collections.defaultdict(set)
    files = sorted(pathlib.Path(root).glob("ctx*.c"))
    if not files:
        pytest.skip(f"no ctx*.c under {root}")
    for f in files:
        for m in entry.finditer(f.read_text(errors="ignore")):
            table[int(m.group(1), 16)].add(int(m.group(2)))
    return table


def test_the_saved_register_ranges_agree_with_nouveau_s_own_tables():
    """The MMCTX descriptor decode, checked against a source outside this tree.

    A descriptor packs the register address into bits 2:25 stored shifted right
    by two, and the count into bits 26:31 as count-1. Both are easy to get
    wrong in a way nothing else notices: a wrong shift moves every address by a
    factor of four and the save still completes, and a wrong count field still
    produces a self-consistent image. The firmware's own lists cannot arbitrate
    that, because they are the thing being decoded.

    nouveau's context tables can. They name the same PGRAPH registers, derived
    from hardware by people who never read these descriptors, so agreement on
    *both* address and count is agreement between two independent readings.

    The thresholds are loose on purpose -- the two lists serve different jobs,
    nouveau's initialises a context and the firmware's saves one, so neither is
    a subset of the other. What a broken decode cannot do is land most of its
    addresses on nouveau's.
    """
    from halucinator.peripheral_models.falcon_ctxctl import (
        MTHD_BIND_POINTER, MTHD_WFI_GOLDEN_SAVE, BIND_DONE, BIND_ERROR,
        GOLDEN_DONE, GOLDEN_ERROR)

    table = _nouveau_context_registers()
    assert len(table) > 500, (
        f"only {len(table)} addresses parsed out of nouveau -- the tables did "
        "not parse, so this would pass without checking anything")

    be, eng, _ = _boot()
    assert eng.fecs_call(be, MTHD_BIND_POINTER, 0x00ABCD, clear=0x30,
                         until=BIND_DONE | BIND_ERROR) & BIND_DONE
    assert eng.fecs_call(be, MTHD_WFI_GOLDEN_SAVE, 0x00ABCD, clear=0x3,
                         until=GOLDEN_DONE | GOLDEN_ERROR) & GOLDEN_DONE
    runs = eng.mmctx_runs
    assert runs, "nothing was saved"

    def overlaps(addr):
        for base, counts in table.items():
            if any(base <= addr < base + c * 4 for c in counts):
                return True
        return False

    exact = [(a, n) for a, n in runs if n in table.get(a, ())]
    hit = [(a, n) for a, n in runs if overlaps(a)]
    assert len(exact) >= len(runs) * 0.3, (
        f"only {len(exact)} of {len(runs)} runs match nouveau on address AND "
        f"count: {[(hex(a), n) for a, n in runs[:8]]}")
    assert len(hit) >= len(runs) * 0.6, (
        f"only {len(hit)} of {len(runs)} runs land on a register nouveau calls "
        "part of a graphics context -- the decode is probably wrong")
