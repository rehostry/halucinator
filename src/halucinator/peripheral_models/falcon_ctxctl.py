"""The CTXCTL engine: a Falcon plus the GPU register bus it talks through.

FECS and GPCCS are the context-switch controllers for PGRAPH. Their IO window
is not a generic falcon engine -- it is divided into the ranges named in
envytools `docs/hw/graph/fermi/ctxctl/intro.rst`, and the one that matters
first is MMIO (0x1c000), a window onto the whole GPU's BAR0 register space.

**The bus protocol, read out of the firmware rather than a document.** The
sequence at GP102 FECS 0x4d7 (write) and 0x68b (read) is::

    write:  MMIO_WRVAL = value
            MMIO_CTRL  = addr | GO | op<<29
            poll MMIO_CTRL until GO clears
    read:   MMIO_CTRL  = addr | GO
            poll MMIO_CTRL until GO clears
            poll SIGNAL until MMIO_RD_DONE
            read MMIO_RDVAL

and the control word is::

    bit 31      GO / busy
    bits 29-30  op: 0 = read, 2 or 3 = write
    bit 26      flag
    bits 2-25   BAR0 address (masked with 0x3fffffc)
    bit 0       flag

**Why this unblocks boot.** GP102 FECS spends its entire run polling BAR0
0x122234 -- 5538 of 5551 control writes in 200k instructions are that one
address. Two call sites decide it: one writes 0x122238 = 0x1e then waits for
0x122234 to read 0, the other writes 0x122230 = 0x1e then waits for 0x122234
to read 0x1e. So those three are a set/clear/status triple on the PRI bus
(PIBUS MMIO_HUB[0] + 0x230/0x234/0x238), and a register file that honours it
is what lets the firmware past its first wall.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .falcon_engine import ENGINE_STATUS as SIGNAL, FalconEngine

log = logging.getLogger(__name__)

# MMIO range (CTXCTL intro.rst: 700/1c000 "MMIO bus access")
MMIO_CTRL = 0x1CA00          # rnndb ctxctl.xml: MMIO_CTRL
MMIO_RDVAL = 0x1CB00         # rnndb: MMIO_RDVAL
MMIO_WRVAL = 0x1CC00         # rnndb: MMIO_WRVAL
MMCTX_SAVE_SWBASE = 0x1C000
MMCTX_LOAD_SWBASE = 0x1C100

# CSREQ range (0x2c000): which channel is loaded, and which one to switch to.
CURRENT_CTX = 0x2C000
NEW_CTX = 0x2C100
CTX_VALID = 1 << 31
CTX_CHAN_MASK = 0x7FFFFFFF

# GRAPH range (0x30000). rnndb calls this ENGINE_STATUS; the falcon's own
# status register is SIGNAL, so it is spelled out here to keep them apart.
GRAPH_ENGINE_STATUS = 0x30000
ENGINE_TRIGGER = 0x30200
ES_CHSW_PENDING = 1 << 0
ES_CHAN_VALID = 1 << 1
ES_CHSW_PULSE = 1 << 3
ES_IDLE_BUSY = 1 << 13

# The falcon interrupt line PFIFO's switch request arrives on. Derived by
# raising each line the firmware enables (INTR_EN = 0x8704 -> lines 2, 8, 9,
# 10, 15) and seeing which one makes it acknowledge the switch.
CHSW_IRQ_LINE = 8

CTRL_GO = 1 << 31
CTRL_ADDR_MASK = 0x03FFFFFC
CTRL_OP_SHIFT, CTRL_OP_MASK = 29, 0x3
OP_READ, OP_WRITE_A, OP_WRITE_B = 0, 2, 3

# A two-bit status field in the same register, at bits 27-28. Turing's FECS
# polls it where Pascal's waits on SIGNAL bit 6 for a read to land:
#
#   GP102 0x51f:  iord MMIO_CTRL; shr 0x1f; loop while bit31   then SIGNAL bit 6
#   TU104 0x875:  iord MMIO_CTRL; extr 0x1b:0x1c; loop until it reads 2
#
# So the read-complete indication moved into the control register between the
# two generations. 2 is what the firmware requires, not a code from any
# documentation -- upstream does not describe this field at all. Reporting it
# unconditionally is safe for Pascal, whose microcode never reads these bits.
CTRL_STATUS_SHIFT, CTRL_STATUS_MASK = 27, 0x3
CTRL_STATUS_DONE = 2

# SIGNAL (CC 0x10000) bits the firmware waits on. Names from rnndb ctxctl.xml.
SIGNAL_MMIO_RD_DONE = 1 << 6
SIGNAL_MMIO_WRS_DONE = 1 << 7
SIGNAL_MMCTX_DONE = 1 << 14
SIGNAL_BAR_0 = 1 << 8
SIGNAL_BAR_1 = 1 << 9

# PRI-bus set/clear/status triples, as (set, status, clear). Derived from the
# firmware's own use: it writes the set register, then polls status until it
# reads back what it set; it writes the clear register, then polls status to 0.
PRI_TRIPLES = {
    0x122230: (0x122230, 0x122234, 0x122238),
}

# Bits hardware clears by itself once it has acted on them. The firmware sets
# one and then polls for it to go away, so a register file that simply stores
# what was written leaves it spinning forever.
#
# 0x404170 is PGRAPH PWR_MODE (rnndb gf100_pgraph, GF119+). GP102 FECS writes
# `mode | 0x10` and polls until bit 4 reads back clear -- bit 4 is the request,
# and hardware drops it when the power-mode change completes.
SELF_CLEARING = {
    0x404170: 0x10,
}

# Values for CTXCTL IO registers the firmware reads before anything has
# written them. A rehost has no real GPC/TPC units behind these, so the honest
# answer to "are the units done?" is yes -- the alternative is a poll that can
# never succeed, which is indistinguishable from a hung rehost.
#
# 0x19000 (BAR0 0x640, unnamed in rnndb): GP102 FECS writes STRAND_CMD and
# then polls this until every bit reads 1, which is the shape of an
# all-units-done mask.
IO_DEFAULTS = {
    0x19000: 0xFFFFFFFF,
}

# FIFO range (0x14000): the command interface the host drives FECS through.
# Nouveau (nvkm/engine/gr/gf100.c) writes BAR0 0x409500 then 0x409504 and polls
# 0x409800 for the answer; those are WRCMD_DATA, WRCMD_CMD and the first
# scratch register, at falcon IO 0x14000, 0x14100 and 0x20000.
WRCMD_DATA = 0x14000
WRCMD_CMD = 0x14100

# The falcon's own method FIFO (rnndb falcon.xml, BAR0 0x64-0x74). The host
# writes WRCMD_DATA/WRCMD_CMD and the hardware pushes an entry here; the
# microcode is interrupted, reads the method and its argument, and acks.
#
# This is how a method actually reaches the firmware. GP102 FECS never reads
# the 0x14000 command window at all -- that is the submission side, which the
# host writes and the hardware forwards. The interrupt handler at 0x2bb reads
# FIFO_CMD and FIFO_DATA when INTR line 2 is pending and writes FIFO_ACK when
# it is done, which is what identifies line 2 as the method-pending line.
FIFO_DATA = 0x01900
FIFO_CMD = 0x01A00
FIFO_OCCUPIED = 0x01C00
FIFO_ACK = 0x01D00
METHOD_IRQ_LINE = 2

# MISC range (0x20000) is the scratch/mailbox block -- BAR0 0x409800 upwards.
# It is the biggest single block of registers the firmware touches, and being
# "misc/unknown stuff" upstream is why that was not obvious.
# Eight scratch registers with the usual set/clear aliases (intro.rst lists
# SCRATCH, SCRATCH_SET and SCRATCH_CLEAR as three consecutive banks from BAR0
# 0x800). The driver uses more than the first one: nouveau writes BAR0 0x409810
# -- SCRATCH[4] -- to hand an instance pointer to methods 0x31 and 0x32, and
# the firmware reports an error code in SCRATCH[6].
#
# The aliases are taken on the documentation's word, and that is worth stating
# plainly. GP102 FECS uses a *second* pair in the same shape -- it writes a bit
# to 0x23200 on entering a routine and the same bit to 0x21200 on leaving it --
# and 0x21200 is inside the bank intro.rst calls SCRATCH_CLEAR, which would make
# the matching set register 0x20a00 rather than 0x23200. Nothing in the image
# ever reads any of these back, so no rehost observation can decide between the
# two readings; it follows the documented one. What depends on the choice is
# only what a host watching the scratch bank would see, and the registers the
# driver actually reads -- SCRATCH[0] for replies, SCRATCH[4] for parameters --
# are written directly and are unaffected either way.
N_SCRATCH = 8
SCRATCH = 0x20000                # BAR0 0x800 + i*4
SCRATCH_SET = 0x20800            # BAR0 0x820 + i*4
SCRATCH_CLEAR = 0x21000          # BAR0 0x840 + i*4

SCRATCH0 = SCRATCH
SCRATCH1 = SCRATCH + 0x100

# The firmware reports why it refused a method here, and raises the upstream
# interrupt. 0x11 is "method not recognised" -- its dispatcher's default case
# at 0x4e6a writes exactly that before signalling.
SCRATCH_ERROR = 6
ERR_UNKNOWN_METHOD = 0x11
INTR_UP_SET = 0x30700

# FECS methods, from the driver that issues them.
# FECS control methods, named after the nouveau helper that issues each one
# (nvkm/engine/gr/gf100.c). The handler each lands in is from the dispatch
# table at 0x4b80 in the GP102 image.
MTHD_BIND_POINTER = 0x03                 # -> 0x4d54
MTHD_WFI_GOLDEN_SAVE = 0x09              # -> 0x4d2e
MTHD_DISCOVER_IMAGE_SIZE = 0x10          # -> 0x4d89 -> 0x2876
MTHD_DISCOVER_ZCULL_IMAGE_SIZE = 0x16    # -> 0x4da1 -> 0x255d
MTHD_SET_WATCHDOG_TIMEOUT = 0x21         # -> 0x4d06
MTHD_DISCOVER_PM_IMAGE_SIZE = 0x25       # -> 0x4dab -> 0x1e23
MTHD_DISCOVER_REGLIST_IMAGE_SIZE = 0x30  # -> 0x4e28 -> 0x47d5
MTHD_SET_REGLIST_BIND_INSTANCE = 0x31    # -> 0x4e32
MTHD_SET_REGLIST_VIRTUAL_ADDRESS = 0x32  # -> 0x4e48

# The scratch register nouveau hands an instance pointer through, for the two
# reglist methods: BAR0 0x409810, which is SCRATCH[4].
SCRATCH_INSTANCE = 4

# Reply bits, from the driver's own poll conditions. bind_pointer masks 0x30
# and then waits for 0x10 (done) or 0x20 (error); wfi_golden_save masks 0x3 and
# waits for 0x1 or 0x2. The discover_* methods instead put a whole value in
# SCRATCH[0] and any non-zero read is the answer.
BIND_DONE, BIND_ERROR = 0x10, 0x20
GOLDEN_DONE, GOLDEN_ERROR = 0x01, 0x02

# MMCTX: the engine that moves the MMIO half of a context image. Names and
# fields from rnndb graph/gf100_pgraph/ctxctl.xml (BAR0 0x700-0x74c).
#
# The firmware drives it as a queue. It writes MMCTX_CTRL with START_TRIGGER and
# a queue limit, then pushes descriptors into MMCTX_QUEUE for as long as
# MMCTX_CTRL.QFREE says there is room, then writes STOP_TRIGGER and waits for
# that bit to clear. Each descriptor names a run of consecutive BAR0 registers;
# the engine copies that run to or from the image, and DIR says which way.
#
# QFREE reading zero is what used to stall this: with no queue modelled the
# firmware waited for room forever, and the golden-context save never finished.
MMCTX_BASE = 0x1C400             # BAR0 0x710
MMCTX_CTRL = 0x1C500             # BAR0 0x714
MMCTX_MULTI_STRIDE = 0x1C600     # BAR0 0x718
MMCTX_MULTI_MASK = 0x1C700       # BAR0 0x71c
MMCTX_QUEUE = 0x1C800            # BAR0 0x720
MMCTX_LOAD_COUNT = 0x1D300       # BAR0 0x74c

MMCTX_QFREE = 0x0000001F
MMCTX_QLIMIT_SHIFT, MMCTX_QLIMIT_MASK = 8, 0x1F
MMCTX_DIR = 1 << 16             # 0 save, 1 load
MMCTX_START_TRIGGER = 1 << 17
MMCTX_STOP_TRIGGER = 1 << 18

# MMCTX_QUEUE descriptor fields.
Q_BASE_EN = 1 << 0
Q_MULTI_EN = 1 << 1
Q_ADDR_SHIFT, Q_ADDR_MASK = 2, 0x00FFFFFF
Q_CNTM1_SHIFT, Q_CNTM1_MASK = 26, 0x3F

# STRAND range (0x24000) plus the two strand registers that live in MISC.
# Names and offsets from rnndb graph/gf100_pgraph/ctxctl.xml; falcon IO address
# is the BAR0 offset times 0x40, and an indexed register's strand number adds
# index*4 on top of that.
#
# A strand is one serial chain of context state. The firmware sizes the context
# image by walking the nine STRAND_WORDS registers (its loop at 0xa4a runs
# 0x24400..0x24420 inclusive, which is what fixes the count at nine), giving
# each non-empty strand a save and load base in the image as it goes.
STRANDS_CNT = 0x22000            # BAR0 0x880
STRANDS_CMD_MASK = 0x22100       # BAR0 0x884
STRAND_SAVE_SWBASE = 0x24200     # BAR0 0x908, per strand
STRAND_LOAD_SWBASE = 0x24300     # BAR0 0x90c, per strand
STRAND_WORDS = 0x24400           # BAR0 0x910, per strand
STRAND_DATA = 0x24600            # BAR0 0x918
STRAND_SELECT = 0x24700          # BAR0 0x91c
STRAND_STATUS = 0x24900          # BAR0 0x924
STRAND_CMD = 0x24A00             # BAR0 0x928
STRAND_FILTER = 0x24F00          # BAR0 0x93c

N_STRANDS = 9

# Index 0x3f addresses every strand at once. The firmware only ever uses the
# broadcast form for commands -- `mov $r9 0x24afc` is STRAND_CMD index 0x3f --
# which is why the command registers appear at a +0xfc offset throughout.
STRAND_BROADCAST = 0xFC

# STRAND_CMD values (rnndb). The firmware issues ENABLE/ACTIVATE_FILTER around
# each walk and DISABLE at the end; GP102 also uses 0x11, which upstream does
# not name.
STRAND_CMD_SEEK, STRAND_CMD_GET_INFO = 1, 2
STRAND_CMD_SAVE, STRAND_CMD_LOAD = 3, 4
STRAND_CMD_ENABLE, STRAND_CMD_DISABLE = 0xC, 0xD

# MEMIF range (0x28000): the memory interface the context save and restore run
# through. Names from rnndb ctxctl.xml.
MEM_BASE = 0x28100
MEM_CHAN = 0x28300
MEM_CMD = 0x28400
MEM_TARGET = 0x28800

# Command fields hardware clears once it has carried the command out. The
# firmware writes MEM_CMD and then spins on `MEM_CMD & 0x1f` until it reads
# zero, so a register that simply stores the command never lets it continue.
IO_SELF_CLEARING = {
    MEM_CMD: 0x1F,
}

# The same question on the BAR0 side. Each of these is polled until every bit
# reads 1 -- the firmware is waiting for a set of units to report done, and
# with no units modelled the only answer that lets it proceed is "all of them".
BAR0_DEFAULTS = {
    0x41a640: 0xFFFFFFFF,
}


class FalconCtxctl(FalconEngine):
    """A Falcon engine plus the CTXCTL MMIO bus and a GPU register file."""

    def __init__(self, name: str, address: int, size: int,
                 bar0: Optional[Dict[int, int]] = None,
                 strand_words: Optional[list] = None, **kwargs: Any) -> None:
        super().__init__(name, address, size, **kwargs)
        # The GPU's BAR0 register space as the firmware sees it through the
        # MMIO bus. Not the falcon's own IO window -- a different space
        # entirely, reached only through MMIO_CTRL.
        self.bar0: Dict[int, int] = dict(bar0 or {})
        # SIGNAL (CC 0x10000) is the base engine's ENGINE_STATUS. Its static
        # ready_bits are what the firmware's early polls need; these are the
        # bits the MMIO bus raises as it completes work.
        # Barriers synchronise the HUB controller (FECS) with the per-GPC
        # controllers (GPCCS). With no GPC units modelled there is nobody else
        # to arrive, so every barrier is satisfied the moment it is asked
        # about. Left clear, the firmware waits at its first barrier forever.
        #
        # This is a statement about the modelled system, not about hardware: a
        # rehost that later co-simulates GPCCS must drive these from the real
        # barrier state instead.
        self.intr_signal = SIGNAL_BAR_0 | SIGNAL_BAR_1
        self._rdval = 0
        self._wrval = 0
        self._ctrl = 0
        self.mmio_reads = 0
        self.mmio_writes = 0
        self.mmctx_save_base = 0
        self.mmctx_load_base = 0
        # Context-switch state, as PFIFO would present it.
        self.current_ctx = 0
        self.new_ctx = 0
        self.engine_status = 0
        self.engine_trigger = 0
        self.switch_requests = 0
        self.mem_base = 0
        self.mem_chan = 0
        self.mem_target = 0
        self.mem_commands = []
        self.wrcmd_data = 0
        self.wrcmd_cmd = 0
        self.scratch = [0] * N_SCRATCH
        self.methods = []
        self._fifo = []
        # Strand state. STRAND_WORDS is an input to this model, not an output
        # of it: on hardware the strand unit reports how much state each chain
        # holds, and nothing in a rehost can derive that. Zero for every strand
        # is the honest default -- it says "no context state is modelled" --
        # and a caller with measured values passes them in. What the model does
        # guarantee is that the firmware's own arithmetic runs over whatever it
        # is given, so the size FECS reports moves with these numbers.
        self.strand_words = list(strand_words or [0] * N_STRANDS)[:N_STRANDS]
        self.strand_words += [0] * (N_STRANDS - len(self.strand_words))
        self.strand_save_base = [0] * N_STRANDS
        self.strand_load_base = [0] * N_STRANDS
        self.strand_cmds = []
        self.strand_select = 0
        self.strand_filter = 0
        self.strand_data = 0
        # MMCTX. `ctx_memory` stands in for the memory the engine reaches
        # through MEMIF, word-addressed. A save writes register values there
        # starting at MMCTX_SAVE_SWBASE and a load reads them back from
        # MMCTX_LOAD_SWBASE, so the base registers decide *which* image a
        # transfer touches -- two contexts at different bases do not collide,
        # and a load can only return what a save to that base put there.
        # `mmctx_runs` records the register ranges each descriptor named, which
        # is the firmware's own register list made observable.
        self.ctx_memory: Dict[int, int] = {}
        self.mmctx_base = 0
        self.mmctx_multi_stride = 0
        self.mmctx_multi_mask = 0
        self.mmctx_load_count = 0
        self.mmctx_qlimit = 0
        self.mmctx_dir = 0
        self.mmctx_running = False
        self.mmctx_cursor = 0
        self.mmctx_runs = []
        self.mmctx_saved = 0
        self.mmctx_loaded = 0
        # (memory address, word count) of the most recent transfer.
        self.mmctx_last = (0, 0)

    # -- the scratch bank, as the driver sees it ---------------------------
    #
    # SCRATCH[0] is the reply register for every FECS control method, so it is
    # worth naming. The rest are parameters and status.

    @property
    def scratch0(self) -> int:
        return self.scratch[0]

    @scratch0.setter
    def scratch0(self, value: int) -> None:
        self.scratch[0] = value & 0xFFFFFFFF

    @property
    def scratch1(self) -> int:
        return self.scratch[1]

    @scratch1.setter
    def scratch1(self, value: int) -> None:
        self.scratch[1] = value & 0xFFFFFFFF

    def mask_scratch(self, index: int, mask: int, value: int = 0) -> None:
        """`nvkm_mask` on one scratch register, which is how nouveau clears it.

        Two of the methods clear only their own reply bits rather than the whole
        register -- bind_pointer masks 0x30, wfi_golden_save masks 0x3 -- so a
        blanket zero would be a different operation from the one the driver
        performs.
        """
        cur = self.scratch[index] & ~(mask & 0xFFFFFFFF)
        self.scratch[index] = (cur | (value & mask)) & 0xFFFFFFFF

    def fecs_method(self, method: int, arg: int = 0,
                    clear: Optional[int] = 0xFFFFFFFF) -> None:
        """Issue a FECS method the way the host driver does.

        Nouveau clears the reply register, writes the argument to BAR0 0x409500
        and then the method to 0x409504 -- and the write of the method is what
        sets the microcode going. The answer comes back in SCRATCH[0], which is
        why it is cleared first: the driver is polling that register and only a
        change tells it the reply has arrived.

        `clear` is the mask cleared from SCRATCH[0] before submitting, because
        the driver does not always clear the whole register (see mask_scratch).
        Pass None to leave it untouched.
        """
        if clear:
            self.mask_scratch(0, clear)
        self.wrcmd_data = arg & 0xFFFFFFFF
        self.wrcmd_cmd = method & 0xFFFFFFFF
        self.methods.append((method, arg))
        self._fifo.append((method & 0xFFFFFFFF, arg & 0xFFFFFFFF))
        self.raise_line(METHOD_IRQ_LINE)
        log.info("%s: FECS method 0x%02x arg 0x%08x", self.name, method, arg)

    def fecs_call(self, backend, method: int, arg: int = 0,
                  clear: Optional[int] = 0xFFFFFFFF,
                  until: Optional[int] = None,
                  steps: int = 200_000) -> int:
        """Issue a method, let the microcode run, and return SCRATCH[0].

        The driver-shaped call: submit and poll. It runs the guest because the
        reply only exists once the firmware has produced it -- a rehost has no
        other clock.

        `until` is the bitmask that means "answered", matching what the driver
        polls for: the discover_* methods put a whole value in SCRATCH[0] and
        treat any non-zero read as the reply, which is the default, while
        bind_pointer and wfi_golden_save watch particular bits. Getting this
        wrong the other way is a real hazard -- SCRATCH[0] already has the ready
        bit set, so "non-zero" would look answered before the method ran at all.

        Returns SCRATCH[0] when the condition is met, the guest stops, or the
        budget runs out; a method that does not reply simply spends the budget.
        """
        mask = 0xFFFFFFFF if until is None else until
        self.fecs_method(method, arg, clear=clear)
        for _ in range(steps):
            backend.step()
            if getattr(backend, "_halted", False) or \
                    getattr(backend, "_step_fault_pc", None) is not None:
                break
            if not self._fifo and (self.scratch[0] & mask):
                break                      # acknowledged and answered
        return self.scratch[0]

    # -- the host side -----------------------------------------------------

    def request_context_switch(self, channel: int, valid: bool = True) -> None:
        """Ask the firmware to switch to `channel`, as PFIFO does.

        PFIFO publishes the channel in NEW_CTX and raises CHSW_PENDING; the
        microcode notices, saves whatever CURRENT_CTX names and loads the new
        one. Nothing else in this model drives that -- without a caller the
        firmware sits in its idle loop forever, which is the correct behaviour
        for a GPU with no work queued.
        """
        self.new_ctx = (channel & CTX_CHAN_MASK) | (CTX_VALID if valid else 0)
        self.engine_status |= ES_CHSW_PENDING | (ES_CHAN_VALID if valid else 0)
        self.switch_requests += 1
        # The request only reaches the microcode as an interrupt. Line 8 is the
        # one that carries it: raising each enabled engine line in turn from an
        # identical post-boot state, line 8 is the only one after which the
        # firmware writes CURRENT_CTX, and it executes 393 instructions it
        # never reaches otherwise.
        self.raise_line(CHSW_IRQ_LINE)
        log.info("%s: context switch requested -> channel 0x%x",
                 self.name, channel & CTX_CHAN_MASK)

    # -- the GPU register file --------------------------------------------

    def bar0_read(self, addr: int) -> int:
        if addr in self.bar0:
            return self.bar0[addr] & 0xFFFFFFFF
        return BAR0_DEFAULTS.get(addr, 0) & 0xFFFFFFFF

    def bar0_write(self, addr: int, value: int) -> None:
        value &= 0xFFFFFFFF
        for _key, (set_r, status_r, clear_r) in PRI_TRIPLES.items():
            if addr == set_r:
                self.bar0[status_r] = self.bar0.get(status_r, 0) | value
                log.debug("%s: PRI set 0x%06x |= 0x%x -> status 0x%x",
                          self.name, addr, value, self.bar0[status_r])
                return
            if addr == clear_r:
                self.bar0[status_r] = self.bar0.get(status_r, 0) & ~value
                log.debug("%s: PRI clear 0x%06x &= ~0x%x -> status 0x%x",
                          self.name, addr, value, self.bar0[status_r])
                return
            if addr == status_r:
                # Status is a readback; hardware ignores direct writes.
                return
        self.bar0[addr] = value & ~SELF_CLEARING.get(addr, 0)

    # -- MMIO bus ----------------------------------------------------------

    def _run_ctrl(self, ctrl: int) -> None:
        addr = ctrl & CTRL_ADDR_MASK
        op = (ctrl >> CTRL_OP_SHIFT) & CTRL_OP_MASK
        if op == OP_READ:
            self._rdval = self.bar0_read(addr)
            self.mmio_reads += 1
            self.intr_signal |= SIGNAL_MMIO_RD_DONE
            log.debug("%s: MMIO read  0x%06x -> 0x%08x", self.name, addr,
                      self._rdval)
        else:
            self.bar0_write(addr, self._wrval)
            self.mmio_writes += 1
            self.intr_signal |= SIGNAL_MMIO_WRS_DONE
            log.debug("%s: MMIO write 0x%06x <- 0x%08x", self.name, addr,
                      self._wrval)
        # The request completes inside the instruction, so GO is already clear
        # by the time the firmware polls it. Hardware would clear it a little
        # later; a firmware that depended on observing GO *set* would spin,
        # and none does -- every site polls for it to clear.
        self._ctrl = ((ctrl & ~CTRL_GO
                       & ~(CTRL_STATUS_MASK << CTRL_STATUS_SHIFT))
                      | (CTRL_STATUS_DONE << CTRL_STATUS_SHIFT))

    def level_source(self, line: int):
        """The method-FIFO line's source is "the FIFO is not empty".

        The firmware's own init writes INTR_MODE = 0x4, so line 2 -- and only
        line 2 -- is level-triggered, and its handler at 0x2bb acknowledges by
        writing FIFO_ACK, never INTR_CLEAR. Popping the entry is therefore what
        deasserts the interrupt.
        """
        if line == METHOD_IRQ_LINE:
            return bool(self._fifo)
        return None

    # -- MMCTX: the MMIO half of a context image --------------------------

    def _mmctx_ctrl(self) -> int:
        """MMCTX_CTRL as the firmware reads it back.

        QFREE is the number of free queue slots. Nothing in this model takes
        time, so a descriptor is consumed the moment it is written and the queue
        is always empty -- QFREE therefore reads as the whole limit. Both
        triggers read zero because both have already completed; the firmware
        waits for STOP_TRIGGER to clear and would wait forever otherwise.
        """
        free = self.mmctx_qlimit & MMCTX_QFREE
        return (free
                | ((self.mmctx_qlimit & MMCTX_QLIMIT_MASK) << MMCTX_QLIMIT_SHIFT)
                | (MMCTX_DIR if self.mmctx_dir else 0))

    def _mmctx_image_base(self) -> int:
        """The memory address this transfer works from.

        Both base registers are stored shifted right by eight (rnndb marks them
        `shr="8"`), so the value names a 256-byte unit.
        """
        base = self.mmctx_load_base if self.mmctx_dir else self.mmctx_save_base
        return (base << 8) & 0xFFFFFFFF

    @property
    def mmctx_image(self):
        """The words of the most recent transfer, read back out of memory."""
        base, words = self.mmctx_last
        return [self.ctx_memory.get((base + i * 4) & 0xFFFFFFFF, 0)
                for i in range(words)]

    def _mmctx_write_ctrl(self, value: int) -> None:
        self.mmctx_qlimit = (value >> MMCTX_QLIMIT_SHIFT) & MMCTX_QLIMIT_MASK
        self.mmctx_dir = 1 if value & MMCTX_DIR else 0
        if value & MMCTX_START_TRIGGER:
            self.mmctx_running = True
            self.mmctx_cursor = 0
            self.mmctx_runs = []
            self.mmctx_last = (self._mmctx_image_base(), 0)
            log.info("%s: MMCTX %s started at 0x%08x, queue limit %d", self.name,
                     "load" if self.mmctx_dir else "save",
                     self._mmctx_image_base(), self.mmctx_qlimit)
        if value & MMCTX_STOP_TRIGGER:
            self.mmctx_running = False
            self.mmctx_last = (self._mmctx_image_base(), self.mmctx_cursor)
            log.info("%s: MMCTX %s done, %d registers, %d words at 0x%08x",
                     self.name, "load" if self.mmctx_dir else "save",
                     sum(n for _a, n in self.mmctx_runs), self.mmctx_cursor,
                     self._mmctx_image_base())

    def _mmctx_queue(self, desc: int) -> None:
        """Carry out one queue descriptor.

        A descriptor names `CNTM1 + 1` consecutive BAR0 registers starting at
        ADDR (which is stored shifted right by two), optionally offset by
        MMCTX_BASE. With MULTI_EN the same run is repeated at MULTI_STRIDE
        intervals, once per set bit in MULTI_MASK -- that is how one descriptor
        covers the same registers in every GPC.
        """
        if not self.mmctx_running:
            # Outside a START/STOP pair there is no transfer to contribute to.
            # This guard is about the backend rather than the hardware: writes
            # are delivered by comparing the guest's value against a shadow read
            # back a step later, so a descriptor value still sitting in memory
            # can be re-offered. Refusing it keeps a stale word from appending
            # phantom registers to a finished image.
            log.debug("%s: MMCTX descriptor 0x%08x outside a transfer, ignored",
                      self.name, desc)
            return
        addr = ((desc >> Q_ADDR_SHIFT) & Q_ADDR_MASK) << 2
        count = ((desc >> Q_CNTM1_SHIFT) & Q_CNTM1_MASK) + 1
        if desc & Q_BASE_EN:
            addr += self.mmctx_base
        starts = [addr]
        if desc & Q_MULTI_EN and self.mmctx_multi_mask:
            starts = [addr + i * self.mmctx_multi_stride
                      for i in range(self.mmctx_multi_mask.bit_length())
                      if (self.mmctx_multi_mask >> i) & 1]
        for start in starts:
            self.mmctx_runs.append((start, count))
            for i in range(count):
                reg = start + i * 4
                where = (self._mmctx_image_base()
                         + self.mmctx_cursor * 4) & 0xFFFFFFFF
                if self.mmctx_dir:
                    if where in self.ctx_memory:
                        self.bar0_write(reg, self.ctx_memory[where])
                        self.mmctx_loaded += 1
                    # A word never saved is not restored. Writing zero would be
                    # worse than leaving the register alone: it would look like
                    # a successful restore of a context that was never captured.
                else:
                    self.ctx_memory[where] = self.bar0_read(reg)
                    self.mmctx_saved += 1
                self.mmctx_cursor += 1

    def live_registers(self):
        return tuple(super().live_registers()) + (
            MMIO_CTRL, MMIO_RDVAL, MMIO_WRVAL,
            MMCTX_SAVE_SWBASE, MMCTX_LOAD_SWBASE,
            MMCTX_BASE, MMCTX_CTRL, MMCTX_MULTI_STRIDE, MMCTX_MULTI_MASK,
            MMCTX_QUEUE, MMCTX_LOAD_COUNT,
            CURRENT_CTX, NEW_CTX, GRAPH_ENGINE_STATUS, ENGINE_TRIGGER,
            MEM_BASE, MEM_CHAN, MEM_CMD, MEM_TARGET,
            WRCMD_DATA, WRCMD_CMD,
            FIFO_DATA, FIFO_CMD, FIFO_OCCUPIED, FIFO_ACK,
            STRANDS_CNT, STRAND_STATUS,
            STRAND_CMD + STRAND_BROADCAST, STRAND_FILTER + STRAND_BROADCAST,
            STRAND_SELECT + STRAND_BROADCAST, STRAND_DATA + STRAND_BROADCAST,
        ) + tuple(SCRATCH + i * 0x100 for i in range(N_SCRATCH)) \
          + tuple(SCRATCH_SET + i * 0x100 for i in range(N_SCRATCH)) \
          + tuple(SCRATCH_CLEAR + i * 0x100 for i in range(N_SCRATCH)) \
          + tuple(STRAND_WORDS + i * 4 for i in range(N_STRANDS)) \
          + tuple(STRAND_SAVE_SWBASE + i * 4 for i in range(N_STRANDS)) \
          + tuple(STRAND_LOAD_SWBASE + i * 4 for i in range(N_STRANDS))

    def hw_read(self, offset: int, size: int, pc: int = 0xBAADBAAD,
                **kwargs: Any) -> int:
        if offset == SIGNAL:
            return (self.ready_bits | self.intr_signal) & 0xFFFFFFFF
        if offset == MMIO_CTRL:
            return self._ctrl
        if offset == MMIO_RDVAL:
            return self._rdval
        if offset == MMIO_WRVAL:
            return self._wrval
        if offset == MMCTX_CTRL:
            return self._mmctx_ctrl()
        if offset == MMCTX_BASE:
            return self.mmctx_base
        if offset == MMCTX_MULTI_STRIDE:
            return self.mmctx_multi_stride
        if offset == MMCTX_MULTI_MASK:
            return self.mmctx_multi_mask
        if offset == MMCTX_QUEUE:
            return 0                         # write-only descriptor port
        if offset == MMCTX_LOAD_COUNT:
            return self.mmctx_load_count
        if offset == MMCTX_SAVE_SWBASE:
            return self.mmctx_save_base
        if offset == MMCTX_LOAD_SWBASE:
            return self.mmctx_load_base
        if offset == FIFO_CMD:
            return self._fifo[0][0] if self._fifo else 0
        if offset == FIFO_DATA:
            return self._fifo[0][1] if self._fifo else 0
        if offset == FIFO_OCCUPIED:
            return len(self._fifo)
        if offset == FIFO_ACK:
            return 0
        if offset == WRCMD_DATA:
            return self.wrcmd_data
        if offset == WRCMD_CMD:
            return self.wrcmd_cmd
        if SCRATCH <= offset < SCRATCH + N_SCRATCH * 0x100 and \
                offset % 0x100 == 0:
            return self.scratch[(offset - SCRATCH) // 0x100]
        if (SCRATCH_SET <= offset < SCRATCH_SET + N_SCRATCH * 0x100
                or SCRATCH_CLEAR <= offset < SCRATCH_CLEAR + N_SCRATCH * 0x100) \
                and offset % 0x100 == 0:
            # Write-only alias ports. Reading the merged value back through them
            # would also make the backend's shadow miss a repeat: it delivers a
            # write by noticing the guest's value differs from what it last
            # read, so a port that reads back what was just written swallows the
            # next identical write.
            return 0
        if offset == MEM_BASE:
            return self.mem_base
        if offset == MEM_CHAN:
            return self.mem_chan
        if offset == MEM_TARGET:
            return self.mem_target
        if offset == CURRENT_CTX:
            return self.current_ctx
        if offset == NEW_CTX:
            return self.new_ctx
        if offset == GRAPH_ENGINE_STATUS:
            return self.engine_status
        if offset == ENGINE_TRIGGER:
            return self.engine_trigger
        if offset == STRANDS_CNT:
            return N_STRANDS
        if offset == STRAND_STATUS:
            # LAST_CMD, bits 0:3. Nothing is ever busy in this model, so the
            # status only has to report what was asked for last.
            return (self.strand_cmds[-1] & 0xF) if self.strand_cmds else 0
        if STRAND_WORDS <= offset < STRAND_WORDS + N_STRANDS * 4:
            return self.strand_words[(offset - STRAND_WORDS) // 4] & 0xFFFFFFFF
        if STRAND_SAVE_SWBASE <= offset < STRAND_SAVE_SWBASE + N_STRANDS * 4:
            return self.strand_save_base[(offset - STRAND_SAVE_SWBASE) // 4]
        if STRAND_LOAD_SWBASE <= offset < STRAND_LOAD_SWBASE + N_STRANDS * 4:
            return self.strand_load_base[(offset - STRAND_LOAD_SWBASE) // 4]
        if offset in IO_DEFAULTS and offset not in self.registers:
            return IO_DEFAULTS[offset]
        return super().hw_read(offset, size, pc=pc, **kwargs)

    def hw_write(self, offset: int, size: int, value: int,
                 pc: int = 0xBAADBAAD, **kwargs: Any) -> bool:
        if offset == MMIO_CTRL:
            self._run_ctrl(value & 0xFFFFFFFF)
            return True
        if offset == MMIO_WRVAL:
            self._wrval = value & 0xFFFFFFFF
            return True
        if offset == MMIO_RDVAL:
            return True                      # readback only
        if offset == MMCTX_CTRL:
            self._mmctx_write_ctrl(value & 0xFFFFFFFF)
            return True
        if offset == MMCTX_QUEUE:
            self._mmctx_queue(value & 0xFFFFFFFF)
            return True
        if offset == MMCTX_BASE:
            self.mmctx_base = value & 0xFFFFFFFF
            return True
        if offset == MMCTX_MULTI_STRIDE:
            self.mmctx_multi_stride = value & 0xFFFFFFFF
            return True
        if offset == MMCTX_MULTI_MASK:
            self.mmctx_multi_mask = value & 0xFFFFFFFF
            return True
        if offset == MMCTX_LOAD_COUNT:
            self.mmctx_load_count = value & 0xFFFFFFFF
            return True
        if offset == MMCTX_SAVE_SWBASE:
            self.mmctx_save_base = value & 0xFFFFFFFF
            return True
        if offset == MMCTX_LOAD_SWBASE:
            self.mmctx_load_base = value & 0xFFFFFFFF
            return True
        if offset == FIFO_ACK:
            if self._fifo:
                done = self._fifo.pop(0)
                log.info("%s: method 0x%02x acknowledged", self.name, done[0])
            return True
        if offset == STRAND_CMD + STRAND_BROADCAST:
            self.strand_cmds.append(value & 0xFFFFFFFF)
            return True
        if offset == STRAND_FILTER + STRAND_BROADCAST:
            self.strand_filter = value & 0xFFFFFFFF
            return True
        if offset == STRAND_SELECT + STRAND_BROADCAST:
            self.strand_select = value & 0xFFFFFFFF
            return True
        if offset == STRAND_DATA + STRAND_BROADCAST:
            self.strand_data = value & 0xFFFFFFFF
            return True
        if STRAND_SAVE_SWBASE <= offset < STRAND_SAVE_SWBASE + N_STRANDS * 4:
            self.strand_save_base[(offset - STRAND_SAVE_SWBASE) // 4] = value & 0xFFFFFFFF
            return True
        if STRAND_LOAD_SWBASE <= offset < STRAND_LOAD_SWBASE + N_STRANDS * 4:
            self.strand_load_base[(offset - STRAND_LOAD_SWBASE) // 4] = value & 0xFFFFFFFF
            return True
        if STRAND_WORDS <= offset < STRAND_WORDS + N_STRANDS * 4:
            return True                      # reported by hardware, not set
        if offset == WRCMD_DATA:
            self.wrcmd_data = value & 0xFFFFFFFF
            return True
        if offset == WRCMD_CMD:
            self.wrcmd_cmd = value & 0xFFFFFFFF
            return True
        if SCRATCH <= offset < SCRATCH + N_SCRATCH * 0x100 and \
                offset % 0x100 == 0:
            i = (offset - SCRATCH) // 0x100
            self.scratch[i] = value & 0xFFFFFFFF
            log.info("%s: scratch%d <- 0x%08x", self.name, i, self.scratch[i])
            return True
        if SCRATCH_SET <= offset < SCRATCH_SET + N_SCRATCH * 0x100 and \
                offset % 0x100 == 0:
            self.scratch[(offset - SCRATCH_SET) // 0x100] |= value & 0xFFFFFFFF
            return True
        if SCRATCH_CLEAR <= offset < SCRATCH_CLEAR + N_SCRATCH * 0x100 and \
                offset % 0x100 == 0:
            i = (offset - SCRATCH_CLEAR) // 0x100
            self.scratch[i] &= ~value & 0xFFFFFFFF
            return True
        if offset == MEM_BASE:
            self.mem_base = value & 0xFFFFFFFF
            return True
        if offset == MEM_CHAN:
            self.mem_chan = value & 0xFFFFFFFF
            return True
        if offset == MEM_TARGET:
            self.mem_target = value & 0xFFFFFFFF
            return True
        if offset == MEM_CMD:
            cmd = value & 0xFFFFFFFF
            self.mem_commands.append((cmd, self.mem_chan, self.mem_base,
                                      self.mem_target))
            log.info("%s: MEM_CMD 0x%x (chan 0x%08x base 0x%08x target 0x%x)",
                     self.name, cmd & 0x1F, self.mem_chan, self.mem_base,
                     self.mem_target)
            self.registers[MEM_CMD] = cmd & ~IO_SELF_CLEARING[MEM_CMD]
            return True
        if offset == CURRENT_CTX:
            # The microcode publishes the context it has loaded. Writing it is
            # how it tells the host the switch is done.
            self.current_ctx = value & 0xFFFFFFFF
            log.info("%s: CURRENT_CTX <- 0x%08x", self.name, self.current_ctx)
            return True
        if offset == NEW_CTX:
            self.new_ctx = value & 0xFFFFFFFF
            return True
        if offset == GRAPH_ENGINE_STATUS:
            # Acknowledging a switch clears the pending bit.
            self.engine_status = value & 0xFFFFFFFF
            return True
        if offset == ENGINE_TRIGGER:
            self.engine_trigger = value & 0xFFFFFFFF
            return True
        return super().hw_write(offset, size, value, pc=pc, **kwargs)
