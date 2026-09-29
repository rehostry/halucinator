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
SCRATCH0 = 0x20000
SCRATCH1 = 0x20100

# FECS methods, from the driver that issues them.
MTHD_DISCOVER_IMAGE_SIZE = 0x10
MTHD_DISCOVER_ZCULL_IMAGE_SIZE = 0x16
MTHD_SET_WATCHDOG_TIMEOUT = 0x21
MTHD_DISCOVER_PM_IMAGE_SIZE = 0x25
MTHD_DISCOVER_REGLIST_IMAGE_SIZE = 0x30

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
                 bar0: Optional[Dict[int, int]] = None, **kwargs: Any) -> None:
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
        self.scratch0 = 0
        self.scratch1 = 0
        self.methods = []
        self._fifo = []

    def fecs_method(self, method: int, arg: int = 0) -> None:
        """Issue a FECS method the way the host driver does.

        Nouveau clears the scratch register, writes the argument, then writes
        the method -- and the write of the method is what sets the microcode
        going. The answer comes back in the same scratch register, which is
        why it is cleared first: a non-zero read is how the driver knows the
        reply has arrived.
        """
        self.scratch0 = 0
        self.wrcmd_data = arg & 0xFFFFFFFF
        self.wrcmd_cmd = method & 0xFFFFFFFF
        self.methods.append((method, arg))
        self._fifo.append((method & 0xFFFFFFFF, arg & 0xFFFFFFFF))
        self.raise_line(METHOD_IRQ_LINE)
        log.info("%s: FECS method 0x%02x arg 0x%08x", self.name, method, arg)

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
        self._ctrl = ctrl & ~CTRL_GO

    def live_registers(self):
        return tuple(super().live_registers()) + (
            MMIO_CTRL, MMIO_RDVAL, MMIO_WRVAL,
            MMCTX_SAVE_SWBASE, MMCTX_LOAD_SWBASE,
            CURRENT_CTX, NEW_CTX, GRAPH_ENGINE_STATUS, ENGINE_TRIGGER,
            MEM_BASE, MEM_CHAN, MEM_CMD, MEM_TARGET,
            WRCMD_DATA, WRCMD_CMD, SCRATCH0, SCRATCH1,
            FIFO_DATA, FIFO_CMD, FIFO_OCCUPIED, FIFO_ACK,
        )

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
        if offset == SCRATCH0:
            return self.scratch0
        if offset == SCRATCH1:
            return self.scratch1
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
        if offset == WRCMD_DATA:
            self.wrcmd_data = value & 0xFFFFFFFF
            return True
        if offset == WRCMD_CMD:
            self.wrcmd_cmd = value & 0xFFFFFFFF
            return True
        if offset == SCRATCH0:
            self.scratch0 = value & 0xFFFFFFFF
            log.info("%s: scratch0 <- 0x%08x", self.name, self.scratch0)
            return True
        if offset == SCRATCH1:
            self.scratch1 = value & 0xFFFFFFFF
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
