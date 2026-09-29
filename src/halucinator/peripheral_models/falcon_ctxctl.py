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

CTRL_GO = 1 << 31
CTRL_ADDR_MASK = 0x03FFFFFC
CTRL_OP_SHIFT, CTRL_OP_MASK = 29, 0x3
OP_READ, OP_WRITE_A, OP_WRITE_B = 0, 2, 3

# SIGNAL (CC 0x10000) bits the firmware waits on. Names from rnndb ctxctl.xml.
SIGNAL_MMIO_RD_DONE = 1 << 6
SIGNAL_MMIO_WRS_DONE = 1 << 7
SIGNAL_MMCTX_DONE = 1 << 14

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
        self.intr_signal = 0
        self._rdval = 0
        self._wrval = 0
        self._ctrl = 0
        self.mmio_reads = 0
        self.mmio_writes = 0
        self.mmctx_save_base = 0
        self.mmctx_load_base = 0

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
        return super().hw_write(offset, size, value, pc=pc, **kwargs)
