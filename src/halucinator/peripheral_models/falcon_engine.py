# Copyright 2026 Christopher Wright
"""NVIDIA Falcon engine I/O space.

Models the parts of a Falcon engine's MMIO that its microcode actually touches
during bring-up, so a rehost does not need a harness poking registers from
outside. Three groups, each derived from a documented register map or from the
firmware's own code:

**Interrupt controller** (envytools ``docs/hw/falcon/intr.rst``) --
set/clear/status triples, implemented with their real aliasing semantics::

    I[0x00000] INTR_SET        write-only; 1 bits make lines pending
    I[0x00100] INTR_CLEAR      write-only; 1 bits acknowledge
    I[0x00200] INTR            status, read-only
    I[0x00400] INTR_EN_SET     write-only
    I[0x00500] INTR_EN_CLEAR   write-only
    I[0x00600] INTR_EN         status, read-only

Writes to SET/CLEAR for a line configured level-triggered are ignored, as on
hardware. (``FalconIrqController`` writes the status register directly instead;
that is deliberate -- it is an outside-in injection path, and this class is
where the hardware's aliasing actually belongs.)

**Engine status** -- the register the microcode polls for readiness. Which bits
mean "ready" is read off the firmware's own poll loops rather than assumed: a
loop of the shape ``iord`` / ``and`` or ``shr`` / branch-back states the value
needed to leave it. For GP102 FECS and GPCCS both, that is bits 0x40, 0x80 and
0x4000 of ``I[0x10000]``, and both images agree independently.

**Timers** (``timer.rst``) -- the periodic timer counts ``PERIODIC_TIME`` down
while ``PERIODIC_ENABLE`` bit 0 is set, raises line 0 at zero and reloads from
``PERIODIC_PERIOD``; the watchdog is the one-shot equivalent on line 1. These
are what let the model raise an interrupt on its own rather than waiting for a
harness to poke one in. Note the unit: hardware counts *clock cycles* and a
rehost steps *instructions*, so ``cycles_per_step`` scales between them and is
a modelling choice, not a hardware fact.

**Indirect-register mailbox** -- the microcode reaches registers outside its own
I/O window through a request/response pair, per its access helper::

    iowr I[0x1ca00] <address | flags>   submit
    poll I[0x1ca00] until bit31 clear   not busy
    poll I[0x10000] until bit6 set      result ready
    iord I[0x1cb00]                     take the value

``registers`` supplies values for those indirect addresses; anything not listed
reads as zero, which is the honest default for a register nothing has modelled.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from halucinator.peripherals.hal_peripheral import HalPeripheral as AvatarPeripheral

log = logging.getLogger(__name__)

# Interrupt controller
INTR_SET, INTR_CLEAR, INTR = 0x00000, 0x00100, 0x00200
INTR_MODE = 0x00300
INTR_EN_SET, INTR_EN_CLEAR, INTR_EN = 0x00400, 0x00500, 0x00600
INTR_ROUTING = 0x00700

# intr.rst: INTR_MODE bits 0-15 are per-line trigger modes, 0 edge and 1
# level, and the register reads 0xfc04 out of reset.
INTR_MODE_RESET = 0xFC04

# intr.rst: a line's routing is two bits, taken from bit N of each half of
# INTR_ROUTING. 0 and 2 are the falcon's own vectors; 1 and 3 go to the host
# and never reach the microcontroller at all.
ROUTE_VECTOR0, ROUTE_PMC_HOST, ROUTE_VECTOR1, ROUTE_PMC_NRHOST = 0, 1, 2, 3

# Timers (docs/hw/falcon/timer.rst)
PERIODIC_PERIOD, PERIODIC_TIME, PERIODIC_ENABLE = 0x00800, 0x00900, 0x00A00
TIME_LOW, TIME_HIGH = 0x00B00, 0x00C00
WATCHDOG_TIME, WATCHDOG_ENABLE = 0x00D00, 0x00E00

LINE_PERIODIC, LINE_WATCHDOG = 0, 1

# Engine status, and the mailbox pair
ENGINE_STATUS = 0x10000
MAILBOX_REQ, MAILBOX_RESP = 0x1CA00, 0x1CB00

# Bits of ENGINE_STATUS the GP102 graphics-block microcode waits on. Derived
# from its poll loops, not assumed: 0x80 at fecs:0x53b, 0x40 at fecs:0x6ce,
# 0x4000 at fecs:0xb9e -- and the same three in gpccs.
DEFAULT_READY_BITS = 0x40 | 0x80 | 0x4000        # == 0x40C0

# The request word carries the target address in its low bits and control in
# the top (the helper ORs in `r11` and `r12 << 26`).
REQ_ADDR_MASK = 0x03FFFFFF
BUSY_BIT = 1 << 31


class FalconEngine(AvatarPeripheral):
    """Falcon engine MMIO: interrupt controller, status, indirect mailbox."""

    def __init__(self, name: str, address: int, size: int,
                 registers: Optional[Dict[int, int]] = None,
                 ready_bits: int = DEFAULT_READY_BITS,
                 level_lines: Optional[int] = None,
                 cycles_per_step: int = 1,
                 **kwargs: Any) -> None:
        AvatarPeripheral.__init__(self, name, address, size)
        self.registers: Dict[int, int] = dict(registers or {})
        self.ready_bits = ready_bits
        # Lines wired level-triggered; SET/CLEAR writes to these are ignored.
        # Defaults match intr.rst: line 2 (FIFO) plus the engine-specific
        # 10..15 range.
        # Both shipped GP102 images program INTR_MODE and INTR_ROUTING, so
        # neither can be a constant here. level_lines is kept as a constructor
        # override for tests; otherwise it tracks INTR_MODE.
        self.intr_mode = INTR_MODE_RESET if level_lines is None else level_lines
        self.intr_routing = 0
        self.intr = 0
        self.intr_en = 0
        self._pending_req = 0
        # Hardware counts clock cycles; a rehost steps instructions. This is
        # the conversion, and it is a choice rather than a measurement.
        self.cycles_per_step = cycles_per_step
        self.periodic_period = 0
        self.periodic_time = 0
        self.periodic_enable = 0
        self.watchdog_time = 0
        self.watchdog_enable = 0
        self.time = 0
        self.timer_ticks = 0
        self.read_handler[0:size] = self.hw_read
        self.write_handler[0:size] = self.hw_write
        log.info("%s: Falcon engine MMIO at 0x%08x (+0x%x), %d modelled "
                 "indirect registers", name, address, size, len(self.registers))

    def live_registers(self):
        """Offsets the backend should exchange with this model every step.

        Bounded on purpose: the I/O window is 256 KB and sweeping it per
        instruction would dominate the run. These are the registers the
        microcode actually polls or drives.
        """
        return (INTR, INTR_EN, ENGINE_STATUS, MAILBOX_REQ, MAILBOX_RESP,
                INTR_SET, INTR_CLEAR, INTR_EN_SET, INTR_EN_CLEAR,
                INTR_MODE, INTR_ROUTING,
                PERIODIC_PERIOD, PERIODIC_TIME, PERIODIC_ENABLE,
                WATCHDOG_TIME, WATCHDOG_ENABLE, TIME_LOW, TIME_HIGH)

    # -- timers ------------------------------------------------------------

    def tick(self, steps: int = 1) -> None:
        """Advance the timers, raising their lines when they expire.

        timer.rst: PERIODIC_TIME decreases by 1 each cycle while enabled; at 0
        it raises line 0 and reloads from PERIODIC_PERIOD. The watchdog is the
        same but one-shot, on line 1, and disables itself when it fires.
        """
        cycles = steps * self.cycles_per_step
        self.time += cycles

        if self.periodic_enable & 1:
            remaining = cycles
            fired = 0
            while remaining > 0:
                if self.periodic_time == 0:
                    # Nothing left to count: reload and take the tick. Guard
                    # against a zero period, which would otherwise spin here.
                    self.periodic_time = self.periodic_period
                    if self.periodic_time == 0:
                        self.raise_line(LINE_PERIODIC)
                        self.timer_ticks += 1
                        break
                step = min(remaining, self.periodic_time)
                self.periodic_time -= step
                remaining -= step
                if self.periodic_time == 0:
                    # Expiry is on *reaching* zero, per timer.rst.
                    self.raise_line(LINE_PERIODIC)
                    self.timer_ticks += 1
                    fired += 1
                    self.periodic_time = self.periodic_period
                    if self.periodic_period == 0 or fired > cycles:
                        break

        if self.watchdog_enable & 1:
            if self.watchdog_time <= cycles:
                self.watchdog_time = 0
                self.watchdog_enable = 0        # one-shot
                self.raise_line(LINE_WATCHDOG)
            else:
                self.watchdog_time -= cycles

    # -- interrupt lines ---------------------------------------------------

    def raise_line(self, num: int) -> None:
        """Make line `num` pending, as the engine's own hardware would."""
        self.intr |= 1 << num

    def route_of(self, line: int) -> int:
        """intr.rst: a line's two routing bits come from bit N of each half."""
        lo = (self.intr_routing >> line) & 1
        hi = (self.intr_routing >> (16 + line)) & 1
        return lo | (hi << 1)

    def pending_and_enabled(self, vector: Optional[int] = None) -> int:
        """Lines that are pending and unmasked.

        With `vector`, only the lines INTR_ROUTING sends to that falcon vector.
        Lines routed to the host (PMC) are excluded from both: they never reach
        the microcontroller, so delivering them into a handler would invent an
        interrupt the core would never see. Both shipped GP102 images program
        INTR_ROUTING, so this is not hypothetical.
        """
        live = self.intr & self.intr_en
        if vector is None:
            return live
        want = ROUTE_VECTOR0 if vector == 0 else ROUTE_VECTOR1
        out = 0
        for line in range(16):
            if (live >> line) & 1 and self.route_of(line) == want:
                out |= 1 << line
        return out

    # -- MMIO --------------------------------------------------------------

    def hw_read(self, offset: int, size: int, pc: int = 0xBAADBAAD,
                **kwargs: Any) -> int:
        if offset == INTR:
            return self.intr
        if offset == INTR_EN:
            return self.intr_en
        if offset == INTR_MODE:
            return self.intr_mode
        if offset == INTR_ROUTING:
            return self.intr_routing
        if offset == ENGINE_STATUS:
            # Always ready: there is no modelled latency, and a poll that can
            # never succeed is indistinguishable from a hung rehost.
            return self.ready_bits
        if offset == MAILBOX_REQ:
            return 0                      # busy bit clear: the request is done
        if offset == PERIODIC_PERIOD:
            return self.periodic_period
        if offset == PERIODIC_TIME:
            return self.periodic_time
        if offset == PERIODIC_ENABLE:
            return self.periodic_enable
        if offset == WATCHDOG_TIME:
            return self.watchdog_time
        if offset == WATCHDOG_ENABLE:
            return self.watchdog_enable
        if offset == TIME_LOW:
            return self.time & 0xFFFFFFFF
        if offset == TIME_HIGH:
            return (self.time >> 32) & 0xFFFFFFFF
        if offset == MAILBOX_RESP:
            val = self.registers.get(self._pending_req, 0)
            log.debug("%s: mailbox read 0x%06x -> 0x%08x",
                      self.name, self._pending_req, val)
            return val
        if offset in (INTR_SET, INTR_CLEAR, INTR_EN_SET, INTR_EN_CLEAR):
            # Write-only on hardware; reads are undefined. Return 0 rather than
            # inventing a value, and say so once.
            log.debug("%s: read of write-only register 0x%05x", self.name, offset)
            return 0
        return self.registers.get(offset, 0)

    def hw_write(self, offset: int, size: int, value: int,
                 pc: int = 0xBAADBAAD, **kwargs: Any) -> bool:
        if offset == INTR_SET:
            self.intr |= value & ~self.intr_mode
        elif offset == INTR_CLEAR:
            self.intr &= ~(value & ~self.intr_mode)
        elif offset == INTR_EN_SET:
            self.intr_en |= value
        elif offset == INTR_EN_CLEAR:
            self.intr_en &= ~value
        elif offset == INTR_MODE:
            self.intr_mode = value & 0xFFFF
        elif offset == INTR_ROUTING:
            self.intr_routing = value & 0xFFFFFFFF
        elif offset == PERIODIC_PERIOD:
            self.periodic_period = value
        elif offset == PERIODIC_TIME:
            self.periodic_time = value
        elif offset == PERIODIC_ENABLE:
            self.periodic_enable = value
        elif offset == WATCHDOG_TIME:
            self.watchdog_time = value
        elif offset == WATCHDOG_ENABLE:
            self.watchdog_enable = value
        elif offset in (TIME_LOW, TIME_HIGH):
            log.debug("%s: ignoring write to read-only %s",
                      self.name, "TIME_LOW" if offset == TIME_LOW else "TIME_HIGH")
        elif offset == MAILBOX_REQ:
            self._pending_req = value & REQ_ADDR_MASK
            log.debug("%s: mailbox request 0x%06x (raw 0x%08x)",
                      self.name, self._pending_req, value)
        elif offset in (INTR, INTR_EN):
            log.debug("%s: ignoring write to read-only status 0x%05x",
                      self.name, offset)
        else:
            self.registers[offset] = value
        return True
