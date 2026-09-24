"""Falcon engine MMIO model: interrupt aliasing, status, indirect mailbox."""
from __future__ import annotations

import pytest

from halucinator.peripheral_models.falcon_engine import (
    DEFAULT_READY_BITS, ENGINE_STATUS, FalconEngine, INTR, INTR_CLEAR,
    INTR_EN, INTR_EN_CLEAR, INTR_EN_SET, INTR_MODE, INTR_ROUTING,
    INTR_SET, LINE_PERIODIC,
    LINE_WATCHDOG, MAILBOX_REQ, MAILBOX_RESP, PERIODIC_ENABLE,
    PERIODIC_PERIOD, PERIODIC_TIME, TIME_LOW, WATCHDOG_ENABLE, WATCHDOG_TIME,
)


@pytest.fixture
def eng():
    return FalconEngine("eng", 0x0, 0x40000, registers={0x122234: 0x1E})


# -- interrupt controller aliasing ---------------------------------------

def test_set_makes_an_edge_line_pending(eng):
    eng.hw_write(INTR_SET, 4, 1 << 3)          # 3 = CHSW, edge
    assert eng.hw_read(INTR, 4) == 1 << 3


def test_clear_acknowledges_an_edge_line(eng):
    eng.hw_write(INTR_SET, 4, 1 << 3)
    eng.hw_write(INTR_CLEAR, 4, 1 << 3)
    assert eng.hw_read(INTR, 4) == 0


def test_set_is_ignored_for_a_level_line(eng):
    """intr.rst: "Attempts to SET or CLEAR level-triggered interrupts are
    ignored." Line 2 (FIFO) is level."""
    eng.hw_write(INTR_SET, 4, 1 << 2)
    assert eng.hw_read(INTR, 4) == 0


def test_hardware_can_still_raise_a_level_line(eng):
    """SET being ignored is about the register alias, not the wire."""
    eng.raise_line(2)
    assert eng.hw_read(INTR, 4) == 1 << 2


def test_enable_set_and_clear(eng):
    eng.hw_write(INTR_EN_SET, 4, (1 << 2) | (1 << 3))
    assert eng.hw_read(INTR_EN, 4) == (1 << 2) | (1 << 3)
    eng.hw_write(INTR_EN_CLEAR, 4, 1 << 2)
    assert eng.hw_read(INTR_EN, 4) == 1 << 3


def test_pending_and_enabled_is_the_intersection(eng):
    eng.hw_write(INTR_EN_SET, 4, 1 << 3)
    eng.raise_line(3)
    eng.raise_line(4)                           # pending but not enabled
    assert eng.pending_and_enabled() == 1 << 3


def test_status_registers_ignore_writes(eng):
    eng.raise_line(3)
    eng.hw_write(INTR, 4, 0)                    # read-only on hardware
    assert eng.hw_read(INTR, 4) == 1 << 3


# -- engine status --------------------------------------------------------

def test_status_reports_ready(eng):
    """The bits the GP102 microcode's poll loops wait on."""
    assert eng.hw_read(ENGINE_STATUS, 4) == DEFAULT_READY_BITS
    for bit in (0x40, 0x80, 0x4000):
        assert eng.hw_read(ENGINE_STATUS, 4) & bit


# -- indirect mailbox -----------------------------------------------------

def test_mailbox_serves_a_modelled_register(eng):
    eng.hw_write(MAILBOX_REQ, 4, 0x122234)
    assert eng.hw_read(MAILBOX_RESP, 4) == 0x1E


def test_mailbox_masks_the_control_bits_off_the_request(eng):
    """The helper ORs flags into the top of the request word."""
    eng.hw_write(MAILBOX_REQ, 4, 0x122234 | (0x2 << 26))
    assert eng.hw_read(MAILBOX_RESP, 4) == 0x1E


def test_unmodelled_indirect_register_reads_zero(eng):
    eng.hw_write(MAILBOX_REQ, 4, 0x409604)
    assert eng.hw_read(MAILBOX_RESP, 4) == 0


def test_request_register_reads_back_not_busy(eng):
    """The microcode polls bit 31 of the request register for "not busy"."""
    eng.hw_write(MAILBOX_REQ, 4, 0x122234)
    assert eng.hw_read(MAILBOX_REQ, 4) & (1 << 31) == 0


# -- timers ---------------------------------------------------------------

def _arm_periodic(eng, period):
    eng.hw_write(PERIODIC_PERIOD, 4, period)
    eng.hw_write(PERIODIC_TIME, 4, period)
    eng.hw_write(PERIODIC_ENABLE, 4, 1)


def test_periodic_timer_does_not_fire_early(eng):
    _arm_periodic(eng, 100)
    eng.tick(99)
    assert eng.hw_read(INTR, 4) == 0


def test_periodic_timer_fires_on_reaching_zero(eng):
    """timer.rst: "When PERIODIC_TIME reaches 0, an interrupt is generated on
    line 0 and the counter is reset to PERIODIC_PERIOD"."""
    _arm_periodic(eng, 100)
    eng.tick(100)
    assert eng.hw_read(INTR, 4) == 1 << LINE_PERIODIC
    assert eng.hw_read(PERIODIC_TIME, 4) == 100          # reloaded


def test_periodic_timer_keeps_firing(eng):
    _arm_periodic(eng, 100)
    eng.tick(1000)
    assert eng.timer_ticks == 10


def test_periodic_timer_is_inert_when_disabled(eng):
    _arm_periodic(eng, 100)
    eng.hw_write(PERIODIC_ENABLE, 4, 0)
    eng.tick(1000)
    assert eng.hw_read(INTR, 4) == 0
    assert eng.hw_read(PERIODIC_TIME, 4) == 100          # counter frozen


def test_a_zero_period_does_not_spin(eng):
    """A period of 0 must not loop forever inside one tick()."""
    eng.hw_write(PERIODIC_PERIOD, 4, 0)
    eng.hw_write(PERIODIC_TIME, 4, 0)
    eng.hw_write(PERIODIC_ENABLE, 4, 1)
    eng.tick(10)                                          # must return
    assert eng.hw_read(INTR, 4) == 1 << LINE_PERIODIC


def test_watchdog_is_one_shot_and_disables_itself(eng):
    eng.hw_write(WATCHDOG_TIME, 4, 50)
    eng.hw_write(WATCHDOG_ENABLE, 4, 1)
    eng.tick(50)
    assert eng.hw_read(INTR, 4) == 1 << LINE_WATCHDOG
    assert eng.hw_read(WATCHDOG_ENABLE, 4) == 0
    eng.hw_write(INTR_CLEAR, 4, 1 << LINE_WATCHDOG)
    eng.tick(1000)
    assert eng.hw_read(INTR, 4) == 0                      # never fires again


def test_time_advances_and_is_read_only(eng):
    eng.tick(1200)
    assert eng.hw_read(TIME_LOW, 4) == 1200
    eng.hw_write(TIME_LOW, 4, 0)
    assert eng.hw_read(TIME_LOW, 4) == 1200


def test_timer_line_only_counts_as_pending_when_enabled(eng):
    _arm_periodic(eng, 10)
    eng.tick(10)
    assert eng.pending_and_enabled() == 0                  # raised, not enabled
    eng.hw_write(INTR_EN_SET, 4, 1 << LINE_PERIODIC)
    assert eng.pending_and_enabled() == 1 << LINE_PERIODIC


# -- INTR_MODE and INTR_ROUTING --------------------------------------------
#
# Both shipped GP102 images program both registers, so neither can be the
# constant it used to be: the level-trigger mask was hardcoded to the reset
# value and routing was ignored entirely, which sends every line to vector 0.

def test_intr_mode_reads_its_reset_value(eng):
    """intr.rst: INTR_MODE is 0xfc04 out of reset."""
    assert eng.hw_read(INTR_MODE, 4) == 0xFC04


def test_intr_mode_is_writable_and_gates_set_and_clear(eng):
    """SET and CLEAR are ignored for level-triggered lines, and which lines
    are level-triggered is whatever the firmware last wrote."""
    eng.hw_write(INTR_MODE, 4, 0x0000)          # every line edge-triggered
    eng.hw_write(INTR_SET, 4, 0x0004)
    assert eng.hw_read(INTR, 4) == 0x0004       # line 2 now settable
    eng.hw_write(INTR_MODE, 4, 0x0004)          # line 2 back to level
    eng.hw_write(INTR_CLEAR, 4, 0x0004)
    assert eng.hw_read(INTR, 4) == 0x0004       # and no longer clearable


def test_routing_splits_lines_between_the_two_vectors(eng):
    eng.hw_write(INTR_EN_SET, 4, 0xFFFF)
    eng.raise_line(0)
    eng.raise_line(1)
    # line 1 -> vector 1 (route 2 = high bit only), line 0 stays on vector 0
    eng.hw_write(INTR_ROUTING, 4, 1 << (16 + 1))
    assert eng.pending_and_enabled(vector=0) == 1 << 0
    assert eng.pending_and_enabled(vector=1) == 1 << 1


def test_a_line_routed_to_the_host_reaches_neither_vector(eng):
    """Routes 1 and 3 go to PMC. The microcontroller never sees them, so
    delivering one into a handler would invent an interrupt."""
    eng.hw_write(INTR_EN_SET, 4, 0xFFFF)
    eng.raise_line(3)
    eng.hw_write(INTR_ROUTING, 4, 1 << 3)        # low bit only -> PMC HOST
    assert eng.pending_and_enabled() == 1 << 3   # pending, as hardware would
    assert eng.pending_and_enabled(vector=0) == 0
    assert eng.pending_and_enabled(vector=1) == 0
