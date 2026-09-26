"""Falcon interrupt controller + exception delivery.

Uses a fake backend rather than the real microcode: the NVIDIA firmware's
redistribution terms were not verified, so it is not vendored here. These
tests pin the architectural behaviour from envytools docs/hw/falcon/intr.rst,
which is what would silently regress.
"""
from __future__ import annotations

import pytest

from halucinator.backends.irq import IrqConfigError, build_irq_controller
from halucinator.backends.irq.falcon import (
    INTR_EN, INTR_STATUS, FalconIrqController,
)
from halucinator.backends.irq.delivery import (
    DeliveryPlan, build_exception_deliverer,
)


class FakeBackend:
    """Minimal HalBackend stand-in with space-tagged memory."""

    def __init__(self, **regs):
        self.regs = {"sp": 0x4000, "pc": 0x1000,
                     "Ie0": 0, "Ie1": 0, "Is0": 0, "Is1": 0,
                     "iv0": 0, "iv1": 0}
        self.regs.update(regs)
        self.mem = {}

    def read_register(self, name):
        return self.regs[name]

    def write_register(self, name, value):
        self.regs[name] = value

    def write_registers(self, mapping):
        self.regs.update(mapping)

    def read_memory(self, addr, size, num_words=1, raw=False, space=None):
        return self.mem.get((space, addr), 0)

    def write_memory(self, addr, size, value, num_words=1, raw=False,
                     space=None):
        self.mem[(space, addr)] = value
        return True


def test_controller_resolves_by_arch():
    assert isinstance(build_irq_controller("falcon"), FalconIrqController)


@pytest.mark.parametrize("num", [0, 2, 4, 15])
def test_trigger_sets_the_pending_bit(num):
    be = FakeBackend()
    build_irq_controller("falcon").trigger(be, num)
    assert be.read_memory(INTR_STATUS, 4, space="io") == 1 << num


def test_trigger_preserves_other_pending_lines():
    be = FakeBackend()
    ctl = build_irq_controller("falcon")
    ctl.trigger(be, 2)
    ctl.trigger(be, 5)
    assert be.read_memory(INTR_STATUS, 4, space="io") == (1 << 2) | (1 << 5)


@pytest.mark.parametrize("num", [-1, 16, 99])
def test_out_of_range_line_is_rejected(num):
    with pytest.raises(IrqConfigError):
        build_irq_controller("falcon").trigger(FakeBackend(), num)


def test_enabled_reads_intr_en():
    be = FakeBackend()
    be.write_memory(INTR_EN, 4, 1 << 3, space="io")
    assert FalconIrqController.enabled(be, 3)
    assert not FalconIrqController.enabled(be, 4)


# -- delivery -------------------------------------------------------------

def test_delivery_is_refused_when_the_enable_is_clear():
    be = FakeBackend(Ie0=0, iv0=0x1234)
    assert build_exception_deliverer("falcon").deliver(be, 2, DeliveryPlan()) is False
    assert be.read_register("pc") == 0x1000      # untouched


def test_delivery_refuses_to_jump_to_zero():
    be = FakeBackend(Ie0=1, iv0=0)               # no vector installed anywhere
    assert build_exception_deliverer("falcon").deliver(be, 2, DeliveryPlan()) is False


def test_delivery_falls_back_to_a_configured_isr():
    be = FakeBackend(Ie0=1, iv0=0)
    plan = DeliveryPlan(isr_addr=0x0542)
    assert build_exception_deliverer("falcon").deliver(be, 2, plan) is True
    assert be.read_register("pc") == 0x0542


def test_delivery_matches_the_documented_entry_sequence():
    """intr.rst: sp -= 4; ST(32,sp,pc); isX = ieX; ieX = 0; pc = $ivX."""
    be = FakeBackend(Ie0=1, Ie1=1, iv0=0x0542, pc=0x06C8, sp=0x3FF0)
    plan = DeliveryPlan(stack_space="dmem")
    assert build_exception_deliverer("falcon").deliver(be, 2, plan) is True

    assert be.read_register("pc") == 0x0542               # jumped to the vector
    assert be.read_register("sp") == 0x3FEC               # sp -= 4
    # the pushed return address goes to the DATA space, not the code space --
    # Falcon is Harvard and a default-space write would hit the instruction
    # stream instead.
    assert be.read_memory(0x3FEC, 4, space="dmem") == 0x06C8
    assert (be.read_register("Is0"), be.read_register("Is1")) == (1, 1)
    assert (be.read_register("Ie0"), be.read_register("Ie1")) == (0, 0)


def test_vector_one_is_selectable():
    be = FakeBackend(Ie0=1, Ie1=1, iv0=0x0100, iv1=0x0200)
    plan = DeliveryPlan(falcon_vector=1, stack_space="dmem")
    assert build_exception_deliverer("falcon").deliver(be, 9, plan) is True
    assert be.read_register("pc") == 0x0200


def test_vector_one_respects_its_own_enable():
    be = FakeBackend(Ie0=1, Ie1=0, iv0=0x0100, iv1=0x0200)
    plan = DeliveryPlan(falcon_vector=1)
    assert build_exception_deliverer("falcon").deliver(be, 9, plan) is False


# -- autonomous, peripheral-driven delivery -------------------------------
#
# The loop these pin is: a peripheral raises its own line as emulated time
# passes, and the backend takes the interrupt without a harness poking one in.
# Only the two backend methods are exercised, bound onto FakeBackend, so the
# test does not need a Ghidra installation to run.

from halucinator.peripheral_models.falcon_engine import (
    FalconEngine, INTR_CLEAR, INTR_EN_SET, PERIODIC_ENABLE, PERIODIC_PERIOD,
    PERIODIC_TIME,
)


class IrqBackend(FakeBackend):
    """FakeBackend plus the real tick/deliver methods under test."""

    arch = "falcon"

    def __init__(self, engine, **kw):
        super().__init__(**kw)
        self._mmio = True
        self._mmio_live = [(0x0, engine, {}, {})]
        self.auto_deliver_peripheral_irqs = True
        self.peripheral_irq_plan = DeliveryPlan(falcon_vector=0,
                                                stack_space="dmem")

    def _space_of(self, per):
        return "io"

    from halucinator.backends.ghidra_backend import GhidraBackend
    _tick_peripherals = GhidraBackend._tick_peripherals
    _deliver_peripheral_irq = GhidraBackend._deliver_peripheral_irq
    del GhidraBackend


def _armed_engine(period=100):
    eng = FalconEngine("eng", 0x0, 0x20000)
    eng.hw_write(PERIODIC_PERIOD, 4, period)
    eng.hw_write(PERIODIC_TIME, 4, period)
    eng.hw_write(PERIODIC_ENABLE, 4, 1)
    eng.hw_write(INTR_EN_SET, 4, 1 << 0)
    return eng


def _run(backend, steps, ack=True):
    """Run `steps` instructions, standing in for the handler on each entry.

    A real Falcon handler acknowledges its line (INTR_CLEAR) and returns
    (`ret`, which restores Ie0). `ack=False` models a handler that only
    returns -- see the re-entry test below for why that is worth separating.
    """
    eng = backend._mmio_live[0][1]
    taken = 0
    for _ in range(steps):
        backend._tick_peripherals()
        if backend._deliver_peripheral_irq():
            taken += 1
            if ack:
                eng.hw_write(INTR_CLEAR, 4, 1 << 0)
            backend.regs["Ie0"] = 1      # stand in for the handler's `ret`
    return taken


def test_the_timer_drives_delivery_with_no_harness_trigger():
    """One entry per expiry, with nothing but emulated time driving it."""
    eng = _armed_engine(100)
    be = IrqBackend(eng, Ie0=1, iv0=0x500)
    assert _run(be, 1000) == 10
    assert eng.timer_ticks == 10                         # entries == expiries
    assert be.regs["pc"] == 0x500


def test_a_handler_that_never_acks_keeps_re_entering():
    """The backend does not ack on the firmware's behalf, so a handler that
    forgets to is visible as repeated entry rather than hidden."""
    be = IrqBackend(_armed_engine(100), Ie0=1, iv0=0x500)
    assert _run(be, 1000, ack=False) > 10


def test_nothing_is_delivered_before_the_timer_expires():
    be = IrqBackend(_armed_engine(100), Ie0=1, iv0=0x500)
    assert _run(be, 99) == 0
    assert be.regs["pc"] == 0x1000                       # never entered


def test_delivery_is_opt_in():
    be = IrqBackend(_armed_engine(10), Ie0=1, iv0=0x500)
    be.auto_deliver_peripheral_irqs = False
    assert _run(be, 1000) == 0


def test_a_disabled_line_is_raised_but_not_taken():
    """The control for the test above: same timer, enable bit clear."""
    eng = _armed_engine(10)
    eng.hw_write(0x00500, 4, 1 << 0)                     # INTR_EN_CLEAR
    be = IrqBackend(eng, Ie0=1, iv0=0x500)
    assert _run(be, 1000) == 0
    assert eng.intr & 1                                  # raised all the same


def test_a_still_pending_line_cannot_re_enter_until_the_handler_returns():
    """Entry clears Ie0; without the `ret` that restores it, no second entry."""
    eng = _armed_engine(10)
    be = IrqBackend(eng, Ie0=1, iv0=0x500)
    taken = 0
    for _ in range(1000):
        be._tick_peripherals()
        if be._deliver_peripheral_irq():
            taken += 1                                   # no Ie0 restore here
    assert taken == 1
