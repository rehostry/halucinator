# Copyright 2026 Christopher Wright
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cortex-M delivery must honour PRIMASK/FAULTMASK.

The backend used to apply a pended external IRQ whenever the drain loop next
ran, without consulting the masks. Because the queue is drained at instruction
*chunk* boundaries, an IRQ queued while interrupts were enabled could land
inside a critical section that had since disabled them. Silicon holds it
pending instead.

These tests pin the decision function and the queue policy without needing a
real unicorn instance -- they exercise ``_cortexm_irq_deferred`` and the
skip-to-first-deliverable rule directly.
"""
import pytest

from halucinator.backends.unicorn_backend import UnicornBackend


class _FakeUc:
    """Minimal stand-in exposing just the two mask registers."""

    def __init__(self, primask=0, faultmask=0, ipsr=0):
        self.primask = primask
        self.faultmask = faultmask
        self.ipsr = ipsr

    def reg_read(self, reg):
        from unicorn import arm_const
        if reg == arm_const.UC_ARM_REG_PRIMASK:
            return self.primask
        if reg == arm_const.UC_ARM_REG_FAULTMASK:
            return self.faultmask
        if reg == arm_const.UC_ARM_REG_IPSR:
            return self.ipsr
        raise AssertionError("unexpected register %r" % reg)


def _backend(arch_name, primask=0, faultmask=0, ipsr=0, guard=True):
    be = UnicornBackend.__new__(UnicornBackend)
    be.arch_name = arch_name
    be._uc = _FakeUc(primask, faultmask, ipsr)
    be._cortexm_primask_guard = guard
    return be


def test_guard_is_off_by_default_so_no_device_changes_behaviour():
    """The guard must be opt-in: enabling it fleet-wide stalls devices tuned
    against the unguarded timing (measured on device-tinysa: 285 injections ->
    55, firmware never reaches its shell)."""
    be = _backend("cortex-m3", primask=1, guard=False)
    assert be._cortexm_irq_deferred(30) is False


# -- the decision itself --------------------------------------------------

def test_external_irq_held_while_primask_set():
    be = _backend("cortex-m3", primask=1)
    assert be._cortexm_irq_deferred(30) is True


def test_external_irq_held_while_faultmask_set():
    be = _backend("cortex-m3", faultmask=1)
    assert be._cortexm_irq_deferred(30) is True


def test_external_irq_delivered_when_unmasked():
    be = _backend("cortex-m3")
    assert be._cortexm_irq_deferred(30) is False


def test_nmi_is_never_masked():
    """NMI (-14) and every other internal exception keep their behaviour.

    This is the property the ChibiOS ARMv6-M context switch depends on: it
    disables interrupts, sets NMIPENDSET and spins waiting for the NMI.
    """
    be = _backend("cortex-m3", primask=1, faultmask=1)
    for internal in (-14, -5, -2, -1):
        assert be._cortexm_irq_deferred(internal) is False


@pytest.mark.parametrize("arch", ["arm", "arm64", "mips", "x86", "powerpc"])
def test_non_m_profile_untouched(arch):
    """A-profile handles its own masking; nothing else should change."""
    be = _backend(arch, primask=1, faultmask=1)
    assert be._cortexm_irq_deferred(30) is False


def test_absent_register_degrades_to_old_behaviour():
    class _NoRegs:
        def reg_read(self, reg):
            raise OSError("register not supported by this unicorn build")

    be = _backend("cortex-m3")
    be._uc = _NoRegs()
    assert be._cortexm_irq_deferred(30) is False


# -- already-in-a-handler (the second half of the validated rule) ----------

def test_irq_held_while_already_in_a_handler():
    """IPSR != 0: do not stack a second exception on a handler that has not
    executed an instruction (playbook trap 98)."""
    be = _backend("cortex-m3", ipsr=16)
    assert be._cortexm_irq_deferred(30) is True


# -- the queue policy -----------------------------------------------------

def _drain(be, queue):
    """Reproduce the drain loop: drop non-deliverable entries, apply the rest."""
    applied, dropped = [], []
    pending = list(queue)
    while pending:
        head = pending[0]
        if be._cortexm_irq_deferred(head):
            dropped.append(pending.pop(0))
            continue
        applied.append(pending.pop(0))
    return applied, dropped


def test_masked_external_irq_is_dropped_not_delivered():
    """Dropping (not holding) is what device-nanovna-h validated: the sources
    re-assert, whereas holding delivers it later at a PC silicon never would."""
    be = _backend("cortex-m3", primask=1)
    applied, dropped = _drain(be, [30])
    assert applied == []
    assert dropped == [30]


def test_nmi_is_delivered_even_behind_a_masked_irq():
    """NMI is never maskable, and must not be lost when a masked external IRQ
    is ahead of it in the queue."""
    be = _backend("cortex-m3", primask=1)
    applied, dropped = _drain(be, [30, -14])
    assert applied == [-14]
    assert dropped == [30]


def test_unmasked_queue_drains_in_order():
    be = _backend("cortex-m3")
    applied, dropped = _drain(be, [30, -14, 7])
    assert applied == [30, -14, 7]
    assert dropped == []


def test_guard_off_delivers_everything():
    be = _backend("cortex-m3", primask=1, ipsr=16, guard=False)
    applied, dropped = _drain(be, [30, -14])
    assert applied == [30, -14]
    assert dropped == []
