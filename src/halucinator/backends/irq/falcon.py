"""NVIDIA Falcon interrupt controller.

Falcon has 16 interrupt lines, exposed through set/clear/status register
triples in the engine's I/O space (envytools docs/hw/falcon/intr.rst,
"Interrupt status and enable registers")::

    I[0x00000]  INTR_SET        write 1s here to make lines pending
    I[0x00100]  INTR_CLEAR      write 1s here to acknowledge
    I[0x00200]  INTR            status, read-only
    I[0x00400]  INTR_EN_SET
    I[0x00500]  INTR_EN_CLEAR
    I[0x00600]  INTR_EN         status, read-only

The documented lines are::

    0 PERIODIC   1 WATCHDOG   2 FIFO   3 CHSW   4 EXIT
    5 (unknown)  6-7 SCRATCH  8-15 engine-specific

A note on what this models and what it does not. On hardware the SET and CLEAR
registers are write-only aliases whose writes are folded into the status
register, and SET/CLEAR are *ignored* for level-triggered lines. The rehost
maps the I/O space as plain memory, so there is no aliasing logic behind those
addresses: this controller writes the **status** register directly. That is
the honest model for making a line pending from outside -- it is what the
hardware's SET alias would have achieved -- but firmware that reads back
INTR_SET or relies on level-triggered writes being dropped will not see
hardware behaviour. Modelling that belongs in a peripheral handler on the
range, not in the controller.

Making a line pending does not by itself run a handler: the p-code emulator
takes no hardware exception, so entry is synthesised by
``FalconExceptionDeliverer`` and gated on the ``$flags.ieX`` enable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import IrqConfigError, IrqController

if TYPE_CHECKING:  # pragma: no cover
    from halucinator.backends.hal_backend import HalBackend

# Offsets within the engine's I/O space.
INTR_STATUS = 0x00200
INTR_EN = 0x00600

NUM_LINES = 16

LINE_NAMES = {
    0: "PERIODIC", 1: "WATCHDOG", 2: "FIFO", 3: "CHSW", 4: "EXIT",
    6: "SCRATCH0", 7: "SCRATCH1",
}


def _rw_kwargs(backend: "HalBackend") -> dict:
    """Address the I/O space when the backend knows about spaces.

    Falcon is Harvard: code, data and engine MMIO are separate Sleigh spaces,
    so an interrupt register written through the default accessor would land
    in the instruction stream. Backends without space support get the default
    space and the caller is expected to have mapped I/O there.
    """
    try:
        import inspect
        if "space" in inspect.signature(backend.write_memory).parameters:
            return {"space": "io"}
    except (TypeError, ValueError):  # pragma: no cover - builtin/bound oddities
        pass
    return {}


class FalconIrqController(IrqController):
    """Set the pending bit for a Falcon interrupt line."""

    name = "falcon"

    def trigger(self, backend: "HalBackend", num: int) -> None:
        if not 0 <= num < NUM_LINES:
            raise IrqConfigError(
                f"Falcon has {NUM_LINES} interrupt lines (0..{NUM_LINES - 1}); "
                f"got {num}. Named lines: "
                + ", ".join(f"{n}={name}" for n, name in sorted(LINE_NAMES.items()))
            )
        kw = _rw_kwargs(backend)
        status = backend.read_memory(INTR_STATUS, 4, 1, **kw)
        backend.write_memory(INTR_STATUS, 4, status | (1 << num), **kw)

    @staticmethod
    def enabled(backend: "HalBackend", num: int) -> bool:
        """Whether INTR_EN allows line `num` to reach a handler."""
        kw = _rw_kwargs(backend)
        return bool(backend.read_memory(INTR_EN, 4, 1, **kw) & (1 << num))
