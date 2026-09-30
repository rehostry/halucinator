# Falcon conformance firmware

`irq_compute.fuc` is a small fuc5 program written for this test suite, and
`irq_compute.bin` is it assembled. Unlike `test/falcon_fecs/`, nothing here is
vendor firmware, so the image is committed and the expected values are derived
from the source rather than recorded from a run.

Rebuild it with envytools' assembler:

```bash
envyas -m falcon -V fuc5 -i -o irq_compute.bin irq_compute.fuc
```

What it is for: an answer that can be computed independently, where the
interrupt is part of the answer. `test/pytest/backends/test_falcon_conformance.py`
runs it and checks four words in DMEM.

It exercises, in one run: `mulu`'s documented 16x16->32 truncation, `add`,
`cmpu` and the conditional branch that reads its carry, `ld`/`st` in the data
space, `push`/`pop`, `call`/`ret` with the return address on the *data* stack,
`mov $iv0 $rX` and `mov $flags $rX` (special-register moves, both directions),
`bset`/`bclr $flags ie0`, interrupt entry, and `iret`.

Two properties make it a test rather than a demonstration:

- **The count is structural, not timed.** The firmware waits for exactly 8
  interrupts before continuing, so `irq_count` is 8 however fast the model
  runs. If delivery stops -- an `iret` that fails to restore `ie0`, say -- it
  hangs instead of quietly producing a smaller number.

- **A wrong semantic changes a number.** `mix` is folded by the handler on
  every entry, in order, through a chain that overflows 16 bits.

The handler saves and restores `$flags` itself. That is not politeness:
interrupt entry saves only `ie0`/`ie1` (intr.rst), so without it the handler's
`add` overwrites the `c` flag the wait loop's `cmpu` just set, and the loop
exits early. The first version of this firmware had that bug, and the rehost
reproduced it faithfully -- as does GP102 FECS's own handler, which saves
`$flags` through `$r10` across its work.

## The multi-register stack family

`mpush`/`mpop`/`mpopret`/`mpopadd`/`mpopaddret` have no upstream documentation
at all -- envydis carries the encodings and the comment "Display these in a
better way, perhaps?" -- so their semantics are derived from shipped GP102
firmware, and this firmware is where the derivation is checked.

Two properties, both once read wrongly:

- **The operand names the highest register saved.** `mpush $r3` pushes r0-r3,
  four words. `clobber` overwrites exactly r0-r3, which puts the boundary
  register under test rather than only the middle of the range.

- **`mpopaddret`'s immediate is applied before the return address is popped.**
  A framed function allocates locals first and saves registers second, so the
  frame sits between the saved registers and the return address and has to be
  reclaimed first. `farroutine` fills its own frame with 0xdead: get the order
  wrong and the return leaves IMEM and the run faults, instead of producing a
  wrong number that could be mistaken for something else.

Reverting either property fails six of the tests, so they are load-bearing.
