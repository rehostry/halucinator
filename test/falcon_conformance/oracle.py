"""Expected results for irq_compute.fuc, derived from the documented ISA.

Written from envytools docs/hw/falcon/arith.rst and intr.rst, deliberately not
from the SLEIGH module, so that agreement means two independent readings of the
specification agree rather than that the model agrees with itself.
"""
M32 = 0xFFFFFFFF
N_IRQ = 8      # the firmware waits for exactly this many interrupts
TERMS = 64     # sum of squares over 1..TERMS

IRQ_COUNT, MIX, SUM_SQ, DONE, ALU, REGS = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14


def _s32(v):
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def _sext(src, bit):
    """arith.rst: keep bits 0..bit-1, replicate bit upward."""
    bit &= 0x1F
    low = src & ((1 << bit) - 1)
    return (low | (-(1 << bit) & M32)) & M32 if (src >> bit) & 1 else low


def _sum_sq():
    # mulu is a 16x16 -> 32 multiply: both operands truncate to 16 bits.
    return sum(((i & 0xFFFF) * (i & 0xFFFF)) for i in range(1, TERMS + 1)) & M32


def _mix():
    # The handler folds on every entry, in order, through a chain that
    # overflows 16 bits -- which is what pins mulu's truncation.
    mix = 0
    for k in range(1, N_IRQ + 1):
        mix = (((mix & 0xFFFF) * 0x1F) + k) & M32
    return mix


def _alu():
    r1 = (0x1234 & 0xFFFF) | 0xABCD0000          # mov + sethi
    r2 = (0x0F0F & 0xFFFF) | 0x0F0F0000
    r3 = r1 & r2
    r4 = r1 | r2
    r5 = r1 ^ r2
    r6 = (r3 + r4) & M32
    r6 ^= r5
    r6 = (r6 << 3) & M32
    r6 = (_s32(r6) >> 2) & M32                   # sar: arithmetic
    r6 = (r6 + (((r6 >> 16) | (r6 << 16)) & M32)) & M32   # hswap b32
    r6 = (r6 + ((r1 >> 0x10) & 1)) & M32         # xbit
    r6 = (r6 + _sext(r2, 0x7)) & M32
    r6 = (r6 + (~r2 & M32)) & M32                # not
    r6 = (r6 + (-r3 & M32)) & M32                # neg
    return r6


def expected():
    """The four-plus-two words the firmware must leave in DMEM."""
    sum_sq, mix, alu = _sum_sq(), _mix(), _alu()
    # r0..r3 = 0x11,0x22,0x33,0x44 must survive `clobber`, which saves them
    # with `mpush $r4` and then overwrites all four.
    regs = 0x11223344
    return {
        "irq_count": N_IRQ,
        "mix": mix,
        "sum_sq": sum_sq,
        "alu": alu,
        "regs": regs,
        "done": mix ^ sum_sq ^ alu ^ regs,
    }


OFFSETS = {"irq_count": IRQ_COUNT, "mix": MIX, "sum_sq": SUM_SQ,
           "done": DONE, "alu": ALU, "regs": REGS}

if __name__ == "__main__":
    for k, v in expected().items():
        print(f"{k:<10} {v:>12}  0x{v:08x}")


# ---------------------------------------------------------------------------
# fuc4 variant (irq_compute_fuc4.fuc): fewer terms, fewer interrupts, and none
# of the fuc5-only instructions.
# ---------------------------------------------------------------------------

FUC4_N_IRQ, FUC4_TERMS = 6, 32


def expected_fuc4():
    sum_sq = sum(((i & 0xFFFF) * (i & 0xFFFF))
                 for i in range(1, FUC4_TERMS + 1)) & M32
    mix = 0
    for k in range(1, FUC4_N_IRQ + 1):
        mix = (((mix & 0xFFFF) * 0x1F) + k) & M32
    return {"irq_count": FUC4_N_IRQ, "mix": mix, "sum_sq": sum_sq,
            "done": mix ^ sum_sq}


FUC4_OFFSETS = {"irq_count": IRQ_COUNT, "mix": MIX,
                "sum_sq": SUM_SQ, "done": DONE}
