"""Expected results for irq_compute.fuc, derived from the documented ISA.

Written from envytools docs/hw/falcon/arith.rst and intr.rst, deliberately not
from the SLEIGH module, so that agreement means two independent readings of the
specification agree rather than that the model agrees with itself.
"""
M32 = 0xFFFFFFFF
N_IRQ = 8      # the firmware waits for exactly this many interrupts
TERMS = 64     # sum of squares over 1..TERMS

IRQ_COUNT, MIX, SUM_SQ, DONE, ALU, REGS = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14
DMA, TLB, SIZED = 0x18, 0x1C, 0x20
CARRY, ROTC = 0x24, 0x28
MULTI, TRAPW, SIGNED = 0x2C, 0x30, 0x34
MISC, MISC2 = 0x38, 0x3C

# The code page the harness places in external port 0, and where.
CODE_PAGE_EXT_OFF = 0x2200
CODE_PAGE_PHYS = 0x1000          # IMEM byte address -> physical page 0x10
DMA_PATTERN = 0x5EED0042


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


def _tlb():
    """vm.rst, folded the way the firmware folds it.

    ptlb(phys)  = flags << 24 | virt << 8
    vtlb(virt)  = phys | flags << 24          (bit 31 if no match)
    then itlb(phys) clears it, so the second ptlb reads 0.
    """
    phys = CODE_PAGE_PHYS // 0x100                     # 0x10
    virt = (CODE_PAGE_EXT_OFF >> 8) & 0xFFFF           # 0x22
    flags = 0x1                                        # usable, not secret
    ptlb = ((flags << 24) | (virt << 8)) & M32
    vtlb = (phys | (flags << 24)) & M32
    return ((ptlb ^ vtlb) + 0) & M32                   # + ptlb-after-itlb (0)


def _sized():
    """arith.rst: a sized op computes in sz bits, flags come from bit sz-1.

        add b8  0xff + 0x02 -> 0x01, carry out of bit 7
        add b16 0xffff + 1  -> 0x0000, zero set, carry out of bit 15

    Packed by the firmware as r1 | c << 8 | z << 12 | r4 << 16.
    """
    r1 = (0xFF + 0x02) & 0xFF                  # 0x01
    c = 1 if (0xFF + 0x02) > 0xFF else 0       # 1
    r4 = (0xFFFF + 1) & 0xFFFF                 # 0x0000
    z = 1 if r4 == 0 else 0                    # 1
    return (r1 | (c << 8) | (z << 12) | (r4 << 16)) & M32


def _carry():
    """adc/sbb/shlc/div/mod, packed as the firmware packs them."""
    c = 1 if (0xFF + 1) > 0xFF else 0        # add b8 carried out
    adc = (0x10 + 0x20 + c) & M32            # 0x31
    borrow = 1 if 0x50 < 0x20 else 0         # cmpu sets c on borrow
    sbb = (0x50 - 0x20 - borrow) & M32       # 0x30
    shlc = ((0x1 << 1) | 1) & M32            # carry shifted in at bit 0 -> 3
    div, mod = 100 // 7, 100 % 7             # 14, 2
    return (adc | (sbb << 8) | (shlc << 16)
            | (div << 24) | (mod << 28)) & M32


def _rotc():
    """shrc shifts the carry into the top bit; setp then writes c directly."""
    shrc = ((0x2 >> 1) | (1 << 31)) & M32    # 0x80000001
    return (shrc ^ (1 << 1)) & M32           # setp c 1 -> xbit -> 1, at bit 1


def _multi():
    """mpop and mpopadd must restore exactly what mpush saved."""
    return (0xA1 | (0xB2 << 8) | (0xC3 << 16)) & M32


def _trapw():
    """The handler records `0x100 | reason`; `trap 0x1` has reason 1."""
    return (0x100 | 1) & M32


def _signed():
    """cmps answers signed less-than in c; cmpu answers unsigned."""
    cmps_c = 1 if -1 < 1 else 0                        # 1
    cmpu_c = 1 if (0xFFFFFFFF < 1) else 0              # 0
    btgl_c = 1                                         # cleared then toggled
    return cmps_c | (cmpu_c << 1) | (btgl_c << 2)


def _misc():
    """sub, cmp's zero flag, muls (signed 16x16 -> 32), and extr."""
    sub = (0x50 - 0x30) & 0xFF                 # 0x20
    z = 1                                      # cmp of equal operands
    muls = ((-2) * 3) & M32                    # 0xfffffffa
    muls_byte = muls & 0x30                    # masked with r2 (0x30)
    extr = (0xABCD >> 4) & 0xF                 # width 4 at position 4 -> 0xc
    return (sub | (z << 8) | (muls_byte << 16) | (extr << 24)) & M32


def _misc2():
    """iord reading back an iowr, extrs' sign bit, and a far call's callee."""
    iord = 0x1234
    r0 = 0x77                                  # preserved across mpush/mpopaddret
    extrs_sign = 1                             # 0xc sign-extended from bit 3
    return (iord | (r0 << 16) | (extrs_sign << 24)) & M32


def expected():
    """Every word the firmware must leave in DMEM."""
    sum_sq, mix, alu = _sum_sq(), _mix(), _alu()
    # r0..r3 = 0x11,0x22,0x33,0x44 must survive `clobber`, which saves them
    # with `mpush $r4` and then overwrites all four.
    regs = 0x11223344
    dma = DMA_PATTERN          # written to DMEM, pushed out by xdst, read back
    tlb = _tlb()
    sized = _sized()
    carry = _carry()
    rotc = _rotc()
    multi = _multi()
    trapw = _trapw()
    signed = _signed()
    misc = _misc()
    misc2 = _misc2()
    return {
        "irq_count": N_IRQ,
        "mix": mix,
        "sum_sq": sum_sq,
        "alu": alu,
        "regs": regs,
        "dma": dma,
        "tlb": tlb,
        "sized": sized,
        "carry": carry,
        "rotc": rotc,
        "multi": multi,
        "trapw": trapw,
        "signed": signed,
        "misc": misc,
        "misc2": misc2,
        "done": (mix ^ sum_sq ^ alu ^ regs ^ dma ^ tlb ^ sized ^ carry
                 ^ rotc ^ multi ^ trapw ^ signed ^ misc ^ misc2),
    }


OFFSETS = {"irq_count": IRQ_COUNT, "mix": MIX, "sum_sq": SUM_SQ,
           "done": DONE, "alu": ALU, "regs": REGS, "dma": DMA, "tlb": TLB,
           "sized": SIZED, "carry": CARRY, "rotc": ROTC,
           "multi": MULTI, "trapw": TRAPW,
           "signed": SIGNED, "misc": MISC, "misc2": MISC2}

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
