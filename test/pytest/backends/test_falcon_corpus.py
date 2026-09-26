"""Run every Falcon microcode image in a linux-firmware checkout.

The conformance firmware proves semantics against an oracle; this proves the
model survives real code it has never seen. A wedge here names an instruction
that shipping firmware actually reaches, which no encoding sweep can tell you:
the sweep proves a byte pattern decodes, this proves a *sequence* executes.

The images are not vendored -- their redistribution terms were not verified --
so point FALCON_FIRMWARE_DIR at the `nvidia/` directory of a linux-firmware
checkout:

    git clone --depth 1 --filter=blob:none --sparse \\
        https://gitlab.com/kernel-firmware/linux-firmware.git
    cd linux-firmware && git sparse-checkout set nvidia
    FALCON_FIRMWARE_DIR=$PWD/nvidia python -m pytest test/pytest/backends/test_falcon_corpus.py

Scope, measured rather than assumed: the `gr` microcode of every chip from
Maxwell to Turing runs clean, and five of six SEC2 images do. The exceptions --
Ampere SEC2, the Tegra PMU images and the NVDEC scrubbers -- stop at bytes
envydis itself refuses to decode, or fall through into data because the
scrubber needs NVDEC-specific register values this generic engine does not
provide. Neither is an ISA gap, and both were checked against envydis rather
than waved away, so `gr` is what this test asserts.
"""
from __future__ import annotations

import os
import pathlib

import pytest

pytest.importorskip("pyghidra")

STEPS = int(os.environ.get("FALCON_CORPUS_STEPS", "20000"))


def _corpus():
    root = os.environ.get("FALCON_FIRMWARE_DIR")
    if not root:
        return []
    return sorted(pathlib.Path(root).glob("*/gr/*_inst.bin"))


CORPUS = _corpus()


def _run_image(fw: pathlib.Path):
    from halucinator.backends.ghidra_backend import GhidraBackend
    from halucinator.backends.hal_backend import MemoryRegion
    from halucinator.peripheral_models.falcon_engine import FalconEngine

    be = GhidraBackend(arch="falcon", cpu_model="fuc5")
    engine = FalconEngine("engine", 0x0, 0x40000)
    size = max(0x8000, (fw.stat().st_size + 0xFFF) & ~0xFFF)
    be.add_memory_region(MemoryRegion("imem", 0x0, size,
                                      permissions="rwx", file=str(fw)))
    be.add_memory_region(MemoryRegion("dmem", 0x0, 0x4000,
                                      permissions="rw", space="dmem"))
    be.add_memory_region(MemoryRegion("io", 0x0, 0x40000, permissions="rw",
                                      space="io", emulate=engine))
    be.init()
    be.write_register("sp", 0x4000)
    be.write_register("pc", 0x0)
    for _ in range(STEPS):
        be.step()
        if be._halted or be._step_fault_pc is not None:
            break
    return be


@pytest.mark.skipif(not CORPUS,
                    reason="set FALCON_FIRMWARE_DIR to a linux-firmware nvidia/ tree")
@pytest.mark.parametrize("fw", CORPUS, ids=lambda p: "/".join(p.parts[-3:]))
def test_gr_microcode_runs_without_wedging(fw):
    """No instruction this image reaches may fail to execute."""
    if not os.environ.get("GHIDRA_INSTALL_DIR"):
        pytest.skip("GHIDRA_INSTALL_DIR not set")
    be = _run_image(fw)
    assert be._step_fault_pc is None, (
        f"{fw.name} could not execute the instruction at "
        f"0x{be._step_fault_pc:x}")


@pytest.mark.skipif(not CORPUS, reason="no firmware corpus")
def test_the_corpus_is_not_trivially_small():
    """A corpus of one would pass while proving nothing.

    The `gr` tree carries fecs and gpccs for every supported chip, so a real
    checkout has dozens. Fewer than ten means the glob matched something
    unexpected and the sweep above is not covering what it claims.
    """
    assert len(CORPUS) >= 10, f"only {len(CORPUS)} images found under FALCON_FIRMWARE_DIR"
