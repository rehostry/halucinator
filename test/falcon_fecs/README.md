# Falcon FECS test config

Runs NVIDIA Pascal GP102 FECS microcode under the Ghidra backend.

**The firmware is not included.** Its redistribution terms were not verified,
so fetch it yourself from `linux-firmware` and drop it here:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://gitlab.com/kernel-firmware/linux-firmware.git
cd linux-firmware && git sparse-checkout set nvidia/gp102
cp nvidia/gp102/gr/fecs_inst.bin <this directory>/
```

Then:

```bash
GHIDRA_INSTALL_DIR=/path/to/ghidra \
  halucinator --emulator ghidra -c test/falcon_fecs/falcon_fecs.yaml -n falcon
```

Needs the `ghidra-falcon` processor module installed under
`$GHIDRA_INSTALL_DIR/Ghidra/Processors/Falcon`.

Note the three memory regions map to three *different* Sleigh address spaces —
Falcon is Harvard, with code, data and engine MMIO kept apart. A region
without a `space:` lands in the default (code) space, where the firmware's
loads and stores will not see it.
