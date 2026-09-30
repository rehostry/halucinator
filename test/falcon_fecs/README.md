# Falcon FECS test config

Runs NVIDIA Pascal GP102 FECS microcode under the Ghidra backend.

**The firmware is not included.** Its redistribution terms were not verified,
so fetch it yourself from `linux-firmware` and drop it here:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://gitlab.com/kernel-firmware/linux-firmware.git
cd linux-firmware && git sparse-checkout set nvidia/gp102
cp nvidia/gp102/gr/fecs_inst.bin nvidia/gp102/gr/fecs_data.bin <this directory>/
```

Both halves are needed. Nouveau loads the code image into IMEM and the data
image into DMEM, and the data image holds tables the microcode's own init
reads -- without it the firmware boots but answers nothing.

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


## What it does once it is up

The config attaches `FalconCtxctl`, which models the CTXCTL register ranges on
top of the base Falcon engine. With it, the microcode completes the bring-up
nouveau's `gf100_gr_init_ctxctl_ext` performs and then answers the driver's
control methods -- the sizes it needs to allocate a graphics context, the
instance binding, and a golden-context save that moves real register state
through the MMCTX queue.

Run the whole verification through `test/falcon_fecs/verify.sh`, which is the
documented path and checks its own prerequisites rather than assuming them --
including the extension jar, whose absence otherwise shows up as Ghidra failing
to construct an emulator. `--fast` leaves out the two slow files (ctxctl boots
the microcode once per test, corpus sweeps 28 images) for about a minute of
tests instead of an hour.

`test/pytest/backends/test_falcon_ctxctl.py` is the executable version of that
claim, including the controls: an unrecognised method must be refused with the
firmware's own error code, a request the firmware declines must move nothing,
and the reported context size must change with the strand state by exactly the
amount the firmware's arithmetic specifies.

## What is not modelled

Stated here rather than left to be discovered, because several of these are
things a run will quietly work around:

- **Crypt.** `crypt.rst` is "todo: write me" throughout and rnndb gives op names
  only, so there is nothing to implement from. Every crypt op is logged and
  recorded; `strict_crypt=True` halts on one. Neither GP102 image executes any.
- **Strand streaming.** The strand unit's command sequence, its nine
  `STRAND_WORDS` and the save/load bases are modelled, and the firmware's size
  arithmetic runs over them -- but `STRAND_CMD` SAVE/LOAD moves no data through
  `STRAND_DATA`. Nothing exercised here needs it: a golden save goes through
  MMCTX and a channel switch through the MMIO bus, both measured.
- **`STRAND_WORDS` and the GPC/TPC counts are inputs, not facts.** On hardware
  the strand unit and the MC UNITS registers report how much state exists;
  nothing in a rehost can derive that. They default to zero, which is the honest
  "no context state is modelled", and the sizes FECS reports follow from them by
  the firmware's own arithmetic. A caller with measured values passes them in.
- **GPCCS is not co-simulated.** It boots under the same model (see
  `test_the_same_model_carries_gpccs`) but not *alongside* FECS, so the
  cross-controller barriers in SIGNAL are answered as always-satisfied. A rehost
  that runs both must drive them from real barrier state instead.
- **Register values behind the MMIO bus.** Anything not in the `bar0` dict reads
  as zero. A golden save therefore captures mostly zeros; what is checked is
  that the values it captures are the register file's, and that the ranges it
  names agree with nouveau's own context tables.
