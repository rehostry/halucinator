"""Falcon's DMA controller and code TLB.

Both are modelled from envytools ``docs/hw/falcon/xfer.rst`` and ``vm.rst``,
which specify them completely -- unlike the crypt coprocessor, whose document
is "todo: write me" throughout and which is therefore left unimplemented and
loud rather than guessed at.

**External memory lives here, not in the emulator.** Falcon's SLEIGH declares
three address spaces (code, data, I/O) and no fourth, so there is nowhere in
Ghidra to put the memory a transfer reads from. Ports are backed by Python
bytearrays and the engine moves bytes between them and emulator memory.

**Transfers complete synchronously.** Hardware queues them and runs them
asynchronously, which is why the ISA has ``xdwait``/``xcwait``. Completing
inside the instruction means the queue is always drained by the time a wait
executes, so the waits are correctly no-ops rather than conveniently ignored.
What this does not model is a firmware that races its own transfer -- reading a
buffer before waiting. Such code is broken on hardware and works here.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

CODE_PAGE = 0x100          # a code xfer always moves one 0x100-byte page

# TLB entry flags (xfer.rst: a finished page is "usable", or "secret" if the
# secret flag was set; vm.rst: ITLB cannot clear a page whose flags have 0x4).
FLAG_USABLE = 0x1
FLAG_BUSY = 0x2
FLAG_SECRET = 0x4


class FalconXferEngine:
    """The DMA controller, its external memory ports, and the code TLB."""

    def __init__(self, code_pages: int = 0x80, vm_page_bits: int = 16) -> None:
        self.ports: Dict[int, bytearray] = {}
        self.code_pages = code_pages
        self.vm_page_mask = (1 << vm_page_bits) - 1
        # TLB[i] = (virt_page_index, flags) for physical code page i
        self.tlb: List[List[int]] = [[0, 0] for _ in range(code_pages)]
        self.loads = self.stores = self.code_loads = 0

    # -- external memory ---------------------------------------------------

    def port(self, num: int) -> bytearray:
        return self.ports.setdefault(num & 7, bytearray())

    def load_port(self, num: int, data: bytes, offset: int = 0) -> None:
        """Place `data` at `offset` in an external port, growing it as needed."""
        buf = self.port(num)
        end = offset + len(data)
        if len(buf) < end:
            buf.extend(b"\x00" * (end - len(buf)))
        buf[offset:end] = data

    def _read_ext(self, num: int, addr: int, length: int) -> bytes:
        buf = self.port(num)
        if addr + length > len(buf):
            # Reading past the end of a port is a firmware or configuration
            # error. Zero-fill so the result is defined, and say so once.
            log.warning("FalconXfer: port %d read 0x%x+0x%x past end (0x%x); "
                        "zero-filling", num, addr, length, len(buf))
            chunk = bytes(buf[addr:addr + length])
            return chunk + b"\x00" * (length - len(chunk))
        return bytes(buf[addr:addr + length])

    def _write_ext(self, num: int, addr: int, data: bytes) -> None:
        buf = self.port(num)
        end = addr + len(data)
        if len(buf) < end:
            buf.extend(b"\x00" * (end - len(buf)))
        buf[addr:end] = data

    # -- transfers ---------------------------------------------------------

    @staticmethod
    def _data_len(size: int) -> int:
        """xfer.rst: a data xfer copies (4 << size) bytes, size 0-6."""
        return 4 << (size & 7)

    def data_load(self, backend, port: int, ext_base: int, ext_off: int,
                  local: int, size: int) -> int:
        """External -> DMEM."""
        length = self._data_len(size)
        addr = ((ext_base << 8) + ext_off) & 0xFFFFFFFF
        data = self._read_ext(port, addr, length)
        backend.write_memory(local & 0xFFFF, 1, data, len(data), raw=True,
                             space="dmem")
        self.loads += 1
        log.info("FalconXfer: data load port%d 0x%x -> D[0x%x] (%d bytes)",
                 port, addr, local & 0xFFFF, length)
        return length

    def data_store(self, backend, port: int, ext_base: int, ext_off: int,
                   local: int, size: int) -> int:
        """DMEM -> external."""
        length = self._data_len(size)
        addr = ((ext_base << 8) + ext_off) & 0xFFFFFFFF
        data = backend.read_memory(local & 0xFFFF, 1, length, raw=True,
                                   space="dmem")
        self._write_ext(port, addr, bytes(data))
        self.stores += 1
        log.info("FalconXfer: data store D[0x%x] -> port%d 0x%x (%d bytes)",
                 local & 0xFFFF, port, addr, length)
        return length

    def code_load(self, backend, port: int, ext_base: int, ext_off: int,
                  local: int, secret: bool = False) -> int:
        """External -> IMEM, one 0x100-byte page, and map it in the TLB.

        xfer.rst: the page is mapped to virtual address `ext_offset`, and on
        completion its flags become "usable", or "secret" when the secret flag
        was set. Since the copy finishes inside this call the page is never
        observably busy.
        """
        addr = ((ext_base << 8) + ext_off) & 0xFFFFFFFF
        data = self._read_ext(port, addr, CODE_PAGE)
        phys = (local & 0xFFFF) // CODE_PAGE
        backend.write_memory((local & 0xFFFF) & ~(CODE_PAGE - 1), 1, data,
                             len(data), raw=True)
        if 0 <= phys < self.code_pages:
            self.tlb[phys] = [(ext_off >> 8) & self.vm_page_mask,
                              FLAG_SECRET if secret else FLAG_USABLE]
        self.code_loads += 1
        log.info("FalconXfer: code load port%d 0x%x -> IMEM page %d "
                 "(virt 0x%x)%s", port, addr, phys, (ext_off >> 8) & 0xFFFF,
                 " secret" if secret else "")
        return CODE_PAGE

    # -- TLB ---------------------------------------------------------------

    def ptlb(self, phys: int) -> int:
        """vm.rst: `TLB[phys].flags << 24 | TLB[phys].virt << 8`."""
        phys &= 0xFFFFFF
        if not 0 <= phys < self.code_pages:
            return 0
        virt, flags = self.tlb[phys]
        return ((flags & 0x7) << 24 | (virt & 0xFFFF) << 8) & 0xFFFFFFFF

    def vtlb(self, virtaddr: int) -> int:
        """vm.rst: search every entry; OR the flags, report hit count.

        bit 31 set when nothing matched, bit 30 when more than one did.
        """
        want = (virtaddr >> 8) & self.vm_page_mask
        phys = flags = matches = 0
        for i, (virt, fl) in enumerate(self.tlb):
            if fl and virt == want:
                flags |= fl
                phys = i
                matches += 1
        res = (phys & 0xFF) | ((flags & 0x7) << 24)
        if matches == 0:
            res |= 0x80000000
        if matches > 1:
            res |= 0x40000000
        return res & 0xFFFFFFFF

    def itlb(self, phys: int) -> None:
        """vm.rst: clear the entry, unless it holds secret code."""
        phys &= 0xFFFFFF
        if not 0 <= phys < self.code_pages:
            return
        if not (self.tlb[phys][1] & FLAG_SECRET):
            self.tlb[phys] = [0, 0]
