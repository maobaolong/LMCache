# SPDX-License-Identifier: Apache-2.0
"""POSIX shared-memory primitives shared by SHM-based transports.

Hand-rolled ``shm_open`` / ``mmap`` helpers on top of libc/librt via
``ctypes``. Kept here (rather than under ``platform.cpu``) because
the same primitives are needed by every SHM transport, not only the
CPU-only KV-cache wrapper:

* :mod:`lmcache.v1.platform.cpu.shm` -- per-tensor SHM segments used
  as a CUDA-IPC equivalent on CPU-only hosts.
* The MP non-GPU SHM transport (single L1-pool segment shared by the
  server and worker processes).

We deliberately do not use stdlib ``mmap`` for the segment-creation
path: ``mmap.mmap`` would work for the in-process side, but we still
need ``shm_open`` / ``shm_unlink`` (not exposed by stdlib), and we
hand the raw address to ``ctypes.from_address`` + ``torch.frombuffer``
to share storage with a migrated tensor. So we keep the raw mmap
pointers and pair every successful mmap with a matching ``munmap``
on the caller side.

TODO(maobaolong): replace with ``posix_ipc`` once we are willing to
take that runtime dependency.
"""

# Future
from __future__ import annotations

# Standard
import ctypes
import ctypes.util
import mmap
import os

_O_RDWR = os.O_RDWR
_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_SHARED = 0x01

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
# macOS exposes shm_open in libSystem (== libc), Linux needs librt.
_librt = _libc
if not hasattr(_libc, "shm_open"):
    _librt = ctypes.CDLL(ctypes.util.find_library("rt"), use_errno=True)

_librt.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32]
_librt.shm_open.restype = ctypes.c_int
_librt.shm_unlink.argtypes = [ctypes.c_char_p]
_librt.shm_unlink.restype = ctypes.c_int

_libc.ftruncate.argtypes = [ctypes.c_int, ctypes.c_int64]
_libc.ftruncate.restype = ctypes.c_int
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int64,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.munmap.restype = ctypes.c_int
_libc.close.argtypes = [ctypes.c_int]
_libc.close.restype = ctypes.c_int

MAP_FAILED = ctypes.c_void_p(-1).value


def shm_create_readwrite(name: str, nbytes: int) -> int:
    """Create + size a POSIX SHM segment, return mapped address.

    Every failure path tears down whatever has already been allocated
    (fd, named segment) so the caller never has to compensate.
    """
    name_b = name.encode("ascii")
    fd = _librt.shm_open(name_b, _O_RDWR | _O_CREAT | _O_EXCL, 0o600)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "shm_open(create) failed for %s" % name)
    addr = 0
    try:
        if _libc.ftruncate(fd, nbytes) != 0:
            raise OSError(ctypes.get_errno(), "ftruncate failed for %s" % name)
        addr = _libc.mmap(None, nbytes, _PROT_READ | _PROT_WRITE, _MAP_SHARED, fd, 0)
        if addr in (0, MAP_FAILED):
            addr = 0
            raise OSError(ctypes.get_errno(), "mmap failed for %s" % name)
    except BaseException:
        # Roll back whatever we have so far: the named segment is
        # always created at this point; mmap may or may not have
        # succeeded.
        if addr:
            _libc.munmap(ctypes.c_void_p(addr), nbytes)
        _librt.shm_unlink(name_b)
        raise
    finally:
        _libc.close(fd)
    return addr


def shm_map_readwrite(name: str, nbytes: int) -> int:
    """Open an existing POSIX SHM segment, return mapped address.

    The fd is always closed before returning (success or failure) so
    we never leak a file descriptor even when ``mmap`` fails.
    """
    fd = _librt.shm_open(name.encode("ascii"), _O_RDWR, 0o600)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "shm_open(open) failed for %s" % name)
    try:
        addr = _libc.mmap(None, nbytes, _PROT_READ | _PROT_WRITE, _MAP_SHARED, fd, 0)
        if addr in (0, MAP_FAILED):
            raise OSError(ctypes.get_errno(), "mmap failed for %s" % name)
    finally:
        _libc.close(fd)
    return addr


def shm_munmap(addr: int, nbytes: int) -> None:
    """Best-effort ``munmap`` of a previously ``mmap``-ed SHM segment."""
    if not addr or addr == MAP_FAILED:
        return
    _libc.munmap(ctypes.c_void_p(addr), nbytes)


def shm_unlink(name: str) -> None:
    """Best-effort SHM segment removal."""
    _librt.shm_unlink(name.encode("ascii"))


def shm_open_pool_as_mmap(name: str, nbytes: int) -> mmap.mmap:
    """Open an existing POSIX SHM segment as a Python ``mmap.mmap``.

    Convenience helper for non-GPU SHM transports that want to consume
    the segment via ``torch.frombuffer(mmap_obj, ...)`` rather than a
    raw address. The fd is closed once the mapping is established
    (the kernel keeps the mapping alive until ``mmap.close()``).
    """
    fd = _librt.shm_open(name.encode("ascii"), _O_RDWR, 0o600)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "shm_open(open) failed for %s" % name)
    try:
        return mmap.mmap(fd, nbytes, access=mmap.ACCESS_WRITE)
    finally:
        _libc.close(fd)
