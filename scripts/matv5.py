"""Minimal reader for MATLAB v5 .mat files (numeric arrays, cells, structs, char; zlib-compressed
elements). Only NumPy and zlib are needed, so the data-preparation scripts do not depend on SciPy."""
from __future__ import annotations

import struct
import zlib

import numpy as np

_NUMERIC = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32,
            7: np.float32, 9: np.float64, 12: np.int64, 13: np.uint64}
_MATRIX, _COMPRESSED, _UTF8 = 14, 15, 16
_CELL, _STRUCT, _CHAR = 1, 2, 4


def _tag(buf: bytes, pos: int) -> tuple[int, int, int, int]:
    """(type, nbytes, data_start, next_element)."""
    first, second = struct.unpack_from("<II", buf, pos)
    if first >> 16:                                   # small data element: 4 bytes of data inside the tag
        return first & 0xFFFF, first >> 16, pos + 4, pos + 8
    start = pos + 8
    return first, second, start, start + second + (-second % 8)


def _element(buf: bytes, pos: int):
    kind, size, start, nxt = _tag(buf, pos)
    if kind == _COMPRESSED:
        inner = zlib.decompress(buf[start:start + size])
        return _element(inner, 0)[0], start + size
    if kind == _MATRIX:
        return _matrix(buf[start:start + size]), nxt
    if kind in _NUMERIC:
        return np.frombuffer(buf, dtype=_NUMERIC[kind], count=size // np.dtype(_NUMERIC[kind]).itemsize,
                             offset=start).copy(), nxt
    if kind == _UTF8:
        return buf[start:start + size].decode("utf-8"), nxt
    raise ValueError(f"unsupported MAT element type {kind}")


def _matrix(buf: bytes):
    if not buf:
        return np.zeros((0, 0))
    flags, pos = _element(buf, 0)
    mclass = int(flags[0]) & 0xFF
    dims, pos = _element(buf, pos)
    dims = tuple(int(d) for d in dims)
    _, pos = _element(buf, pos)                       # array name
    count = int(np.prod(dims))
    if mclass == _CELL:
        items = []
        for _ in range(count):
            item, pos = _element(buf, pos)
            items.append(item)
        out = np.empty(count, dtype=object)
        out[:] = items
        return out.reshape(dims, order="F")
    if mclass == _STRUCT:
        length, pos = _element(buf, pos)
        raw, pos = _element(buf, pos)
        width = int(length[0])
        names = [bytes(raw[i:i + width]).rstrip(b"\x00").decode() for i in range(0, len(raw), width)]
        records = []
        for _ in range(count):
            record = {}
            for name in names:
                record[name], pos = _element(buf, pos)
            records.append(record)
        return records[0] if count == 1 else records
    if mclass == _CHAR:
        data, pos = _element(buf, pos)
        if isinstance(data, str):                   # stored as miUTF8
            return data
        return "".join(chr(int(c)) for c in np.ravel(data, order="F"))
    real, pos = _element(buf, pos)
    return np.asarray(real, dtype=np.float64).reshape(dims, order="F")


def loadmat(path: str) -> dict:
    buf = open(path, "rb").read()
    if buf[126:128] != b"IM":
        raise ValueError("only little-endian MAT v5 files are supported")
    out, pos = {}, 128
    while pos < len(buf):
        kind, size, start, nxt = _tag(buf, pos)
        if kind == _COMPRESSED:
            inner = zlib.decompress(buf[start:start + size])
            name, value = _named(inner)
            pos = start + size
        elif kind == _MATRIX:
            name, value = _named(buf[pos:nxt])
            pos = nxt
        else:
            pos = nxt
            continue
        out[name] = value
    return out


def _named(buf: bytes):
    kind, size, start, _ = _tag(buf, 0)
    body = buf[start:start + size]
    _, pos = _element(body, 0)
    _, pos = _element(body, pos)
    name, _ = _element(body, pos)
    name = bytes(np.asarray(name, dtype=np.uint8)).decode() if not isinstance(name, str) else name
    return name, _matrix(body)
