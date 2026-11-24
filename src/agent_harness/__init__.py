"""Controlled self-evolving harness around the official CPU baselines.

The official model.py sets BLAS thread variables before importing NumPy. Once NumPy is loaded
those variables no longer take effect, so they are set here, before any submodule imports NumPy;
otherwise the thread limit silently fails and training-time accounting is wrong.
"""
import os as _os

_threads = _os.environ.setdefault("BASELINE_CPU_THREADS", "1")
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_name, _threads)

__version__ = "0.2.0"
