from . import __version__  # noqa: F401  (sets BLAS thread variables before NumPy loads)
from .cli import main

main()
