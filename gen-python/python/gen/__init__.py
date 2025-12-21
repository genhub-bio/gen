"""Python bindings to the Gen version control system."""

__version__ = "0.1.0"

# Bindings can come through a Python intermediate layer (helpers.py) or the compiled Rust library itself

try:
    # Directly from Rust
    from .gen import (
        DbContext,
        Repository,
        PyBlockGroup,
        PyBaseLayout,
        PyScaledLayout,
        export_fasta,
        get_gen_dir,
        import_fasta,
        init,
        update_with_fasta,
    )

    # Through Python (helpers.py), currently not used
    # from .helpers import ...

    # Make those classes and functions available at the package level
    __all__ = [
        "DbContext",
        "Repository",
        "PyBlockGroup",
        "PyBaseLayout",
        "PyScaledLayout",
        "export_fasta",
        "get_gen_dir",
        "import_fasta",
        "init",
        "update_with_fasta",
    ]

except ImportError as e:
    import warnings
    import os

    warnings.warn(f"Failed to import Gen modules: {e}")

    # Try to print diagnostic information to help with troubleshooting
    package_dir = os.path.dirname(__file__)
    warnings.warn(f"Package directory contents: {os.listdir(package_dir)}")

    __all__ = []
