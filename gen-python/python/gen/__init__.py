"""Python bindings to the Gen version control system."""

__version__ = "0.1.0"

# Bindings can come through a Python intermediate layer (helpers.py) or the compiled Rust library itself

try:
    # Directly from Rust
    from .gen import (
        DbContext,
        Repository,
        PyBlockGroup,
        PyHashId,
        PyBaseLayout,
        PyScaledLayout,
        PyNodeKey,
        export_fasta,
        export_genbank,
        export_gfa,
        get_gen_dir,
        import_fasta,
        import_genbank,
        import_gfa,
        import_library,
        init,
        update_with_fasta,
        update_with_gaf,
        update_with_genbank,
        update_with_gfa,
        update_with_library,
        update_with_sequence,
        update_with_vcf,
    )

    # Through Python (helpers.py), currently not used
    # from .helpers import ...

    # Make those classes and functions available at the package level
    __all__ = [
        "DbContext",
        "Repository",
        "PyBlockGroup",
        "PyHashId",
        "PyBaseLayout",
        "PyScaledLayout",
        "PyNodeKey",
        "export_fasta",
        "export_genbank",
        "export_gfa",
        "get_gen_dir",
        "import_fasta",
        "import_genbank",
        "import_gfa",
        "import_library",
        "init",
        "update_with_fasta",
        "update_with_gaf",
        "update_with_genbank",
        "update_with_gfa",
        "update_with_library",
        "update_with_sequence",
        "update_with_vcf",
    ]

except ImportError as e:
    import warnings
    import os

    warnings.warn(f"Failed to import Gen modules: {e}")

    # Try to print diagnostic information to help with troubleshooting
    package_dir = os.path.dirname(__file__)
    warnings.warn(f"Package directory contents: {os.listdir(package_dir)}")

    __all__ = []
