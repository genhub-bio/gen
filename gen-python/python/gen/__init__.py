"""Python bindings to the Gen version control system."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gen")
except PackageNotFoundError:
    __version__ = "0.0.0"


# Bindings can come through a Python intermediate layer (helpers.py) or the compiled Rust library itself

try:
    # Directly from Rust
    from .gen import (
        DbContext,
        PyBlockGroup,
        PyHashId,
        PyNodeKey,
        Repository,
        PySequencePart,
        derive_chunks,
        derive_subgraph,
        export_fasta,
        export_genbank,
        export_gfa,
        get_gen_dir,
        import_fasta,
        import_genbank,
        import_gfa,
        import_library,
        init,
        make_stitch,
        update_with_fasta,
        update_with_gaf,
        update_with_genbank,
        update_with_gfa,
        update_with_library,
        update_with_sequence,
        update_with_vcf,
    )

    # Jupyter widget — only available with `pip install gen[jupyter]`
    try:
        from .jupyter_widget import GenGraphWidget
    except ImportError:
        GenGraphWidget = None

    # Make those classes and functions available at the package level
    __all__ = [
        "DbContext",
        "Repository",
        "PyBlockGroup",
        "PyHashId",
        "PyNodeKey",
        "PySequencePart",
        "GenGraphWidget",
        "derive_chunks",
        "derive_subgraph",
        "export_fasta",
        "export_genbank",
        "export_gfa",
        "get_gen_dir",
        "import_fasta",
        "import_genbank",
        "import_gfa",
        "import_library",
        "init",
        "make_stitch",
        "update_with_fasta",
        "update_with_gaf",
        "update_with_genbank",
        "update_with_gfa",
        "update_with_library",
        "update_with_sequence",
        "update_with_vcf",
    ]

except ImportError as e:
    import os
    import warnings

    warnings.warn(f"Failed to import Gen modules: {e}")

    # Try to print diagnostic information to help with troubleshooting
    package_dir = os.path.dirname(__file__)
    warnings.warn(f"Package directory contents: {os.listdir(package_dir)}")

    __all__ = []
