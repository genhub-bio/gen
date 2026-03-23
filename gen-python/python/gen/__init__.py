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

    # ---------------------------------------------------------------------------
    # Jupyter display hook for PyBlockGroup
    #
    # When the gen_widget package is installed (i.e. the WASM bundle has been
    # built and placed in gen/static/widget.js), evaluating a PyBlockGroup in a
    # notebook cell renders the interactive graph widget automatically.
    #
    # Gracefully degrades to plain __repr__ if:
    #   - anywidget is not installed
    #   - the widget.js bundle has not been built yet
    #   - the block group has no associated repository
    # ---------------------------------------------------------------------------

    def _in_jupyter():
        try:
            from IPython import get_ipython
            ip = get_ipython()
            return ip is not None and 'IPKernelApp' in ip.config
        except ImportError:
            return False

    if _in_jupyter():
        def _block_group_ipython_display_(self, **kwargs):
            if self.repository is None:
                print(repr(self))
                return
            try:
                from .widget import GenGraphWidget
                from IPython.display import display
                display(GenGraphWidget(self, self.repository))
            except ImportError:
                print(repr(self))

        PyBlockGroup._ipython_display_ = _block_group_ipython_display_

    # Through Python (helpers.py), currently not used
    # from .helpers import ...

    # Make those classes and functions available at the package level
    __all__ = [
        "DbContext",
        "Repository",
        "PyBlockGroup",
        "PyHashId",
        "PyNodeKey",
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
