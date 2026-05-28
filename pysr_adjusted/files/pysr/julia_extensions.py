"""Julia package loading (no-op with Python backend)."""


def load_all_packages():
    """Load all Julia extension packages.

    With the Python backend, this is a no-op since Julia is not used.
    """
    pass


def load_required_packages(**kwargs):
    """Load required Julia packages.

    With the Python backend, this is a no-op since Julia is not used.
    """
    pass
