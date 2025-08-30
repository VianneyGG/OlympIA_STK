import os
from pathlib import Path
from typing import Any


def path_to_str(p: Any) -> Any:
    """Convert Path or os.PathLike to str; leave other types unchanged.

    Used to ensure values passed to tarfile or that are checked with
    .startswith are plain strings.
    """
    try:
        # Path and PosixPath are instances of os.PathLike, so this will
        # convert them to a str. If p is already a str, this is a no-op.
        if isinstance(p, os.PathLike):
            return str(p)
    except Exception:
        pass
    return p


def normalize_path_in_iterable(obj: Any) -> Any:
    """Recursively convert path-like objects in common containers to str.

    This helps when lists/dicts of paths may be passed to libraries that
    expect strings.
    """
    if isinstance(obj, dict):
        return {k: normalize_path_in_iterable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        container_type = type(obj)
        converted = [normalize_path_in_iterable(x) for x in obj]
        return container_type(converted)
    return path_to_str(obj)


def startswith_safe(s: Any, prefix: str) -> bool:
    """Check startswith on strings and path-likes by coercing to str first.

    Example:
        if startswith_safe(name, os.sep):
            ...
    """
    s2 = path_to_str(s)
    if not isinstance(s2, str):
        return False
    return s2.startswith(prefix)
