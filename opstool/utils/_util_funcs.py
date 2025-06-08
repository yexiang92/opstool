import os
import shutil
import sys
from contextlib import contextmanager
from itertools import cycle
from pathlib import Path
from typing import Union

import numpy as np

from .consts import CONFIGS

CONSOLE = CONFIGS.get_console()
PKG_PREFIX = CONFIGS.get_pkg_prefix()


RESULTS_DIR = CONFIGS.get_output_dir()


def _check_odb_path():
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)


def set_odb_path(path: str):
    """Set the output directory for the results saving.

    Parameters:
    ------------
    path: str
        The path to the output directory.
    """
    CONFIGS.set_output_dir(path)
    if os.path.exists(RESULTS_DIR):
        for item in os.listdir(RESULTS_DIR):
            source_path = os.path.join(RESULTS_DIR, item)
            target_path = os.path.join(path, item)
            shutil.move(source_path, target_path)
        shutil.rmtree(RESULTS_DIR)


def check_file_type(file_name: str, file_type: Union[str, list, tuple]):
    """Check a file type.

    Parameters
    ----------
    file_name: str
        The file to be checked.
    file_type: Union[str, list, tuple]
        The target file type.

    Returns
    -------
    None
    """
    if file_name:
        if isinstance(file_type, str):
            if not file_name.endswith(file_type):
                raise ValueError(f"file must be endswith {file_type}!")  # noqa: TRY003
        elif isinstance(file_type, (list, tuple)):
            check = False
            for type_ in file_type:
                if file_name.endswith(type_):
                    check = True
            if not check:
                raise ValueError(f"file must be endswith in {file_type}!")  # noqa: TRY003
        else:
            raise ValueError("file_type must be str or list or tuple!")  # noqa: TRY003


def add_ops_hints_file():
    """
    Add ``opensees.pyi`` file to the OpenSeesPy package directory.
    This file can help you better with type hints and code completion.

    Example
    -------
    >>> add_ops_hints_file()
    """
    src_file = Path(__file__).resolve().parent / "opensees.pyi"
    if sys.platform.startswith("linux"):
        import openseespylinux.opensees as ops

        tar_file = Path(ops.__file__).resolve().parent / "opensees.pyi"
    elif sys.platform.startswith("win"):
        import openseespywin.opensees as ops

        tar_file = Path(ops.__file__).resolve().parent / "opensees.pyi"
    elif sys.platform.startswith("darwin"):
        import openseespymac.opensees as ops

        tar_file = Path(ops.__file__).resolve().parent / "opensees.pyi"
    else:
        raise RuntimeError(sys.platform + " is not supported yet")
    tar_file.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")
    txt1 = get_cycle_color_rich("opensees.pyi", style="bold")
    txt2 = get_cycle_color_rich(tar_file, style="bold")
    CONSOLE.print(f"{PKG_PREFIX}{txt1} file has been created to {txt2}!")


def print_version():
    """Print pacakge version."""
    from ..__about__ import __version__

    CONSOLE.print(__version__, style="bold #0343df")


def get_random_color():
    colors = [
        "#00aeff",
        "#3369e7",
        "#8e43e7",
        "#b84592",
        "#ff4f81",
        "#ff6c5f",
        "#ffc168",
        "#2dde98",
        "#1cc7d0",
        "#ce181e",
        "#007cc0",
        "#ffc20e",
        "#0099e5",
        "#ff4c4c",
        "#34bf49",
        "#d20962",
        "#f47721",
        "#00c16e",
        "#7552cc",
        "#00bce4",
    ]
    idx = np.random.choice(15)
    return colors[idx]


def get_cycle_color():
    colors = [
        "#00aeff",
        "#3369e7",
        "#8e43e7",
        "#b84592",
        "#ff4f81",
        "#ff6c5f",
        "#ffc168",
        "#2dde98",
        "#1cc7d0",
        "#ce181e",
        "#007cc0",
        "#ffc20e",
        "#0099e5",
        "#ff4c4c",
        "#34bf49",
        "#d20962",
        "#f47721",
        "#00c16e",
        "#7552cc",
        "#00bce4",
    ]
    return cycle(colors)


def get_random_color_rich(txt, style: str = "bold"):
    color = get_random_color()
    return f"[{style} {color}]{txt}[/{style} {color}]"


def get_cycle_color_rich(txt, style: str = "bold"):
    color = get_cycle_color()
    return f"[{style} {color}]{txt}[/{style} {color}]"


def get_color_rich(txt, color: str = "#0343df", style: str = "bold"):
    return f"[{style} {color}]{txt}[/{style} {color}]"


def gram_schmidt(v1, v2):
    x, y_ = v1, v2
    y = y_ - (np.dot(x, y_) / np.dot(x, x)) * x
    z = np.cross(x, y)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    return x, y, z


# Context manager to temporarily suppress stdout and stderr
@contextmanager
def suppress_ops_print():
    # Save the original stdout and stderr
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        # Redirect stdout and stderr to null (discard output)
        with open(os.devnull, "w") as fnull:
            sys.stdout = fnull
            sys.stderr = fnull
            yield
    finally:
        # Restore the original stdout and stderr
        sys.stdout = stdout
        sys.stderr = stderr
