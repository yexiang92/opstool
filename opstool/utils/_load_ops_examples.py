from ._util_funcs import get_random_color_rich
from .consts import CONFIGS

CONSOLE = CONFIGS.get_console()
PKG_PREFIX = CONFIGS.get_pkg_prefix()


def run_model(filepath: str):
    """
    Run your OpenSees model python file.

    Parameters
    ----------
    filepath: str
        OpenSees model python file path.

    Example
    --------
    >>> file_path = 'model.py' # in the current directory
    >>> # file_path = "path/to/your/model.py"
    >>> run_model(file_path)

    Returns
    --------
    None
    """
    if not filepath.endswith(".py"):
        filepath += ".py"
    with open(filepath, "r") as f:
        exec(f.read())


def load_ops_examples(name: str):  # noqa: C901
    """Run the pre-built OpenSeesPy model examples in this package.

    Parameters:
    -----------
    name: str,
        Optional, "ArchBridge", "ArchBridge-2", "CableStayedBridge", "SuspensionBridge",
        "TrussBridge", "Frame3D", "Frame3D-2", "GridFrame", "Shell3D",
        "Dam-Brick", "Igloo-Brick", "Pier-Brick", "DamBreak", "FiberSection".

    Example
    --------
    >>> load_ops_examples("SuspensionBridge")

    Returns:
    --------
    None
    """
    if name.lower() == "archbridge":
        from .ops_models.ArchBridge import ArchBridge

        ArchBridge()
        # exec("from opstool.ops_models.ArchBridge import *")
    elif name.lower() in ["archbridge2", "archbridge-2"]:
        from .ops_models.ArchBridge2 import ArchBridge2

        ArchBridge2()
        # exec("from opstool.ops_models.ArchBridge2 import *")
    elif name.lower() == "trussbridge":
        from .ops_models.TrussBridge import TrussBridge

        TrussBridge()
        # exec("from opstool.ops_models.TrussBridge import *")
    elif name.lower() == "cablestayedbridge":
        from .ops_models.CableStayedBridge import CableStayedBridge

        CableStayedBridge()
        # exec("from opstool.ops_models.CableStayedBridge import *")
    elif name.lower() == "dam-brick":
        from .ops_models.Dam import Dam

        Dam()
        # exec("from opstool.ops_models.Dam import *")
    elif name.lower() == "frame3d":
        from .ops_models.Frame3D import Frame3D

        Frame3D()
        # exec("from opstool.ops_models.Frame3D import *")
    elif name.lower() in ["frame3d-2", "frame3d2"]:
        from .ops_models.Frame3D2 import Frame3D2

        Frame3D2()
    elif name.lower() == "igloo-brick":
        from .ops_models.Igloo import Igloo

        Igloo()
        # exec("from opstool.ops_models.Igloo import *")
    elif name.lower() == "pier-brick":
        from .ops_models.Pier import Pier

        Pier()
        # exec("from opstool.ops_models.Pier import *")
    elif name.lower() == "suspensionbridge":
        from .ops_models.SuspensionBridge import SuspensionBridge

        SuspensionBridge()
        # exec("from opstool.ops_models.SuspensionBridge import *")
    elif name.lower() == "FiberSection":
        from .ops_models.FiberSec import FiberSection

        FiberSection()
        # exec("from opstool.ops_models.SDOF import *")
    elif name.lower() == "dambreak":
        from .ops_models.DamBreak import DamBreak

        DamBreak()
        # exec("from opstool.ops_models.DamBreak import *")
    elif name.lower() == "gridframe":
        from .ops_models.GridFrame import GridFrame

        GridFrame()
    elif name.lower() == "shell3d":
        from .ops_models.shell3D import Shell3D

        Shell3D()
    else:
        txt = get_random_color_rich(name, style="bold")
        CONSOLE.print(f"{PKG_PREFIX}Not supported example {txt}!")
        CONFIGS.CONSOLE.print(f"{PKG_PREFIX}Now try treating {txt} as your own model file and run it!")
        run_model(name)
