from ._load_ops_examples import load_ops_examples, run_model
from ._util_funcs import (
    _check_odb_path,
    add_ops_hints_file,
    check_file_type,
    get_color_rich,
    get_cycle_color,
    get_cycle_color_rich,
    get_random_color,
    get_random_color_rich,
    gram_schmidt,
    print_version,
    set_odb_path,
    suppress_ops_print,
)
from .consts import CONFIGS
from .ele_shape_func import get_gp2node_func, get_shape_func, get_shell_gp2node_func
from .ops_ele_class_tags import OPS_ELE_CLASSTAG2TYPE, OPS_ELE_TAGS, OPS_ELE_TYPES

_check_odb_path()


__all__ = [
    # ----------------------
    "CONFIGS",
    "OPS_ELE_CLASSTAG2TYPE",
    # --------------------
    "OPS_ELE_TAGS",
    "OPS_ELE_TYPES",
    "_check_odb_path",
    "add_ops_hints_file",
    # -----------------------
    "check_file_type",
    "get_color_rich",
    "get_cycle_color",
    "get_cycle_color_rich",
    "get_random_color",
    "get_random_color_rich",
    "gram_schmidt",
    "load_ops_examples",
    "print_version",
    "run_model",
    "set_odb_path",
    "suppress_ops_print",
]

__all__ += ["get_gp2node_func", "get_shape_func", "get_shell_gp2node_func"]
