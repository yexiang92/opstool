import os
from rich.console import Console


class CONSTANTS:

    CONSOLE = Console()

    PKG_NAME = "OPSTOOL"
    RESULTS_DIR = ".opstool.output"   # _OPSTOOL_ODB
    PKG_PREFIX = f"[bold #cf6275]{PKG_NAME}[/bold #cf6275] [bold #12e193]::[/bold #12e193] "

    MODEL_FILE_NAME = "ModelData"
    EIGEN_FILE_NAME = "EigenData"
    RESP_FILE_NAME = "RespStepData"

    # shape dict used to subplots
    SHAPE_MAP = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3),
        10: (3, 4),
        11: (3, 4),
        12: (3, 4),
        13: (4, 4),
        14: (4, 4),
        15: (4, 4),
        16: (4, 4),
        17: (4, 5),
        18: (4, 5),
        19: (4, 5),
        20: (4, 5),
        21: (5, 5),
        22: (5, 5),
        23: (5, 5),
        24: (5, 5),
        25: (5, 5),
        26: (5, 6),
        27: (5, 6),
        28: (5, 6),
        29: (5, 6),
        30: (5, 6),
        31: (6, 6),
        32: (6, 6),
        33: (6, 6),
        34: (6, 6),
        35: (6, 6),
        36: (6, 6),
        37: (6, 7),
        38: (6, 7),
        39: (6, 7),
        40: (6, 7),
        41: (6, 7),
        42: (6, 7),
        43: (7, 7),
        44: (7, 7),
        45: (7, 7),
        46: (7, 7),
        47: (7, 7),
        48: (7, 7),
        49: (7, 7),
        50: (7, 8),
        51: (7, 8),
        52: (7, 8),
        53: (7, 8),
        54: (7, 8),
        55: (7, 8),
        56: (7, 8),
        57: (8, 8),
        58: (8, 8),
        59: (8, 8),
        60: (8, 8),
        61: (8, 8),
        62: (8, 8),
        63: (8, 8),
        64: (8, 8),
    }

    @classmethod
    def set_output_dir(cls, output_dir: str):
        cls.RESULTS_DIR = output_dir
        if not os.path.exists(cls.RESULTS_DIR):
            os.mkdir(cls.RESULTS_DIR)

    @classmethod
    def get_output_dir(cls):
        return cls.RESULTS_DIR

    @classmethod
    def get_pkg_prefix(cls):
        return cls.PKG_PREFIX

    @classmethod
    def get_console(cls):
        return cls.CONSOLE

    @classmethod
    def get_pkg_name(cls):
        return cls.PKG_NAME

    @classmethod
    def get_model_filename(cls):
        return cls.MODEL_FILE_NAME

    @classmethod
    def get_eigen_filename(cls):
        return cls.EIGEN_FILE_NAME

    @classmethod
    def get_resp_filename(cls):
        return cls.RESP_FILE_NAME

    @classmethod
    def get_shape_map(cls):
        return cls.SHAPE_MAP

class ConfigUpdate:
    def __init__(self):
        self.RESULTS_DIR = None
        self.CONSOLE = None
        self.PKG_PREFIX = None
        self.RESP_FILE_NAME = None
        self.MODEL_FILE_NAME = None

        self.update()

    def update(self):
        self.RESULTS_DIR = CONSTANTS.get_output_dir()
        self.CONSOLE = CONSTANTS.get_console()
        self.PKG_PREFIX = CONSTANTS.get_pkg_prefix()
        self.RESP_FILE_NAME = CONSTANTS.get_resp_filename()
        self.MODEL_FILE_NAME = CONSTANTS.get_model_filename()