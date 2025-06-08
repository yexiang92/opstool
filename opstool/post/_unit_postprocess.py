from typing import Optional, Union

from ..pre import UnitSystem

# from opstool.pre import UnitSystem

FORCE_MAP = {"lb": "lb", "lbf": "lbf", "kip": "kip", "n": "N", "kn": "kN", "mn": "MN", "kgf": "kgf", "tonf": "tonf"}
LENGTH_MAP = {"inch": "inch", "ft": "ft", "mm": "mm", "cm": "cm", "m": "m", "km": "km"}
STRESS_MAP = {
    "n/m": "Pa",
    "kn/m": "kPa",
    "mn/m": "MPa",
    "n/mm": "MPa",
    "gn/m": "GPa",
    "kn/mm": "GPa",
    "lbf/inch": "psi",
    "lb/inch": "psi",
    "kip/inch": "ksi",
    "lbf/ft": "psf",
    "lb/ft": "psf",
    "kip/ft": "ksf",
}


class UnitPostProcess:
    def __init__(
        self, basic_length: Optional[str] = None, basic_force: Optional[str] = None, basic_time: Optional[str] = None
    ):
        self.unit_system = None

        if basic_length is None:
            basic_length = "m"
        if basic_force is None:
            basic_force = "N"
        if basic_time is None:
            basic_time = "sec"

        self.set_basic_units(length=basic_length, force=basic_force, time=basic_time)

    def set_basic_units(self, length: Optional[str] = None, force: Optional[str] = None, time: Optional[str] = None):
        self.unit_system = UnitSystem(length=length, force=force, time=time)

    def get_force_multiplier(self, force_unit: Optional[str] = None) -> float:
        if force_unit is None:
            return 1.0
        else:
            factor = getattr(self.unit_system, force_unit)
            if factor == 0:
                return 1.0
            else:
                return 1 / factor

    def get_force_per_length_multiplier(
        self, force_unit: Optional[str] = None, length_unit: Optional[str] = None
    ) -> float:
        if force_unit is None or length_unit is None:
            return 1.0
        else:
            factor = getattr(self.unit_system, force_unit) / getattr(self.unit_system, length_unit)
            if factor == 0:
                return 1.0
            else:
                return 1 / factor

    def get_moment_multiplier(self, force_unit: Optional[str] = None, length_unit: Optional[str] = None) -> float:
        if force_unit is None or length_unit is None:
            return 1.0
        else:
            factor = getattr(self.unit_system, force_unit) * getattr(self.unit_system, length_unit)
            if factor == 0:
                return 1.0
            else:
                return 1 / factor

    def get_moment_per_length_multiplier(
        self, force_unit: Optional[str] = None, length_unit: Optional[str] = None
    ) -> float:
        if force_unit is None or length_unit is None:
            return 1.0
        else:
            factor = getattr(self.unit_system, force_unit) * getattr(self.unit_system, length_unit)
            factor = factor / getattr(self.unit_system, length_unit)
            if factor == 0:
                return 1.0
            else:
                return 1 / factor

    def get_curvature_multiplier(self, length_unit: Optional[str] = None) -> float:
        if length_unit is None:
            return 1.0
        else:
            factor = getattr(self.unit_system, length_unit)
            if factor == 0:
                return 1.0
            else:
                return factor

    def get_stress_multiplier(self, force_unit: Optional[str] = None, length_unit: Optional[str] = None) -> float:
        if force_unit is None or length_unit is None:
            return 1.0
        else:
            force = getattr(self.unit_system, force_unit)
            length = getattr(self.unit_system, length_unit)
            factor = force / (length**2)
            if factor == 0:
                return 1.0
            else:
                return 1 / factor

    def get_pressure_multiplier(self, force_unit: Optional[str] = None, length_unit: Optional[str] = None) -> float:
        return self.get_stress_multiplier(force_unit=force_unit, length_unit=length_unit)

    def get_disp_multiplier(self, length_unit: Optional[str] = None) -> Union[None, float]:
        if length_unit is None:
            return 1.0
        else:
            length = getattr(self.unit_system, length_unit)
            if length == 0:
                return 1.0
            else:
                return 1 / length

    def get_vel_multiplier(
        self, length_unit: Optional[str] = None, time_unit: Optional[str] = None
    ) -> Union[None, float]:
        if length_unit is None or time_unit is None:
            return 1.0
        else:
            length = getattr(self.unit_system, length_unit)
            time = getattr(self.unit_system, time_unit)
            factor = length / time
            if factor == 0:
                return 1.0
            else:
                return 1 / factor

    def get_accel_multiplier(
        self, length_unit: Optional[str] = None, time_unit: Optional[str] = None
    ) -> Union[None, float]:
        if length_unit is None or time_unit is None:
            return 1.0
        else:
            length = getattr(self.unit_system, length_unit)
            time = getattr(self.unit_system, time_unit)
            factor = length / time / time
            if factor == 0:
                return 1.0
            else:
                return 1 / factor

    def get_angular_vel_multiplier(self, time_unit: Optional[str] = None) -> Union[None, float]:
        if time_unit is None:
            return 1.0
        else:
            factor = 1 / getattr(self.unit_system, time_unit)
            if factor == 0:
                return 1.0
            else:
                return 1 / factor

    def get_angular_accel_multiplier(self, time_unit: Optional[str] = None) -> Union[None, float]:
        if time_unit is None:
            return 1.0
        else:
            factor = 1 / (getattr(self.unit_system, time_unit) ** 2)
            if factor == 0:
                return 1.0
            else:
                return 1 / factor

    @staticmethod
    def get_force_symbol(force_unit: Optional[str] = None) -> Union[None, str]:
        if force_unit is None:
            return None
        else:
            return FORCE_MAP.get(force_unit.lower(), force_unit)

    @staticmethod
    def get_force_per_length_symbol(
        force_unit: Optional[str] = None, length_unit: Optional[str] = None
    ) -> Union[None, str]:
        if force_unit is None or length_unit is None:
            return None
        else:
            fsym = FORCE_MAP.get(force_unit.lower(), force_unit)
            lsym = LENGTH_MAP.get(length_unit.lower(), length_unit)
            return fsym + "/" + lsym

    @staticmethod
    def get_moment_symbol(force_unit: Optional[str] = None, length_unit: Optional[str] = None) -> Union[None, str]:
        if force_unit is None or length_unit is None:
            return None
        else:
            fsym = FORCE_MAP.get(force_unit.lower(), force_unit)
            lsym = LENGTH_MAP.get(length_unit.lower(), length_unit)
            return fsym + "·" + lsym

    @staticmethod
    def get_moment_per_length_symbol(
        force_unit: Optional[str] = None, length_unit: Optional[str] = None
    ) -> Union[None, str]:
        if force_unit is None or length_unit is None:
            return None
        else:
            fsym = FORCE_MAP.get(force_unit.lower(), force_unit)
            lsym = LENGTH_MAP.get(length_unit.lower(), length_unit)
            return fsym + "·" + lsym + "/" + lsym

    @staticmethod
    def get_curvature_symbol(length_unit: Optional[str] = None) -> Union[None, str]:
        if length_unit is None:
            return None
        else:
            lsm = LENGTH_MAP.get(length_unit.lower(), length_unit)
            return lsm + "⁻¹"

    @staticmethod
    def get_stress_symbol(force_unit: Optional[str] = None, length_unit: Optional[str] = None) -> Union[None, str]:
        if force_unit is None or length_unit is None:
            return None
        else:
            key = force_unit.lower() + "/" + length_unit.lower()
            return STRESS_MAP.get(key, force_unit + "/" + length_unit + "²")

    def get_pressure_symbol(
        self, force_unit: Optional[str] = None, length_unit: Optional[str] = None
    ) -> Union[None, str]:
        return self.get_stress_symbol(force_unit=force_unit, length_unit=length_unit)

    @staticmethod
    def get_disp_symbol(length_unit: Optional[str] = None) -> Union[None, str]:
        if length_unit is None:
            return None
        else:
            return LENGTH_MAP.get(length_unit.lower(), length_unit)

    @staticmethod
    def get_vel_symbol(length_unit: Optional[str] = None, time_unit: Optional[str] = None) -> Union[None, str]:
        if length_unit is None or time_unit is None:
            return None
        else:
            return LENGTH_MAP.get(length_unit.lower(), length_unit) + "/" + time_unit.lower()

    @staticmethod
    def get_accel_symbol(length_unit: Optional[str] = None, time_unit: Optional[str] = None) -> Union[None, str]:
        if length_unit is None or time_unit is None:
            return None
        else:
            return LENGTH_MAP.get(length_unit.lower(), length_unit) + "/" + time_unit.lower() + "²"

    @staticmethod
    def get_rotation_symbol() -> Union[None, str]:
        return "rad"

    @staticmethod
    def get_angular_vel_symbol(time_unit: Optional[str] = None) -> Union[None, str]:
        if time_unit is None:
            return None
        else:
            return "rad" + "/" + time_unit

    @staticmethod
    def get_angular_accel_symbol(time_unit: Optional[str] = None) -> Union[None, str]:
        if time_unit is None:
            return None
        else:
            return "rad" + "/" + time_unit + "²"

    @staticmethod
    def get_strain_multiplier() -> float:
        return 1.0

    @staticmethod
    def get_strain_symbol() -> Union[None, str]:
        return "-"


def get_post_unit_multiplier(
    analysis_length: Optional[str] = None,
    analysis_force: Optional[str] = None,
    analysis_time: Optional[str] = None,
    post_length: Optional[str] = None,
    post_force: Optional[str] = None,
    post_time: Optional[str] = None,
) -> dict:
    """Get post unit multiplier.

    Parameters
    --------------
    analysis_length: str, default None
        Length unit used in analysis.
        Optional ["inch", "ft", "mm", "cm", "m", "km"]
    analysis_force: str, default None
        Force unit used in analysis.
        Optional ["lb"("lbf"), "kip", "n", "kn", "mn", "kgf", "tonf"].
    analysis_time:  str, default None
        Time unit used in analysis. Optional ["sec"].
    post_length: str, default None
        Length unit will be used in post-processing.
        Optional ["inch", "ft", "mm", "cm", "m", "km"]
    post_force: str, default None
        Force unit will be used in post-processing.
        Optional ["lb"("lbf"), "kip", "n", "kn", "mn", "kgf", "tonf"].
    post_time: str, default None
        Time unit will be used in post-processing. Optional ["sec"].

    Returns
    ---------
    dict[str, float]
    """
    if analysis_time is None:
        analysis_time = "sec"
    if post_time is None:
        post_time = "sec"
    unit_system_post = UnitPostProcess(
        basic_length=analysis_length, basic_force=analysis_force, basic_time=analysis_time
    )
    disp_fact = unit_system_post.get_disp_multiplier(length_unit=post_length)
    vel_fact = unit_system_post.get_vel_multiplier(length_unit=post_length, time_unit=post_time)
    acc_fact = unit_system_post.get_accel_multiplier(length_unit=post_length, time_unit=post_time)
    force_fact = unit_system_post.get_force_multiplier(force_unit=post_force)
    moment_fact = unit_system_post.get_moment_multiplier(force_unit=post_force, length_unit=post_length)
    curvature_fact = unit_system_post.get_curvature_multiplier(length_unit=post_length)
    stress_fact = unit_system_post.get_stress_multiplier(force_unit=post_force, length_unit=post_length)
    force_fact_per_length = unit_system_post.get_force_per_length_multiplier(
        force_unit=post_force, length_unit=post_length
    )
    moment_fact_per_length = unit_system_post.get_moment_per_length_multiplier(
        force_unit=post_force, length_unit=post_length
    )
    angular_vel_fact = unit_system_post.get_angular_vel_multiplier(time_unit=post_time)
    angular_accel_fact = unit_system_post.get_angular_accel_multiplier(time_unit=post_time)

    fact = {
        "disp": disp_fact,
        "vel": vel_fact,
        "accel": acc_fact,
        "force": force_fact,
        "moment": moment_fact,
        "curvature": curvature_fact,
        "stress": stress_fact,
        "force_per_length": force_fact_per_length,
        "moment_per_length": moment_fact_per_length,
        "angular_vel": angular_vel_fact,
        "angular_accel": angular_accel_fact,
    }

    return fact


def get_post_unit_symbol(
    analysis_length: Optional[str] = None,
    analysis_force: Optional[str] = None,
    analysis_time: Optional[str] = None,
    post_length: Optional[str] = None,
    post_force: Optional[str] = None,
    post_time: Optional[str] = None,
) -> dict:
    """Get post unit symbol.

    Parameters
    --------------
    analysis_length: str, default None
        Length unit used in analysis.
        Optional ["inch", "ft", "mm", "cm", "m", "km"]
    analysis_force: str, default None
        Force unit used in analysis.
        Optional ["lb"("lbf"), "kip", "n", "kn", "mn", "kgf", "tonf"].
    analysis_time:  str, default None
        Time unit used in analysis. Optional ["sec"].
    post_length: str, default None
        Length unit will be used in post-processing.
        Optional ["inch", "ft", "mm", "cm", "m", "km"]
    post_force: str, default None
        Force unit will be used in post-processing.
        Optional ["lb"("lbf"), "kip", "n", "kn", "mn", "kgf", "tonf"].
    post_time: str, default None
        Time unit will be used in post-processing. Optional ["sec"].

    Returns
    ---------
    dict[str, str]
    """
    if analysis_time is None:
        analysis_time = "sec"
    if post_time is None:
        post_time = "sec"
    unit_system_post = UnitPostProcess(
        basic_length=analysis_length, basic_force=analysis_force, basic_time=analysis_time
    )
    disp_sym = unit_system_post.get_disp_symbol(length_unit=post_length)
    vel_sym = unit_system_post.get_vel_symbol(length_unit=post_length, time_unit=post_time)
    acc_sym = unit_system_post.get_accel_symbol(length_unit=post_length, time_unit=post_time)
    force_sym = unit_system_post.get_force_symbol(force_unit=post_force)
    moment_sym = unit_system_post.get_moment_symbol(force_unit=post_force, length_unit=post_length)
    curvature_sym = unit_system_post.get_curvature_symbol(length_unit=post_length)
    stress_sym = unit_system_post.get_stress_symbol(force_unit=post_force, length_unit=post_length)
    force_sym_per_length = unit_system_post.get_force_per_length_symbol(force_unit=post_force, length_unit=post_length)
    moment_sym_per_length = unit_system_post.get_moment_per_length_symbol(
        force_unit=post_force, length_unit=post_length
    )
    angular_vel = unit_system_post.get_angular_vel_symbol(time_unit=post_time)
    angular_accel = unit_system_post.get_angular_accel_symbol(time_unit=post_time)

    sym = {
        "disp": disp_sym,
        "vel": vel_sym,
        "accel": acc_sym,
        "force": force_sym,
        "moment": moment_sym,
        "curvature": curvature_sym,
        "stress": stress_sym,
        "force_per_length": force_sym_per_length,
        "moment_per_length": moment_sym_per_length,
        "angular_vel": angular_vel,
        "angular_accel": angular_accel,
    }

    return sym


if __name__ == "__main__":
    data = get_post_unit_multiplier(
        analysis_length="m",
        analysis_force="kN",
        analysis_time="sec",
        post_length="mm",
        post_force="N",
        post_time="sec",
    )
    print(data)

    data = get_post_unit_symbol(
        analysis_length="m",
        analysis_force="kN",
        analysis_time="sec",
        post_length="mm",
        post_force="N",
        post_time="sec",
    )
    print(data)
