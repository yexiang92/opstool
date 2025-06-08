import difflib
import re
from typing import Literal, TypeAlias, get_args

ratio_length = {
    "inch2m": 0.0254,
    "inch2dm": 0.254,
    "inch2cm": 2.54,
    "inch2mm": 25.4,
    "inch2km": 2.54e-5,
    "inch2ft": 0.0833333333,
    "ft2mm": 304.8,
    "ft2cm": 30.48,
    "ft2dm": 3.048,
    "ft2m": 0.3048,
    "ft2km": 3.048e-4,
    "mm2cm": 0.1,
    "mm2dm": 0.01,
    "mm2m": 0.001,
    "mm2km": 1e-6,
    "cm2dm": 0.1,
    "cm2m": 0.01,
    "cm2km": 1e-5,
    "m2km": 1e-3,
}
ratio_force = {
    "lb2lbf": 1,
    "lb2kip": 0.001,
    "lb2n": 4.4482216282509,
    "lb2kn": 4.4482216282509e-3,
    "lb2mn": 4.448e-6,
    "lb2kgf": 0.45359236844386,
    "lb2tonf": 0.0004535923699994,
    "lbf2kip": 0.001,
    "lbf2n": 4.4482216282509,
    "lbf2kn": 4.4482216282509e-3,
    "lbf2mn": 4.448e-6,
    "lbf2kgf": 0.45359236844386,
    "lbf2tonf": 0.0004535923699994,
    "kip2n": 4448.2216,
    "kip2kn": 4448.2216e-3,
    "kip2mn": 4448.2216e-6,
    "kip2kgf": 453.59236844386,
    "kip2tonf": 0.4535923699994,
    "n2kn": 1e-3,
    "n2mn": 1e-6,
    "n2kgf": 0.10197162129779283,
    "n2tonf": 0.0001019716212978,
    "kn2mn": 1e-3,
    "kn2kgf": 101.9716212978,
    "kn2tonf": 0.1019716212978,
    "mn2kgf": 101971.6212978,
    "mn2tonf": 101.9716212978,
    "kgf2tonf": 0.001,
}
ratio_time = {
    "sec2msec": 1000,
    "sec2min": 1 / 60,
    "sec2hour": 1 / 3600,
    "sec2day": 1 / 24 / 3600,
    "sec2year": 1 / 365 / 24 / 3600,
    "min2msec": 1000 * 60,
    "min2hour": 1 / 60,
    "min2day": 1 / 24 / 60,
    "min2year": 1 / 365 / 24 / 60,
    "hour2msec": 60 * 60 * 1000,
    "hour2day": 1 / 24,
    "hour2year": 1 / 365 / 24,
    "day2msec": 24 * 60 * 60 * 1000,
    "day2hour": 1 / 24,
    "day2year": 1 / 365 / 24,
    "year2msec": 365 * 24 * 60 * 60 * 1000,
}


def ratio_update(ratio_dict):
    temp_dict = {}
    for key, value in ratio_dict.items():
        idx = key.index("2")
        new_key = key[idx + 1 :] + "2" + key[:idx]
        temp_dict[new_key] = 1 / value
        new_key = key[:idx] + "2" + key[:idx]
        temp_dict[new_key] = 1
    ratio_dict.update(temp_dict)


ratio_update(ratio_length)
ratio_update(ratio_force)
ratio_update(ratio_time)

unit_length: TypeAlias = Literal["inch", "ft", "mm", "cm", "m", "km"]
unit_force: TypeAlias = Literal["lb", "lbf", "kip", "N", "kN", "MN", "kgf", "tonf"]
unit_time: TypeAlias = Literal["msec", "sec", "min", "hour", "day", "year"]
unit_mass: TypeAlias = Literal["mg", "g", "kg", "ton", "t", "slug"]
unit_stress: TypeAlias = Literal["Pa", "kPa", "MPa", "GPa", "bar", "psi", "ksi", "psf", "ksf"]


class UnitSystem:
    """A class for unit conversion.

    Parameters
    -----------
    length: str, default="m"
        Length unit base. Optional ["inch", "ft", "mm", "cm", "m", "km"].
    force: str, default="kN"
        Force unit base. Optional ["lb"("lbf"), "kip", "n", "kn", "mn", "kgf", "tonf"].
    time: str, default="sec"
        Time unit base. Optional ["sec"].

    .. note::
        * `Mass` and `stress` units can be automatically determined based on `length` and `force` units,
          optional mass units include ["mg", "g", "kg", "ton"("t"), "slug"],
          and optional stress units include ["pa", "kpa", "mpa", "gpa", "bar", "psi", "ksi", "psf", "ksf"].

        * You can enter any uppercase and lowercase forms, such as ``kn`` and ``kN``, ``mpa`` and ``MPa``
          are equivalent.

        * You can add a number (int) after the unit to indicate a power, such as ``m3`` for ``m*m*m``.
    """

    def __init__(self, length: unit_length = "m", force: unit_force = "kn", time: unit_time = "sec") -> None:
        self._length = length.lower()
        self._force = force.lower()
        self._time = time.lower()
        for unit in get_args(unit_length):
            setattr(self, unit, ratio_length[unit.lower() + "2" + self._length])
        for unit in get_args(unit_force):
            setattr(self, unit, ratio_force[unit.lower() + "2" + self._force])
        for unit in get_args(unit_time):
            setattr(self, unit, ratio_time[unit.lower() + "2" + self._time])
        # mass
        self.kg = self.N / self.m
        self.mg, self.g, self.ton = 1e-6 * self.kg, 1e-3 * self.kg, 1e3 * self.kg
        self.t, self.slug, self.slinch = (
            1e3 * self.kg,
            14.593902937 * self.kg,
            175.126836 * self.kg,
        )
        # stress
        self.pa = self.N / (self.m * self.m)
        self.kpa, self.mpa, self.gpa = 1000 * self.pa, 1e6 * self.pa, 1e9 * self.pa
        self.bar = 1e5 * self.pa
        self.psi, self.ksi = 6894.7572932 * self.pa, 6894757.2932 * self.pa
        self.psf, self.ksf = 47.880208 * self.pa, 47880.2468616010 * self.pa

    @property
    def length(self):
        return self._length

    @property
    def force(self):
        return self._force

    @property
    def time(self):
        return self._time

    # def __getattr__(self, item):
    #     v = re.findall(r"-?\d+\.?\d*e?E?-?\d*?", item)
    #     if v:
    #         v = float(v[0])
    #     else:
    #         return getattr(self, item.lower())
    #     s = "".join([x for x in item if x.isalpha()])
    #     base = getattr(self, s)
    #     return base**v

    def __getattr__(self, name: str):
        # Uniformly convert to lowercase (preserve the numeric part)
        base_name = "".join([c for c in name if not c.isdigit()]).lower()
        valid_units = (
            get_args(unit_length)
            + get_args(unit_force)
            + get_args(unit_time)
            + get_args(unit_mass)
            + get_args(unit_stress)
        )
        if base_name in [u.lower() for u in valid_units]:
            return self.get_unit_ratio(name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")  # noqa: TRY003

    def get_unit_ratio(self, name: str) -> float:
        # Normalize the input name
        clean_name = re.sub(r"[^a-zA-Z0-9]", "", name).lower()

        # Separate the unit base name and exponent
        match = re.fullmatch(r"([a-z]+)(\d*)", clean_name)
        if not match:
            raise AttributeError(f"Invalid unit format: '{name}'")  # noqa: TRY003
        base_part, exponent_part = match.groups()

        # Build a dictionary for attribute lookup (lowercase keys)
        attr_map = {k.lower(): v for k, v in self.__dict__.items() if isinstance(v, (int, float))}

        if base_part.lower() in attr_map:
            exponent = int(exponent_part) if exponent_part else 1
            base_value = attr_map[base_part.lower()]
            return base_value**exponent
        else:
            suggestions = difflib.get_close_matches(base_part, attr_map.keys(), n=3)
            err_msg = f"'{self.__class__.__name__}' has no attribute '{name}'"
            if suggestions:
                err_msg += f". Did you mean: {', '.join(suggestions)}?"
            raise AttributeError(err_msg)

    def __repr__(self) -> str:
        return f"<UnitSystem: length={self.length!r}, force={self.force!r}, time={self.time!r} ({hash(self)})>"

    def print(self):
        """Show all unit conversion coefficients with colorful output"""
        from rich import print as rprint

        txt = "\n[bold #d20962]Length unit:[/bold #d20962]\n"
        for _i, unit in enumerate(get_args(unit_length)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        txt += "\n\n[bold #f47721]Force unit:[/bold #f47721]\n"
        for _i, unit in enumerate(get_args(unit_force)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        txt += "\n\n[bold #7ac143]Time unit:[/bold #7ac143]\n"
        for _i, unit in enumerate(get_args(unit_time)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        txt += "\n\n[bold #00bce4]Mass unit:[/bold #00bce4]\n"
        for _i, unit in enumerate(get_args(unit_mass)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        txt += "\n\n[bold #7d3f98]Pressure unit:[/bold #7d3f98]\n"
        for _i, unit in enumerate(get_args(unit_stress)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        rprint(txt)


if __name__ == "__main__":
    UNIT = UnitSystem(length="m", force="kN", time="min")
    # Call the __repr__ method, print the UnitSystem object information
    print(UNIT)
    # Call the print method, print all common units
    UNIT.print()

    # Show some unit conversion effects
    print("Length:", UNIT.mm, UNIT.Mm2, UNIT.cm, UNIT.m, UNIT.M2, UNIT.inch, UNIT.Ft)
    print("Force", UNIT.n, UNIT.kN, UNIT.kN2, UNIT.lbf, UNIT.kip)
    print("Stress", UNIT.mpa, UNIT.kpa, UNIT.pa, UNIT.psi, UNIT.ksi)
    print("Mass", UNIT.g, UNIT.kg, UNIT.ton, UNIT.slug)
    print("Time", UNIT.msec, UNIT.min, UNIT.hour, UNIT.day, UNIT.year)

    # When inputting invalid unit, it will give smart suggestions
    print(UNIT.mmm)
