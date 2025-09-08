import difflib
import re
from typing import Literal, TypeAlias, get_args

_TOKEN = re.compile(
    r"""
    (?P<op>[*/])?                 # operator
    (?P<unit>[A-Za-z]+)           # unit symbol
    (?:\^?(?P<pow>[-+]?\d+))?     # ^exponent
    """,
    re.VERBOSE,
)

_ratio_length = {
    "inch2m": 0.0254,  # exact
    "inch2dm": 0.254,  # exact
    "inch2cm": 2.54,  # exact
    "inch2mm": 25.4,  # exact
    "inch2km": 2.54e-5,  # exact
    "inch2ft": 1.0 / 12.0,  # exact (0.0833333333333…)
    "ft2mm": 304.8,  # exact
    "ft2cm": 30.48,  # exact
    "ft2dm": 3.048,  # exact
    "ft2m": 0.3048,  # exact
    "ft2km": 3.048e-4,  # exact
    "mm2cm": 0.1,  # exact
    "mm2dm": 0.01,  # exact
    "mm2m": 0.001,  # exact
    "mm2km": 1e-6,  # exact
    "cm2dm": 0.1,  # exact
    "cm2m": 0.01,  # exact
    "cm2km": 1e-5,  # exact
    "m2km": 1e-3,  # exact
}

_ratio_force = {
    "lb2lbf": 1.0,  # by convention
    "lb2kip": 0.001,  # exact
    "lb2n": 4.4482216152605,  # exact (1 lbf = 4.4482216152605 N)
    "lb2kn": 4.4482216152605e-3,
    "lb2mn": 4.4482216152605e-6,
    "lb2kgf": 0.45359237,  # exact (1 lb = 0.45359237 kg)
    "lb2tonf": 0.00045359237,  # exact
    "lbf2kip": 0.001,
    "lbf2n": 4.4482216152605,
    "lbf2kn": 4.4482216152605e-3,
    "lbf2mn": 4.4482216152605e-6,
    "lbf2kgf": 0.45359237,
    "lbf2tonf": 0.00045359237,
    "kip2n": 4448.2216152605,  # = 1000 x lbf2n
    "kip2kn": 4.4482216152605,  # = kip2n / 1000
    "kip2mn": 0.0044482216152605,  # = kip2n / 1e6
    "kip2kgf": 453.59237,  # = 1000 x lb2kgf
    "kip2tonf": 0.45359237,  # = 1000 x lb2tonf
    "n2kn": 1e-3,  # exact
    "n2mn": 1e-6,
    "n2kgf": 0.101971621297793,  # = 1 / 9.80665
    "n2tonf": 1.01971621297793e-4,  # = 1 / 9.80665 / 1000
    "kn2mn": 1e-3,
    "kn2kgf": 101.971621297793,
    "kn2tonf": 0.101971621297793,
    "mn2kgf": 101971.621297793,
    "mn2tonf": 101.971621297793,
    "kgf2tonf": 0.001,  # exact
}
_ratio_time = {
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
    "day2hour": 24,
    "day2year": 1 / 365,
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


ratio_update(_ratio_length)
ratio_update(_ratio_force)
ratio_update(_ratio_time)

_unit_length: TypeAlias = Literal["inch", "ft", "mm", "cm", "m", "km"]
_unit_force: TypeAlias = Literal["lb", "lbf", "kip", "n", "kn", "mn", "kgf", "tonf"]
_unit_time: TypeAlias = Literal["msec", "sec", "min", "hour", "day", "year"]
_unit_mass: TypeAlias = Literal["mg", "g", "kg", "ton", "t", "slug"]
_unit_stress: TypeAlias = Literal["pa", "kpa", "mpa", "gpa", "bar", "psi", "ksi", "psf", "ksf"]


class UnitSystem:
    """A class for unit conversion. All unit factors are transformed into the base unit system as specified by the following ``length``, ``force``, and ``time`` parameters.

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

        * You can add a number (int) after the unit to indicate a power, such as ``.m3`` for ``m*m*m``.

        * You can use key indexing, such as: unit["m^3"], unit["kN/m^2"], unit["N*sec^2/m"], unit["MPa"]

    Examples
    ---------
    >>> UNIT = UnitSystem(length="m", force="kN", time="min")
    >>> # Call the __repr__ method, print the UnitSystem object information
    >>> print(UNIT)
    >>> # Call the print method, print all common units
    >>> UNIT.print()
    >>> # use key indexing
    >>> print("N/mm2", UNIT["N/mm2"])
    >>> print("N*mm/m^2", UNIT["N*mm/m^2"])
    >>> print("MPa", UNIT["MPa"])
    >>> # Show some unit conversion effects
    >>> print("Length:", UNIT.mm, UNIT.Mm2, UNIT.cm, UNIT.m, UNIT.M2, UNIT.inch, UNIT.Ft)
    >>> print("Force", UNIT.n, UNIT.kN, UNIT.kN2, UNIT.lbf, UNIT.kip)
    >>> print("Stress", UNIT.mpa, UNIT.kpa, UNIT.pa, UNIT.psi, UNIT.ksi)
    >>> print("Mass", UNIT.g, UNIT.kg, UNIT.ton, UNIT.slug)
    >>> print("Time", UNIT.msec, UNIT.min, UNIT.hour, UNIT.day, UNIT.year)
    """

    def __init__(self, length: _unit_length = "m", force: _unit_force = "kn", time: _unit_time = "sec") -> None:
        # cache
        self._cache = {}

        self._length = length.lower()
        self._force = force.lower()
        self._time = time.lower()
        for unit in get_args(_unit_length):
            val = _ratio_length[unit.lower() + "2" + self._length]
            setattr(self, unit, val)
            self._cache[unit] = val
        for unit in get_args(_unit_force):
            val = _ratio_force[unit.lower() + "2" + self._force]
            setattr(self, unit, val)
            self._cache[unit] = val
        for unit in get_args(_unit_time):
            val = _ratio_time[unit.lower() + "2" + self._time]
            setattr(self, unit, val)
            self._cache[unit] = val
        # alias
        self.s = self.sec
        self.ms = self.msec
        self.kips = self.kip
        # mass
        self.kg = self.n * (self.sec**2) / self.m
        self.mg, self.g, self.ton = 1e-6 * self.kg, 1e-3 * self.kg, 1e3 * self.kg
        self.t, self.slug, self.slinch = (
            1e3 * self.kg,
            14.593902937 * self.kg,
            175.126836 * self.kg,
        )
        # stress
        self.pa = self.N / (self.m * self.m)

        # SI multiples
        self.kpa = 1e3 * self.pa
        self.mpa = 1e6 * self.pa
        self.gpa = 1e9 * self.pa
        self.bar = 1e5 * self.pa

        # Imperial & US customary
        self.psi = 6894.757293168 * self.pa  # 1 psi = 6894.757293168 Pa
        self.ksi = 6894757.293168 * self.pa  # 1 ksi = 1000 psi
        self.psf = 47.88025898033584 * self.pa  # 1 psf = 47.88025898033584 Pa
        self.ksf = 47880.25898033584 * self.pa  # 1 ksf = 1000 psf

        # gravity acceleration
        self.g0 = 9.80665 * self.m / (self.sec**2)
        self.grav = self.g0

    @property
    def length(self):
        return self._length

    @property
    def force(self):
        return self._force

    @property
    def time(self):
        return self._time

    def __getitem__(self, expr: str) -> float:
        expr = expr.strip()
        if expr in self._cache:
            return self._cache[expr]
        val = self._parse_expr(expr)
        self._cache[expr] = val
        return val

    def _parse_expr(self, expr: str) -> float:
        """Parses expressions such as: "m^3", "kN/m^2", "N*sec^2/m", "MPa".

        Rules:

        Connect terms with * or /.

        Exponents can be written with ^ (e.g., ^3) or by appending the number directly (e.g., "m3").
        """
        expr = expr.replace(" ", "")

        pos = 0
        total = 1.0
        last_op = "*"

        while pos < len(expr):
            m = _TOKEN.match(expr, pos)
            if not m:
                raise ValueError(f"Bad unit token near: '{expr[pos:]}' in '{expr}'")  # noqa: TRY003
            op = m.group("op") or last_op
            unit = m.group("unit")
            pow_str = m.group("pow")
            pos = m.end()

            # Support trailing numeric exponents without '^', e.g., m3
            if pow_str is None:
                # Attempt to match the immediately following numeric sequence
                tail = re.match(r"(\d+)", expr[pos:])
                if tail:
                    pow_str = tail.group(1)
                    pos += len(pow_str)

            attr_map = {k.lower(): v for k, v in self.__dict__.items() if isinstance(v, (int, float))}
            factor = attr_map.get(unit.lower(), None)
            if factor is None:
                raise ValueError(f"Unknown unit: '{unit}' in '{expr}'")  # noqa: TRY003
            power = int(pow_str) if pow_str else 1
            factor = factor**power

            if op == "*":
                total *= factor
            elif op == "/":
                total /= factor
            else:
                raise ValueError(f"Unknown operator '{op}' in '{expr}'")  # noqa: TRY003

            last_op = "*"

        return total

    # def __getattr__(self, item):
    #     v = re.findall(r"-?\d+\.?\d*e?E?-?\d*?", item)
    #     if v:
    #         v = float(v[0])
    #     else:
    #         return getattr(self, item.lower())
    #     s = "".join([x for x in item if x.isalpha()])
    #     base = getattr(self, s)
    #     return base**v

    def _get_cache(self):
        """安全地获取缓存字典，避免递归调用"""
        return object.__getattribute__(self, '_cache')

    def __getattr__(self, expr: str):
        cache = self._get_cache()
        expr = expr.strip()
        if expr in cache:
            return cache[expr]
        # Uniformly convert to lowercase (preserve the numeric part)
        base_name = "".join([c for c in expr if not c.isdigit()]).lower()
        valid_units = (
            get_args(_unit_length)
            + get_args(_unit_force)
            + get_args(_unit_time)
            + get_args(_unit_mass)
            + get_args(_unit_stress)
        )
        if base_name in [u.lower() for u in valid_units]:
            val = self._get_unit_ratio(expr)
            self._cache[expr] = val
            return val
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{expr}'")  # noqa: TRY003

    def _get_unit_ratio(self, name: str) -> float:
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
        for _i, unit in enumerate(get_args(_unit_length)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        txt += "\n\n[bold #f47721]Force unit:[/bold #f47721]\n"
        for _i, unit in enumerate(get_args(_unit_force)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        txt += "\n\n[bold #7ac143]Time unit:[/bold #7ac143]\n"
        for _i, unit in enumerate(get_args(_unit_time)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        txt += "\n\n[bold #00bce4]Mass unit:[/bold #00bce4]\n"
        for _i, unit in enumerate(get_args(_unit_mass)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        txt += "\n\n[bold #7d3f98]Pressure unit:[/bold #7d3f98]\n"
        for _i, unit in enumerate(get_args(_unit_stress)):
            txt += f"{unit}={getattr(self, unit):.3g}; "
        rprint(txt)


if __name__ == "__main__":
    UNIT = UnitSystem(length="m", force="kN", time="min")
    # Call the __repr__ method, print the UnitSystem object information
    print(UNIT)
    # Call the print method, print all common units
    UNIT.print()
    print("N/mm2", UNIT["N/mm2"])
    print("N*mm/mm^2", UNIT["N*mm/mm^2"])  # Example of using __getitem__ to get a unit conversion value

    # Show some unit conversion effects
    print("Length:", UNIT.mm, UNIT.Mm2, UNIT.cm, UNIT.m, UNIT.M2, UNIT.inch, UNIT.Ft)
    print("Force", UNIT.n, UNIT.kN, UNIT.kN2, UNIT.lbf, UNIT.kip)
    print("Stress", UNIT.mpa, UNIT.kpa, UNIT.pa, UNIT.psi, UNIT.ksi)
    print("Mass", UNIT.g, UNIT.kg, UNIT.ton, UNIT.slug)
    print("Time", UNIT.msec, UNIT.min, UNIT.hour, UNIT.day, UNIT.year)

    UNIT = UnitSystem(length="inch", force="kip", time="sec")
    # Call the __repr__ method, print the UnitSystem object information
    print(UNIT)
    # Call the print method, print all common units
    UNIT.print()

    print("MPa", UNIT["MPa"])  # Example of using __getitem__ to get a unit conversion value
    print("kip*sec^2/inch", UNIT["kip*sec^2/inch"])

    # Show some unit conversion effects
    print("Length:", UNIT.mm, UNIT.Mm2, UNIT.cm, UNIT.m, UNIT.M2, UNIT.inch, UNIT.Ft)
    print("Force", UNIT.n, UNIT.kN, UNIT.kN2, UNIT.lbf, UNIT.kip)
    print("Stress", UNIT.mpa, UNIT.kpa, UNIT.pa, UNIT.psi, UNIT.ksi)
    print("Mass", UNIT.g, UNIT.kg, UNIT.ton, UNIT.slug)

    # When inputting invalid unit, it will give smart suggestions
    print(UNIT.mmm)
