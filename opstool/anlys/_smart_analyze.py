import time
from contextlib import contextmanager
from typing import Optional, Union

import numpy as np
import openseespy.opensees as ops
from rich import print  # noqa: A004
from rich.console import RenderableType
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
)

from ..utils import get_random_color

LOG_FILE = ".SmartAnalyze-OpenSees.log"


@contextmanager
def suppress_ops_print(verbose=False):
    if not verbose:
        ops.logFile(LOG_FILE, "-noEcho")
    # else:
    #     ops.logFile(LOG_FILE)
    yield


class HHMMSSMSColumn(ProgressColumn):
    def render(self, task) -> RenderableType:
        t = task.elapsed or 0.0
        total_ms = int(t * 1000)
        hours = total_ms // (3600 * 1000)
        minutes = (total_ms // (60 * 1000)) % 60
        seconds = (total_ms // 1000) % 60
        millis = total_ms % 1000
        return f"[#037ef3]{hours:02} h : [#f85a40]{minutes:02} m : [#00c16e]{seconds:02} s : [#7552cc]{millis:03} ms"


class SmartAnalyze:
    """The SmartAnalyze is a class to provide OpenSeesPy users an easier
    way to conduct analyses.
    Original Tcl version Author: Dr. Dong Hanlin, see
    `here <https://github.com/Hanlin-Dong/SmartAnalyze/>`__.
    Here's the converted python version, with some modifications.

    Parameters
    ---------------------
    analysis_type: str, default="Transient"
        Assign the analysis type, "Transient" or "Static".

    Other Parameters that control convergence
    ----------------------------------------------

    TEST RELATED:
    ===============
    testType: str, default="EnergyIncr"
        Identical to the testType in OpenSees test command.
        Choices see `test command <https://opensees.berkeley.edu/wiki/index.php/Test_Command>`__
    testTol: float, default=1.0e-10
        The initial test tolerance set to the OpenSees test command.
        If tryLooseTestTol is set to True, the test tolerance can be loosened.
    testIterTimes: int, default=10
        The initial number of test iteration times.
        If tryAddTestTimes is set to True, the number of test times can be enlarged.
    testPrintFlag: int, default=0
        The test print flag in OpenSees ``test`` command.
    tryAddTestTimes: bool, default=False
        If True, the number of test times will be enlarged if the last test norm is smaller than `normTol`,
        the enlarged number is specified in `testIterTimesMore`.
        Otherwise, the number of test times will always be equal to `testIterTimes`.
    normTol: float, default=1.e3
        Only useful when tryAddTestTimes is True.
        If unconverged, the last norm of test will be compared to `normTol`.
        If the norm is smaller, the number of test times will be enlarged.
    testIterTimesMore: int or list, default=[50]
        Only useful when tryaddTestTimes is True.
        If unconverge and norm are ok, the test iteration times will be set to this number.
    tryLooseTestTol: bool, default=False
        If this is set to True, if unconverge at a minimum step,
        the test tolerance will be loosened to the number specified by `looseTestTolTo`.
        The step will be set back.
    looseTestTolTo: float, default= 100 * initial test tolerance
        Only useful if tryLooseTestTol is True.
        If unconvergance at the min step, the test tolerance will be set to this value.

    ALGORITHM RELATED:
    ===================
    tryAlterAlgoTypes: bool, default=False
        If True, different algorithm types specified
        in `algoTypes` will be tried during unconvergance.
        If False, the first algorithm type specified in `algoTypes`
        will be used.
    algoTypes: list[int], default=[40, 10, 20, 30, 50, 60, 70, 90]
        A list of flags of the algorithms to be used during unconvergance.
        The integer flag is documented in the following section.
        Only useful when tryAlterAlgoTypes is True.
        The first flag will be used by default when tryAlterAlgoTypes is False.
        The algorithm command in the model will be ignored.
        If you need another algorithm, try a user-defined algorithm. See the following section.
    UserAlgoArgs: list,
        User-defined algorithm parameters, 100 is required in algoTypes,
        and the parameters must be included in the list, for example:
        algoTypes = [10, 20, 100],
        UserAlgoArgs = ["KrylovNewton", "-iterate", "initial", "-maxDim", 20]

    **Algorithm type flag reference**

    .. list-table:: Algorithm type flag reference
       :widths: 10 20
       :header-rows: 1

       * - Flags
         - Algorithm
       * - 0
         - Linear
       * - 1
         - Linear -initial
       * - 2
         - Linear -secant
       * - 3
         - Linear -factorOnce
       * - 4
         - Linear -initial -factorOnce
       * - 5
         - Linear -secant -factorOnce
       * - 10
         - Newton
       * - 11
         - Newton -initial
       * - 12
         - Newton -initialThenCurrent
       * - 13
         - Newton -Secant
       * - 20
         - NewtonLineSearch
       * - 21
         - NewtonLineSearch -type Bisection
       * - 22
         - NewtonLineSearch -type Secant
       * - 23
         - NewtonLineSearch -type RegulaFalsi
       * - 24
         - NewtonLineSearch -type LinearInterpolated
       * - 25
         - NewtonLineSearch -type InitialInterpolated
       * - 30
         - ModifiedNewton
       * - 31
         - ModifiedNewton -initial
       * - 32
         - ModifiedNewton -secant
       * - 40
         - KrylovNewton
       * - 41
         - KrylovNewton -iterate initial
       * - 42
         - KrylovNewton -increment initial
       * - 43
         - KrylovNewton -iterate initial -increment initial
       * - 44
         - KrylovNewton -maxDim 10
       * - 45
         - KrylovNewton -iterate initial -increment initial -maxDim 10
       * - 50
         - SecantNewton
       * - 51
         - SecantNewton -iterate initial
       * - 52
         - SecantNewton -increment initial
       * - 53
         - SecantNewton -iterate initial -increment initial
       * - 60
         - BFGS
       * - 61
         - BFGS -initial
       * - 62
         - BFGS -secant
       * - 70
         - Broyden
       * - 71
         - Broyden -initial
       * - 72
         - Broyden -secant
       * - 80
         - PeriodicNewton
       * - 81
         -  PeriodicNewton -maxDim 10
       * - 90
         -  ExpressNewton
       * - 91
         - ExpressNewton -InitialTangent
       * - 100
         - User-defined0

    STEP SIZE RELATED:
    ===================
    initialStep: float, default=None
        Specifying the initial Step length to conduct analysis.
        If None, equal to `dt`.
    relaxation: float, between 0 and 1, default=0.5
        A factor that is multiplied by each time
        the step length is shortened.
    minStep: float, default=1.e-6
        The step tolerance when shortening the step length.
        If step length is smaller than minStep, special ways to converge the model will be used
        according to `try-` flags.

    LOGGING RELATED:
    ===================
    debugMode: bool, default=False
        If True, print as much information as possible.
        If False, the progress bar will be used.
        If False, a log file named '.SmartAnalyze-OpenSees.log'
        will be generated to store the information printed by OpenSees.
    printPer: int, default=50
        Print to the console every several trials.
        This is only useful when debugMode = True.

    Examples
    ---------
    The following example demonstrates how to use the SmartAnalyze class.

    .. Note::

        * ``test()`` and ``algorithm()`` will run automatically in ``SmartAnalyze``;
        * Static analysis only supports displacement control;
        * Commands such as ``integrator()`` must be defined outside ``SmartAnalyze`` for ransient analysis.

    Example 1: Basic usage for Transient

    >>> import opstool as opst
    >>> ops.constraints('Transformation')
    >>> ops.numberer('Plain')
    >>> ops.system('BandGeneral')
    >>> ops.integrator('Newmark', 0.5, 0.25)  # Dynamic analysis requires external settings
    >>> analysis = opst.anlys.SmartAnalyze(analysis_type="Transient")
    >>> npts, dt = 1000, 0.01
    >>> # Tells the program the total number of steps, which is necessary for outputting a progress bar
    >>> segs = analysis.transient_split(npts)
    >>> for _ in segs:
    >>>     analysis.TransientAnalyze(dt)

    Example 2: Basic usage for Static

    >>> import opstool as opst
    >>> ops.constraints('Transformation')
    >>> ops.numberer('Plain')
    >>> ops.system('BandGeneral')
    >>> protocol=[0.5, -0.5, 1, -1, 0]  # Load Profile
    >>> analysis = opst.anlys.SmartAnalyze(analysis_type="Static")
    >>> segs = analysis.static_split(protocol, 0.01)  # Use a step size of 0.01 to segment the profile.
    >>> print(segs)
    >>> for seg in segs:
    >>>     analysis.StaticAnalyze(node=1, dof=2, seg=seg)  # node tag 1, dof 2

    Example 3: change control parameters

    >>> analysis = opst.anlys.SmartAnalyze(
    >>>    analysis_type="Transient",
    >>>    tryAlterAlgoTypes=True,
    >>>    algoTypes=[40, 30, 20],
    >>>    tryAddTestTimes=True,
    >>>    testIterTimesMore=[50, 100],
    >>>    relaxation=0.5,
    >>>    minStep=1e-5,
    >>>    printPer=20,
    >>>)
    """

    def __init__(self, analysis_type="Transient", **kargs):
        if analysis_type not in ("Transient", "Static"):
            raise ValueError("analysis_type must Transient or Static!")  # noqa: TRY003
        # default
        self.control_args = {
            "analysis": analysis_type,
            "testType": "EnergyIncr",
            "testTol": 1.0e-10,
            "testIterTimes": 10,
            "testPrintFlag": 0,
            "tryAddTestTimes": False,
            "normTol": 1000,
            "testIterTimesMore": [50],
            "tryLooseTestTol": False,
            "looseTestTolTo": 1e-3,
            "tryAlterAlgoTypes": False,
            "algoTypes": [40, 10, 20, 30, 50, 60, 70, 90],
            "UserAlgoArgs": None,
            "initialStep": None,
            "relaxation": 0.5,
            "minStep": 1.0e-6,
            "debugMode": False,
            "printPer": 20,
        }
        self.control_args["looseTestTolTo"] = 100 * self.control_args["testTol"]
        for name in kargs:
            if name not in self.control_args:
                raise ValueError(f"Arg {name} error, valid args are: {self.control_args.keys()}!")  # noqa: TRY003
        self.control_args.update(kargs)

        self.analysis_type = analysis_type
        self.eps = 1.0e-12
        self.logo = "[bold magenta]SmartAnalyze:[/bold magenta]"
        self.logo_progress = "[bold magenta]SmartAnalyze"
        self.logo_analysis_type = f"[bold cerulean]{self.analysis_type}"

        self.debug_mode = self.control_args["debugMode"]

        # initial test commands
        self._set_init_test()
        # initial algorithm
        self._setAlgorithm(
            self.control_args["algoTypes"][0], self.control_args["UserAlgoArgs"], verbose=self.debug_mode
        )

        # Since the intelligent static analysis may reset the integrator,
        # the sensitivity analysis algorithm needs to be reset
        self.sensitivity_algorithm = None

        self.current_args = {
            "startTime": time.time(),
            "counter": 0,
            "progress": 0,
            "npts": 0,
            "step": 0.0,
            "node": 0,
            "dof": 0,
        }

        self.progress = None
        self.task = None

    def _set_progress_bar(self, npts):
        self.progress = Progress(
            # TextColumn(f"{self.logo_progress} • {{task.description}}"),
            TextColumn(f":rocket: {self.logo_progress}"),
            BarColumn(
                bar_width=40,
                style="#44475a",
                complete_style="#ff79c6",
                finished_style="#6fc276",
            ),
            TextColumn("{task.percentage:>3.0f}%", style="bold #6a79f7"),
            TextColumn(":hourglass:"),
            HHMMSSMSColumn(),
        )
        self.progress.start()
        self.task = self.progress.add_task(self.logo_analysis_type, total=npts)

    def _stop_progress_bar(self):
        if self.progress is not None:
            self.progress.update(self.task, completed=self.current_args["progress"])
            time.sleep(0.1)  # flush
            self.progress.stop()
        self.progress = None
        self.task = None

    def transient_split(self, npts: int):
        """Step Segmentation for Transient Analysis.
        The main purpose of this function is to tell the program the total number of analysis steps to show progress.
        However, this is not necessary.

        Parameters
        ----------
        npts : int
            Total steps for transient analysis.

        Returns
        -------
        A list to loop.
        """
        self.current_args["npts"] = npts
        if not self.debug_mode and self.progress is None:
            self._set_progress_bar(npts)
        return list(range(1, npts + 1))

    def static_split(self, targets: Union[list, tuple, np.ndarray], maxStep: Optional[float] = None):
        """
        Returns a sequence of substeps for static analysis.

        Parameters
        ----------
        targets : list, tuple, or np.ndarray
            Target displacements. First element should be ≥ 0.
        maxStep : float, optional
            Maximum step size. If None, uses difference between first two targets.

        Returns
        -------
        segs : list of float
            Sequence of substeps to reach each target displacement.
        """
        targets = np.atleast_1d(targets).astype(float)
        if targets.ndim != 1:
            raise ValueError("targets must be 1D!")  # noqa: TRY003
        if len(targets) == 1 and maxStep is None:
            raise ValueError("When only one target is given, maxStep must be specified!")  # noqa: TRY003
        if targets[0] != 0.0:
            targets = np.insert(targets, 0, 0.0)
        if maxStep is None:
            maxStep = targets[1] - targets[0]

        segs = []
        for start, end in zip(targets[:-1], targets[1:]):
            delta = end - start
            if abs(delta) < self.eps:
                continue
            direction = np.sign(delta)
            num_steps = int(abs(delta) // maxStep)
            remainder = abs(delta) - num_steps * maxStep
            segs.extend([direction * maxStep] * num_steps)
            if remainder > self.eps:
                segs.append(direction * remainder)

        self.current_args["npts"] = len(segs)
        if not self.debug_mode and self.progress is None:
            self._set_progress_bar(len(segs))
        return segs

    def _get_time(self):
        return time.time() - self.current_args["startTime"]

    def set_sensitivity_algorithm(self, algorithm: str = "-computeAtEachStep"):
        """Set analysis sensitivity algorithm. Since the Smart Static Analysis may reset the integrator,
        the sensitivity analysis algorithm will need to be reset afterwards.

        Parameters
        -----------
        algorithm: Sensitivity analysis algorithm, default: "-computeAtEachStep".
            Optional: "-computeAtEachStep" or "-computeByCommand".

        Return
        -------
        None
        """
        if algorithm not in ["-computeAtEachStep", "-computeByCommand"]:
            raise ValueError("algorithm must be '-computeAtEachStep' or '-computeByCommand'")  # noqa: TRY003
        self.sensitivity_algorithm = algorithm

    def _run_sensitivity_algorithm(self):
        if self.sensitivity_algorithm is not None:
            ops.sensitivityAlgorithm(self.sensitivity_algorithm)

    def TransientAnalyze(self, dt: float):
        """Single Step Transient Analysis.

        Parameters
        ----------
        dt : float
            Time Step.

        Returns
        -------
        Return 0 if successful, otherwise returns a negative number.
        """
        if self.control_args["analysis"] != "Transient":
            raise ValueError("Transient! Please check parameter input!")  # noqa: TRY003
        self.control_args["initialStep"] = dt

        ops.analysis(self.control_args["analysis"])

        return self._analyze()

    def StaticAnalyze(self, node: int, dof: int, seg: float):
        """Single step static analysis and applies to displacement control only.

        Parameters
        ----------
        node : int
            The node tag in the displacement control.
        dof : int
            The dof in the displacement control.
        seg : float
            Each load step, i.e., each element returned by static_split.

        Returns
        -------
        Return 0 if successful, otherwise returns a negative number.
        """
        if self.control_args["analysis"] != "Static":
            raise ValueError("Static! Please check parameter input!")  # noqa: TRY003
        self.control_args["initialStep"] = seg
        self.current_args["node"] = node
        self.current_args["dof"] = dof
        self.current_args["step"] = seg

        ops.integrator("DisplacementControl", node, dof, seg)
        ops.analysis(self.control_args["analysis"])

        # reset sensitivity analysis algorithm
        self._run_sensitivity_algorithm()

        return self._analyze()

    def _analyze(self):
        initial_step = self.control_args["initialStep"]
        verbose = bool(self.debug_mode)

        ok = self._analyze_one_step(initial_step, verbose=verbose)

        retry_strategies = [
            self._try_add_test_times,
            self._try_alter_algo_types,
            self._try_relax_step,
            self._try_loose_test_tol,
        ]

        for retry in retry_strategies:
            if ok >= 0:
                break
            ok = retry(initial_step, verbose)

        if ok < 0:
            self._stop_progress_bar()
            self._print_status(success=False)
            return ok

        self.current_args["progress"] += 1
        self.current_args["counter"] += 1

        if verbose and self.current_args["counter"] >= self.control_args["printPer"]:
            self._print_progress()
            self.current_args["counter"] = 0

        if self.progress is not None:
            self.progress.advance(self.task, advance=1)

        if self.current_args.get("npts", 0) > 0 and self.current_args["progress"] >= self.current_args["npts"]:
            self._stop_progress_bar()
            self._print_status(success=True)

        return 0

    def _print_status(self, success: bool):
        color = get_random_color()
        time_val = f"[bold {color}]{self._get_time():.3f}[/bold {color}]"
        if success:
            print(
                f":tada: {self.logo} [{color}]Successfully finished[/{color}]! Time consumption: {time_val} s. :tada:"
            )
        else:
            print(f":x: {self.logo} Analyze failed. Time consumption: {time_val} s.")

    def _print_progress(self):
        color = get_random_color()
        time_val = f"[bold {color}]{self._get_time():.3f}[/bold {color}]"
        prog = self.current_args["progress"]
        total = self.current_args.get("npts", 0)
        if total > 0:
            percent = f"[bold {color}]{100 * prog / total:.3f}[/bold {color}]"
            print(f">>> ✅ {self.logo} progress {percent} %. Time consumption: {time_val} s.")
        else:
            print(f">>> ✅ {self.logo} progress {prog} steps. Time consumption: {time_val} s.")

    def close(self):
        """Close the class.

        Returns:
            None
        """
        self._stop_progress_bar()

    def _analyze_one_step(self, step: float, verbose):
        if self.analysis_type == "Static":
            ops.integrator("DisplacementControl", self.current_args["node"], self.current_args["dof"], step)

            # reset sensitivity analysis algorithm
            self._run_sensitivity_algorithm()
            with suppress_ops_print(verbose=verbose):
                ok = ops.analyze(1)
        else:
            with suppress_ops_print(verbose=verbose):
                ok = ops.analyze(1, step)

        self.current_args["step"] = step

        return ok

    def _try_add_test_times(self, step, verbose):
        if not self.control_args["tryAddTestTimes"]:
            return -1
        times = self.control_args["testIterTimesMore"]
        if isinstance(times, (int, float)):
            times = [int(times)]

        ok = -1
        for num in times:
            norm = ops.testNorm()
            if norm[-1] < self.control_args["normTol"]:
                if verbose:
                    color = get_random_color()
                    print(f">>> ▶️ {self.logo} Adding test times to [bold {color}]{num}[/bold {color}].")
                ops.test(
                    self.control_args["testType"], self.control_args["testTol"], num, self.control_args["testPrintFlag"]
                )
                ok = self._analyze_one_step(step, verbose=verbose)

                if ok == 0:
                    self._set_init_test()
                    return ok
            else:
                if verbose:
                    color = get_random_color()
                    print(
                        f">>> ▶️ {self.logo} Not adding test times for norm [bold {color}]%.3e[/bold {color}]."
                        % (norm[-1])
                    )
        # goback
        self._set_init_test()
        return ok

    def _try_alter_algo_types(self, step, verbose):
        if not self.control_args["tryAlterAlgoTypes"]:
            return -1

        if len(self.control_args["algoTypes"]) <= 1:
            return -1

        ok = -1
        for algo_flag in self.control_args["algoTypes"][1:]:
            color = get_random_color()
            if verbose:
                print(f">>> ▶️ {self.logo} Setting algorithm to [bold {color}]{algo_flag}[/bold {color}].")
            self._setAlgorithm(algo_flag, self.control_args["UserAlgoArgs"], verbose=self.debug_mode)
            ok = self._analyze_one_step(step, verbose=verbose)
            if ok == 0:
                return ok
        if ok < 0:  # goback
            self._setAlgorithm(
                self.control_args["algoTypes"][0], self.control_args["UserAlgoArgs"], verbose=self.debug_mode
            )
        return ok

    def _try_relax_step(self, step, verbose):
        alpha = self.control_args["relaxation"]
        min_step = self.control_args["minStep"]
        step_try = step * alpha  # The current step size we're trying to use
        step_remaining = step  # How much of the time step is left to complete

        if verbose:
            color = get_random_color()
            print(
                f">>> ▶️ {self.logo} Dividing the current step [bold {color}]{step:.3e}[/bold {color}] "
                f"into [bold {color}]{step_try:.3e}[/bold {color}] and [bold {color}]{step - step_try:.3e}[/bold {color}]"
            )

        ok = -1
        while step_remaining > self.eps:
            if step_try < min_step:
                color = get_random_color()
                print(
                    f">>> ▶️ {self.logo} Current step [bold {color}]%.3e[/bold {color}] beyond the min step!" % step_try
                )
                return -1

            if step_try > step_remaining:
                step_try = step_remaining  # avoid overshooting

            # Try to run one substep
            ok = self._analyze_one_step(step_try, verbose=verbose)

            if ok == 0:
                step_remaining -= step_try
                # Try to increase next step size by relaxing alpha
                step_try = step_remaining
                if verbose:
                    color = get_random_color()
                    print(
                        f">>> ▶️ {self.logo} Current total step size [bold {color}]{step}[/bold {color}], "
                        f"completed sub-step size [bold {color}]{step - step_remaining}[/bold {color}], "
                        f"remaining sub-step size [bold {color}]{step_remaining}[/bold {color}]"
                    )
            else:
                step_try *= alpha
                if verbose:
                    color = get_random_color()
                    print(
                        f">>> ▶️ {self.logo} Dividing the current step [bold {color}]{step_try / alpha:.3e}[/bold {color}] "
                        f"into [bold {color}]{step_try:.3e}[/bold {color}] and "
                        f"[bold {color}]{step_try / alpha - step_try:.3e}[/bold {color}]"
                    )
        return ok

    def _try_loose_test_tol(self, step, verbose):
        if not self.control_args["tryLooseTestTol"]:
            return -1
        if verbose:
            color = get_random_color()
            print(
                f">>> ⚠️ {self.logo} Warning: [bold {color}]Loosing test tolerance to "
                f"{self.control_args['looseTestTolTo']}[/bold {color}]"
            )
        ops.test(
            self.control_args["testType"],
            self.control_args["looseTestTolTo"],
            self.control_args["testIterTimes"],
            self.control_args["testPrintFlag"],
        )
        ok = self._analyze_one_step(step, verbose=verbose)

        # goback whenever
        self._set_init_test()

        return ok

    def _set_init_test(self):
        ops.test(
            self.control_args["testType"],
            self.control_args["testTol"],
            self.control_args["testIterTimes"],
            self.control_args["testPrintFlag"],
        )

    def _setAlgorithm(self, algotype, user_algo_args: Optional[list] = None, verbose=True):
        color = get_random_color()
        prefix = ">>> ▶️"

        def log_and_call(name, *args):
            if verbose:
                arg_str = " ".join(str(a) for a in args)
                print(f"{prefix} {self.logo} Setting algorithm to  [bold {color}]{name} {arg_str}...[/bold {color}]")
            ops.algorithm(name, *args)

        algo_map = {
            0: ("Linear",),
            1: ("Linear", "-Initial"),
            2: ("Linear", "-Secant"),
            3: ("Linear", "-FactorOnce"),
            4: ("Linear", "-Initial", "-FactorOnce"),
            5: ("Linear", "-Secant", "-FactorOnce"),
            10: ("Newton",),
            11: ("Newton", "-Initial"),
            12: ("Newton", "-intialThenCurrent"),
            13: ("Newton", "-Secant"),
            20: ("NewtonLineSearch",),
            21: ("NewtonLineSearch", "-type", "Bisection"),
            22: ("NewtonLineSearch", "-type", "Secant"),
            23: ("NewtonLineSearch", "-type", "RegulaFalsi"),
            24: ("NewtonLineSearch", "-type", "LinearInterpolated"),
            25: ("NewtonLineSearch", "-type", "InitialInterpolated"),
            30: ("ModifiedNewton",),
            31: ("ModifiedNewton", "-initial"),
            32: ("ModifiedNewton", "-secant"),
            40: ("KrylovNewton",),
            41: ("KrylovNewton", "-iterate", "initial"),
            42: ("KrylovNewton", "-increment", "initial"),
            43: ("KrylovNewton", "-iterate", "initial", "-increment", "initial"),
            44: ("KrylovNewton", "-maxDim", 10),
            45: ("KrylovNewton", "-iterate", "initial", "-increment", "initial", "-maxDim", 10),
            50: ("SecantNewton",),
            51: ("SecantNewton", "-iterate", "initial"),
            52: ("SecantNewton", "-increment", "initial"),
            53: ("SecantNewton", "-iterate", "initial", "-increment", "initial"),
            60: ("BFGS",),
            61: ("BFGS", "-initial"),
            62: ("BFGS", "-secant"),
            70: ("Broyden",),
            71: ("Broyden", "-initial"),
            72: ("Broyden", "-secant"),
            80: ("PeriodicNewton",),
            81: ("PeriodicNewton", "-maxDim", 10),
            90: ("ExpressNewton",),
            91: ("ExpressNewton", "-InitialTangent"),
            100: user_algo_args,  # handled separately
        }

        if algotype == 100:
            if user_algo_args is None:
                raise ValueError("User algorithm args must be provided for type 100")  # noqa: TRY003
            if verbose:
                print(
                    f"{prefix} {self.logo} Setting algorithm to User Algorithm: [bold {color}]{user_algo_args}...[/bold {color}]"
                )
            ops.algorithm(*user_algo_args)
        elif algotype in algo_map:
            args = algo_map[algotype]
            log_and_call(*args)
        else:
            raise ValueError(">>> :warning: SmartAnalyze: ERROR! WRONG Algorithm Type!")  # noqa: TRY003
