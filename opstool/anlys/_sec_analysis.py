from typing import Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import xarray as xr
from scipy.integrate import trapezoid

from ._smart_analyze import SmartAnalyze


class MomentCurvature:
    """Moment-Curvature Analysis for Fiber Section in OpenSeesPy.

    Parameters
    ----------
    sec_tag : int,
        The previously defined section Tag.
    axial_force : float, optional
        Axial load, compression is negative, by default 0
    """

    def __init__(self, sec_tag: int, axial_force: float = 0) -> None:
        self.P = axial_force
        self.sec_tag = sec_tag
        self.phi, self.M, self.FiberData = None, None, None
        self.cycle_path = None
        # self.phi_cycle, self.M_cycle, self.FiberDataCycle = None, None, None

    def analyze(
        self,
        axis: str = "y",
        max_phi: float = 0.5,
        incr_phi: float = 1e-4,
        limit_peak_ratio: float = 0.8,
        cycle_analyze: bool = False,
        smart_analyze: bool = True,
        debug: bool = False,
    ):
        """Performing Moment-Curvature Analysis.

        Parameters
        ----------
        axis : str, optional, "y" or "z"
            The axis of the section to be analyzed, by default "y".
        max_phi : float, optional
            The maximum curvature to analyze, by default 0.5.
        incr_phi : float, optional
            Curvature analysis increment, by default 1e-4.
        limit_peak_ratio : float, optional
            A ratio of the moment intensity after the peak used to stop the analysis., by default 0.8,
            i.e., a 20% drop after peak.
        cycle_analyze : bool, optional
            Whether to perform cyclic analysis, by default False.
        smart_analyze : bool, optional
            Whether to use smart analysis options, by default True.
        debug: bool, optional
            Whether to use debug mode when smart analysis is True, by default, False.

        .. Note::
            The termination of the analysis depends on whichever reaches `max_phi` or `post_peak_ratio` first.

        """
        self.phi, self.M, self.FiberData = _analyze(
            sec_tag=self.sec_tag,
            P=self.P,
            axis=axis,
            max_phi=max_phi,
            incr_phi=incr_phi,
            stop_ratio=limit_peak_ratio,
            cycle=cycle_analyze,
            cycle_path=self.cycle_path,
            smart_analyze=smart_analyze,
            debug=debug,
        )
        print("MomentCurvature: 🎉 Successfully finished! 🎉")

    def set_cycle_path(self, max_phi: float, n_cycle: int = 20, n_hold: int = 1):
        """set a deformation cycle path.

        Parameters
        ----------
        max_phi : float
            Peak of the path.
        n_cycle : int, optional
            Number of cycles, by default 20
        n_hold : int, optional
            The number of repetitions for each cycle., by default 1

        .. Note::
            The total number of cycles is n_cycle * n_hold.

        Returns
        -------
        1D Arraylike.
            Displacement path sequence
        """
        max_phi = abs(max_phi)
        upper_envelope = np.linspace(0, max_phi, n_cycle)[1:]
        below_envelope = np.linspace(0, -max_phi, n_cycle)[1:]
        pattern = [0.0]
        for upper, below in zip(upper_envelope, below_envelope):
            upper, below = float(upper), float(below)
            for _ in range(n_hold):
                pattern.extend([upper, below])
        pattern.append(0.0)
        # # mesh by step size
        # data = [0.0]
        # for i in range(len(pattern) - 1):
        #     a, b = pattern[i], pattern[i + 1]
        #     n = int(np.abs(b - a) / step_size)
        #     n = 2 if n < 2 else n
        #     data.extend(np.linspace(a, b, n)[1:])
        self.cycle_path = pattern
        return pattern

    def plot_M_phi(self, ax=None):
        """Plot the moment-curvature relationship.

        Parameters
        ------------
        ax : matplotlib.axes.Axes, optional
            The axes to plot the moment-curvature relationship, by default None.

        """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 10 * 0.618))
        ax.plot(self.phi, self.M, lw=3, c="blue")
        ax.set_title(
            "$M-\\phi$",
            fontsize=28,
        )
        ax.set_xlabel("$\\phi$", fontsize=25)
        ax.set_ylabel("$M$", fontsize=25)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        for loc in ["bottom", "left", "right", "top"]:
            ax.spines[loc].set_linewidth(1.0)
        plt.gcf().subplots_adjust(bottom=0.15)

    def plot_fiber_responses(self, return_ax: bool = False):
        """Plot the fiber responses.

        Parameters
        -----------
        return_ax: bool, default=False
            If True, return the axes for the plot of matplotlib.
        """

        fiber_data = self.FiberData
        matTags = np.unique(fiber_data[-1][:, 3])

        _, axs = plt.subplots(len(matTags), 1, figsize=(8, 3 * len(matTags)))
        for mat, ax in zip(matTags, axs):
            idxs = np.argwhere(np.abs(fiber_data[-1][:, 3] - mat) < 1e-6)
            strain = fiber_data[:, idxs, -1]
            stress = fiber_data[:, idxs, -2]
            for i in range(len(idxs)):
                ax.plot(
                    strain[:, i, :],
                    stress[:, i, :],
                    lw=1,
                )
            ax.set_title(f"matTag = {mat:.0f}", fontsize=15)
            ax.tick_params(labelsize=12)
            ax.set_ylabel("stress", fontsize=16)
            ax.set_xlabel("strain", fontsize=16)
        plt.subplots_adjust(wspace=0.02, hspace=0.4)
        plt.tight_layout()
        if return_ax:
            return axs

    def get_phi(self):
        """Get the curvature array.

        Returns
        -------
        phi: 1D array-like
            Curvature array.
        """
        return self.phi

    def get_curvature(self):
        """Get the curvature array.

        Returns
        -------
        phi: 1D array-like
            Curvature array.
        """
        return self.get_phi()

    def get_M(self):
        """Get the moment array.

        Returns
        -------
        m: 1D array-like
            Moment array.
        """
        return self.M

    def get_moment(self):
        """Get the moment array.

        Returns
        -------
        m: 1D array-like
            Moment array.
        """
        return self.get_M()

    def get_M_phi(self):
        """Get the moment and curvature array.

        Returns
        -------
        (1D array-like, 1D array-like)
            (Curvature array, Moment array)
        """
        return self.phi, self.M

    def get_fiber_data(self) -> xr.DataArray:
        """All fiber data.

        Returns
        -------
        FiberData: xr.DataArray
            All fiber data.
            "Steps" is the number of steps in the analysis.
            "Fibers" is the number of fibers in the section.
            "Properties" is the properties of the fibers, including "yloc", "zloc", "area", "mat", "stress", "strain".
        """
        data = xr.DataArray(
            self.FiberData,
            coords={
                "Steps": np.arange(len(self.FiberData)),
                "Fibers": np.arange(len(self.FiberData[0])),
                "Properties": ["yloc", "zloc", "area", "mat", "stress", "strain"],
            },
            dims=("Steps", "Fibers", "Properties"),
            name="FiberData",
        )
        return data

    def get_limit_state(
        self,
        matTag: Union[list[int], int] = 1,
        threshold: Union[list[float], float] = 0.0,
        peak_drop: Union[float, bool] = False,
    ):
        """Get the curvature and moment corresponding to a certain limit state.

        Parameters
        ----------
        matTag : Union[list[int], int]
            The OpenSeesPy material Tag used to determine the limit state., by default 1
        threshold : Union[list[float], float]
            The ``strain threshold`` used to determine the limit state by material `matTag`, by default 0.
            The positive and negative signs are meaningful for tension and compression.

        .. Note::
            If ``matTag`` is a list, the length of `matTag` and `threshold` should be the same.
            As long as any material reaches its corresponding threshold, it will be set to the limit state.

        peak_drop : Union[float, bool], optional, by default False.
            If True, A default 20% drop from the peak value of the moment will be used as the limit state;
            If float in [0, 1], this means that the ratio of ultimate strength to peak value is
            specified by this value, for example, peak_drop = 0.3, the ultimate strength = 0.7 * peak.
            `matTag` and `threshold` are not needed.

        .. Note::
            When using ``peak_drop``, matTag and strain threshold will be ignored!

        Returns
        -------
        (float, float)
            (Limit Curvature, Limit Moment)

        Examples
        ---------
        >>> sec = MomentCurvature(sec_tag=1)
        >>> sec.analyze(axis="y", max_phi=1.0, incr_phi=1e-4, limit_peak_ratio=0.8)
        >>> # Get the limit state by material Tag 1 and strain threshold 0.002
        >>> sec.get_limit_state(matTag=1, threshold=0.002)
        >>> sec.get_limit_state(peak_drop=0.20)
        """
        phi = self.phi
        M = self.M
        fiber_data = self.FiberData
        if peak_drop:
            ratio_ = 0.8 if peak_drop is True else 1 - peak_drop
            idx = np.argmax(M)
            au = np.argwhere(M[idx:] <= np.max(M) * ratio_)
            if au.size == 0:
                warn(
                    f"Peak strength does not drop {1 - ratio_}, please increase target ductility ratio! "
                    f"The last value is used as the limit state.",
                    stacklevel=2,
                )
                bu = -1
            else:
                bu = np.min(au) + idx - 1
        else:
            mat_tags = np.atleast_1d(matTag)
            thresholds = np.atleast_1d(threshold)
            if len(mat_tags) != len(thresholds):
                raise ValueError("The length of matTag and threshold should be the same!")  # noqa: TRY003
            bus = []
            for matTag, threshold in zip(mat_tags, thresholds):
                matTag = int(matTag)
                idxu = np.argwhere(np.abs(fiber_data[-1][:, 3] - matTag) < 1e-6)
                eu = threshold
                if eu >= 0:
                    strain = np.array([np.max(data[idxu, -1]) for data in fiber_data])
                    au = np.argwhere(strain >= eu)
                else:
                    strain = np.array([np.min(data[idxu, -1]) for data in fiber_data])
                    au = np.argwhere(strain < eu)
                if len(au) == 0:
                    warn(
                        "The ultimate strain is not reached, please increase target ductility ratio! "
                        "The last value is used as the limit state.",
                        stacklevel=2,
                    )
                    bu = len(phi) - 1
                else:
                    bu = np.min(au)
                bus.append(bu)
            bu = int(np.min(bus))
        Phi_u = phi[bu]
        M_u = M[bu]
        return Phi_u, M_u

    def bilinearize(self, phiy: float, My: float, phiu: float, plot: bool = False, ax=None):
        """Bilinear Approximation of Moment-Curvature Relation.

        Parameters
        ----------
        phiy : float
            The initial yield curvature.
        My : float
            The initial yield moment.
        phiu : float
            The limit curvature.
        plot : bool, optional
            If True, plot the bilinear approximation, by default, False.
        ax : matplotlib.axes.Axes, optional
            The axes to plot the bilinear approximation, by default None.

        Returns
        -------
        (float, float)
            (Equivalent Curvature, Equivalent Moment)
        """

        phi = self.phi
        M = self.M
        bu = np.argmin(np.abs(phiu - phi))
        Q = trapezoid(M[: bu + 1], phi[: bu + 1])
        k = My / phiy
        Phi_eq = (k * phiu - np.sqrt((k * phiu) ** 2 - 2 * k * Q)) / k
        M_eq = k * Phi_eq

        M_new = np.insert(M[0 : bu + 1], 0, 0, axis=None)
        Phi_new = np.insert(phi[0 : bu + 1], 0, 0, axis=None)

        if plot:
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(10, 10 * 0.618))
            ax.plot(Phi_new, M_new, lw=1.5, c="#2529d8")
            ax.plot([0, phiy, Phi_eq, phiu], [0, My, M_eq, M_eq], lw=1.5, c="#de0f17")
            ax.plot(
                phiy,
                My,
                "o",
                ms=12,
                mec="black",
                mfc="#0099e5",
                label="Initial Yield ($\\phi_y$,$M_y$)",
            )
            ax.plot(
                Phi_eq,
                M_eq,
                "*",
                ms=15,
                mec="black",
                mfc="#ff4c4c",
                label="Equivalent Yield ($\\phi_{{eq}}$,$M_{{eq}}$)",
            )
            maxy = np.max(ax.get_yticks())
            ax.vlines(phiu, 0, maxy, colors="#34bf49", linestyles="dashed", lw=0.75)
            txt = (
                f"$\\phi_y$={phiy:.3E}, $M_y$={My:.3E}\n"
                f"$\\phi_{{eq}}$={Phi_eq:.3E}, $M_{{eq}}$={M_eq:.3E}\n"
                f"$\\phi_{{u}}$={phiu:.3E}, $M_{{u}}$={M[bu]:.3E}"
            )
            ax.text(
                0.5,
                0.4,
                txt,
                fontsize=15,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
            )
            ax.set_title(
                "Moment-Curvature",
                fontsize=22,
            )
            ax.set_xlabel("$\\phi$", fontsize=20)
            ax.set_ylabel("$M$", fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            # ax.set_xlim(0, np.max(ax.get_xticks()))
            ax.set_ylim(0, maxy)
            for loc in ["bottom", "left", "right", "top"]:
                ax.spines[loc].set_linewidth(1.0)
            ax.legend(loc="lower center", fontsize=15)
            plt.gcf().subplots_adjust(bottom=0.15)
        return Phi_eq, M_eq


def _create_model(sec_tag):
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0.0, 0.0, 0.0)
    ops.node(2, 0.0, 0.0, 0.0)
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.fix(2, 0, 1, 1, 1, 0, 0)
    ops.element("zeroLengthSection", 1, 1, 2, sec_tag)


def _create_axial_resp(p):
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, p, 0, 0, 0, 0, 0)  # nd  FX,  FY, Fz, MX, MY, MZ
    ops.wipeAnalysis()
    ops.system("BandGeneral")
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.test("NormDispIncr", 1.0e-10, 10, 3)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1 / 10)
    ops.analysis("Static")
    ops.analyze(10)
    ops.loadConst("-time", 0.0)


def _analyze(
    sec_tag,
    P=0.0,
    axis="y",
    max_phi=0.5,
    incr_phi=1e-5,
    stop_ratio=0.8,
    smart_analyze=True,
    cycle=False,
    cycle_path=None,
    debug=False,
):
    _create_model(sec_tag)
    if P != 0:
        _create_axial_resp(P)

    _setup_load(axis)
    if cycle and cycle_path is not None:
        max_phi = np.max(np.abs(cycle_path))

    protocol = _build_protocol(max_phi, incr_phi, cycle, cycle_path)
    dof = _get_dof(axis)
    results = _run_protocol_segments(protocol, dof, stop_ratio, smart_analyze, cycle_path, max_phi, debug)
    return results


def _setup_load(axis):
    dof = _get_dof(axis)
    load = [0] * 6
    load[dof - 1] = 1
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(2, *load)


def _get_dof(axis):
    axis_map = {"y": 5, "z": 6}
    try:
        return axis_map[axis.lower()]
    except KeyError:
        print("Only supported axis = y or z!")


def _build_protocol(max_phi, incr_phi, cycle, cycle_path):
    if cycle and cycle_path is not None:
        protocol = []
        for i in range(len(cycle_path) - 1):
            diff = cycle_path[i + 1] - cycle_path[i]
            steps = max(1, int(abs(diff / incr_phi)))
            protocol += [diff / steps] * steps
        return protocol
    else:
        steps = max(1, int(abs(max_phi / incr_phi)))
        return [max_phi / steps] * steps


def _run_protocol_segments(protocol, dof, stop_ratio, smart_analyze, cycle_path, max_phi, debug):
    M, PHI, RESP = [0.0], [0.0], [0.0]

    def record():
        phi = ops.nodeDisp(2, dof)
        moment = ops.getLoadFactor(2)
        PHI.append(phi)
        M.append(moment)
        RESP.append(_get_fiber_sec_data(ele_tag=1))
        return phi, moment

    def should_stop(phi, moment):
        flip = (moment - M[-1]) * (phi - PHI[-1]) < 0
        decay = abs(moment) < max(abs(m) for m in M) * (stop_ratio - 0.02)
        return (flip and decay) or abs(phi) >= max_phi

    if smart_analyze:
        analysis = SmartAnalyze(
            analysis_type="Static",
            testType="NormDispIncr",
            testTol=1e-10,
            tryAlterAlgoTypes=True,
            algoTypes=[10, 40],
            relaxation=0.5,
            minStep=1e-12,
            printPer=1e10,
            debugMode=debug,
        )
        for seg in analysis.static_split(cycle_path if cycle_path else [max_phi], maxStep=protocol[0]):
            ok = analysis.StaticAnalyze(2, dof, seg)
            phi, moment = record()
            if should_stop(phi, moment):
                break
            if ok < 0:
                raise RuntimeError("Analysis failed")  # noqa: TRY003
        analysis.close()
    else:
        for step in protocol:
            ops.integrator("DisplacementControl", 2, dof, step)
            ok = ops.analyze(1)
            phi, moment = record()
            if should_stop(phi, moment):
                break
            if ok < 0:
                raise RuntimeError("Analysis failed")  # noqa: TRY003

    RESP[0] = RESP[1] * 0
    return np.array(PHI), np.array(M), np.array(RESP)


def _get_fiber_sec_data(ele_tag: int):
    fiber_data = ops.eleResponse(ele_tag, "section", "fiberData2")
    # From column 1 to 6: "yCoord", "zCoord", "area", 'mat', "stress", "strain"
    fiber_data = np.array(fiber_data).reshape((-1, 6))
    return fiber_data


def _get_center(ys, zs, areas):
    yo = np.sum(ys * areas) / np.sum(areas)
    zo = np.sum(zs * areas) / np.sum(areas)
    return yo, zo
