import tkinter


def tcl2py(
    input_file: str,
    output_file: str,
    prefix: str = "ops",
    encoding: str = "utf-8",
    keep_comments: bool = False,
):
    """Convert OpenSees ``Tcl`` code to OpenSeesPy ``Python`` format.

    .. Note::
        * This function supports ``Tcl`` syntax and will flatten your ``Tcl`` code, including ``loops``,
          ``judgments``, ``assignments``, ``proc``, etc.,

        * Do not use assignment statements for OpenSees commands, such as
          ``set ok [analyze 1]``, ``set lambdaN [eigen 10]``, it will trigger
          an error! This is because **this function does not run the OpenSees command at all**.

        * If an encoding error is reported, please check the file and delete any special
          characters that exist, such as some Chinese characters that cannot be encoded.

    Parameters
    ----------
    input_file : str
        The name of input ``.tcl`` file.
    output_file : str
        The name of output ``.py`` file.
    prefix : str, optional
        prefix name of openseespy, by default ``ops``.
        I.e., ``import openseespy.opensees as ops``.
        If None or void str '', the prefix is not used.
        I.e., ``from openseespy.opensees import *``.
    encoding: str, optional
        file encoding format, by default "utf-8".
    keep_comments: bool, optional
        Comments are preserved, by default False.
        Note that this parameter will replace all opensees commands in the comment line, if any.
    """
    if not input_file.endswith(".tcl"):
        input_file += ".tcl"
    if not output_file.endswith(".py"):
        output_file += ".py"
    if prefix:
        import_txt = f"import openseespy.opensees as {prefix}\n\n"
        prefix += "."
    else:
        import_txt = "from openseespy.opensees import *\n\n"
        prefix = ""
    if keep_comments:
        with open(input_file, encoding=encoding) as f:
            tcl_list = f.readlines()
        for i, src in enumerate(tcl_list):
            if src[0] == "#":
                tcl_list[i] = src.replace("#", "commits___ ").replace("$", "variable___ ")
        tcl_src = "".join(tcl_list)
    else:
        with open(input_file, encoding=encoding) as f:
            tcl_src = f.read()
    tcl_src = tcl_src.replace("{", " { ")
    tcl_src = tcl_src.replace("}", " } ")
    ops_interp = __OPSInterp(prefix)

    try:
        ops_interp.eval(tcl_src)
    finally:
        with open(output_file, mode="w", encoding=encoding) as fw:
            fw.write("# This file is created by opstool.tcl2py(), author:: Yexiang Yan\n\n")
            fw.write(import_txt)
            for line in ops_interp.get_opspy_cmds():
                fw.write(line + "\n")
    print(
        f"[bold #34bf49]OpenSeesPy[/bold #34bf49] file "
        f"[bold #d20962]{output_file}[/bold #d20962] has been created successfully!"
    )


def _type_convert(a):
    if isinstance(a, str):
        try:
            a = int(a)
        except ValueError:
            try:
                a = float(a)
            except ValueError:
                a = str(a)
    return a


def _remove_commit(args, obj="#"):
    if "#" in args:
        idx = args.index(obj)
        args = args[:idx]
    return args


class __OPSInterp:
    def __init__(self, prefix) -> None:
        self.prefix = prefix
        self.interp = tkinter.Tcl()
        self.contents = []

    def _commits(self, *args):
        args = [src.replace("commits___", "#") for src in args]
        args = [src.replace("variable___", "$") for src in args]
        if args:
            args = " ".join(args).replace("# ", "#").replace("$ ", "$")
            self.contents.append(f"# {args}")
        else:
            self.contents.append("#")

    def _puts(self, *args):
        if len(args) == 1:
            self.contents.append(f"print('{args[0]}')")
        else:
            self.contents.append(f"print{args}")

    def _wipe(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}wipe{args}")

    def _model(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}model{args}")

    def _node(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}node{args}")

    def _fix(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}fix{args}")

    def _fixX(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}fixX{args}")

    def _fixY(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}fixY{args}")

    def _fixZ(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}fixZ{args}")

    def _equalDOF(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}equalDOF{args}")

    def _equalDOF_Mixed(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}equalDOF_Mixed{args}")

    def _rigidDiaphragm(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}rigidDiaphragm{args}")

    def _rigidLink(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}rigidLink{args}")

    def _uniaxialMaterial(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}uniaxialMaterial{args}")

    def _nDMaterial(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nDMaterial{args}")

    def _beamIntegration(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}beamIntegration{args}")

    def _section(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        if args[0] in (
            "Fiber",
            "fiberSec",
            "FiberWarping",
            "FiberAsym",
            "FiberThermal",
            "NDFiber",
            "NDFiberWarping",
        ):
            if args[0] not in ["NDFiber", "NDFiberWarping"] and ("-GJ" not in args or "-torsion" not in args):
                print(
                    "[bold #d20962]Warning[/bold #d20962]: "
                    "-GJ or -torsion not used for fiber section, GJ=100000000 is assumed!"
                )
                new_args = (args[0], args[1], "-GJ", 1.0e8)
            else:
                new_args = args[:4]
            self.contents.append(f"{self.prefix}section{new_args}")
            txt = args[-1]
            txt.replace("\\n", "")
            self.interp.eval(txt)
        else:
            self.contents.append(f"{self.prefix}section{args}")

    def _fiber(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}fiber{args}")

    def _patch(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}patch{args}")

    def _layer(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}layer{args}")

    def _element(self, *args):
        args = _remove_commit(args)
        args = [_type_convert(i) for i in args]
        if args[0] not in ["nonlinearBeamColumn", "forceBeamColumn", "dispBeamColumn"]:
            args = tuple([_type_convert(i) for i in args])
            self.contents.append(f"{self.prefix}element{args}")
        else:
            eleTag = args[1]
            secTag = args[5]
            if isinstance(secTag, int):
                Np = args[4]
                transfTag = args[6]
                if args[0] == "dispBeamColumn":
                    self.contents.append(f"{self.prefix}beamIntegration('Legendre', {eleTag}, {secTag}, {Np})")
                else:
                    self.contents.append(f"{self.prefix}beamIntegration('Lobatto', {eleTag}, {secTag}, {Np})")
                idx = 7
            else:
                transfTag = args[4]
                interp_paras = []
                idx = 6
                for i, arg in enumerate(args[6:]):
                    if not isinstance(arg, str):
                        interp_paras.append(arg)
                    else:
                        idx += i
                        break
                self.contents.append(f"{self.prefix}beamIntegration('{args[5]}', {eleTag}, *{interp_paras})")
            if args[0] == "nonlinearBeamColumn":
                args[0] = "forceBeamColumn"
            if "-mass" not in args and "-iter" not in args:
                self.contents.append(
                    f"{self.prefix}element('{args[0]}', {eleTag}, {args[2]}, {args[3]}, {transfTag}, {eleTag})"
                )
            else:
                self.contents.append(
                    f"{self.prefix}element('{args[0]}', {eleTag}, {args[2]}, "
                    f"{args[3]}, {transfTag}, {eleTag}, *{args[idx:]})"
                )

    def _timeSeries(self, *args):
        args = _remove_commit(args)
        args = [_type_convert(i) for i in args]
        if args[0] in ["Path", "Series"]:
            if ("-time" in args) or ("-values" in args):
                time, values = None, None
                if "-time" in args:
                    idx = args.index("-time")
                    time = list(args[idx + 1].split())
                    time = [float(i) for i in time]
                    args.pop(idx)
                    args.pop(idx)
                if "-values" in args:
                    idx = args.index("-values")
                    values = list(args[idx + 1].split())
                    values = [float(i) for i in values]
                    args.pop(idx)
                    args.pop(idx)
                if time and values:
                    args = args[:2] + ["-time", *time, "-values", *values] + args[2:]
                elif values is None:
                    args = args[:2] + ["-time", *time] + args[2:]
                else:
                    args = args[:2] + ["-values", *values] + args[2:]
                txt = f"{self.prefix}timeSeries('Path', {args[1]}, *{args[2:]})"
                self.contents.append(txt)
            else:
                self.contents.append(f"{self.prefix}timeSeries{tuple(args)}")
        else:
            self.contents.append(f"{self.prefix}timeSeries{tuple(args)}")

    def _pattern(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        if args[0].lower() != "uniformexcitation":
            if args[0].lower() == "plain" and isinstance(args[2], str):
                print(
                    f"[bold #d20962]Warning[/bold #d20962]: OpenSeesPy not support a str "
                    f"[bold #0099e5]{args[2]}[/bold #0099e5] "
                    f"followed [bold #ff4c4c]plain[/bold #ff4c4c], "
                    f"and a new [bold #f47721]timeSeries[/bold #f47721] is created with tag "
                    f"[bold #34bf49]{args[1]}[/bold #34bf49], "
                    f"please check this [bold #34bf49]pattern tag={args[1]}[/bold #34bf49]!"
                )
                tsargs = list(args[2].split())
                if len(tsargs) == 1:
                    self.contents.append(f"{self.prefix}timeSeries('{tsargs[0]}', {args[1]})")
                else:
                    self.contents.append(f"{self.prefix}timeSeries('{tsargs[0]}', {args[1]}, *{tsargs[1:]})")
                args = list(args)
                args[2] = args[1]
                args = tuple(args)
                self.contents.append(f"{self.prefix}pattern{args[:-1]}")
            else:
                self.contents.append(f"{self.prefix}pattern{args[:-1]}")
            txt = args[-1]
            txt.replace("\\n", "")
            self.interp.eval(txt)
        else:
            self.contents.append(f"{self.prefix}pattern{args}")

    def _load(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}load{args}")

    def _eleLoad(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}eleLoad{args}")

    def _sp(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sp{args}")

    def _groundMotion(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}groundMotion{args}")

    def _imposedMotion(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}imposedMotion{args}")

    def _mass(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}mass{args}")

    def _frictionModel(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}frictionModel{args}")

    def _geomTransf(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}geomTransf{args}")

    def _region(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}region{args}")

    def _rayleigh(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}rayleigh{args}")

    def _block2D(self, *args):
        args = _remove_commit(args)
        args = [_type_convert(i) for i in args]
        txt = args[-1]
        txt = txt.replace("\n", "").replace("\t", " ")
        crds = txt.split()
        crds = [_type_convert(i) for i in crds]
        self.contents.append(f"crds = {crds}")
        if isinstance(args[-2], str):
            eleargs = args[-2].split()
            eleargs = [_type_convert(i) for i in eleargs]
            args = args[:-2] + eleargs
            args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args]
            args.append("*crds")
        else:
            args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args[:-1]]
            args.append("*crds")
        txt = f"{self.prefix}block2D(" + ", ".join(args) + ")"
        self.contents.append(txt)

    def _block3D(self, *args):
        args = _remove_commit(args)
        args = [_type_convert(i) for i in args]
        txt = args[-1]
        txt = txt.replace("\n", "").replace("\t", " ")
        crds = txt.split()
        crds = [_type_convert(i) for i in crds]
        self.contents.append(f"crds = {crds}")
        if isinstance(args[-2], str):
            eleargs = args[-2].split()
            eleargs = [_type_convert(i) for i in eleargs]
            args = args[:-2] + eleargs
            args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args]
            args.append("*crds")
        else:
            args = [f"'{i}'" if isinstance(i, str) else str(i) for i in args[:-1]]
            args.append("*crds")
        txt = f"{self.prefix}block3D(" + ", ".join(args) + ")"
        self.contents.append(txt)

    def _ShallowFoundationGen(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}ShallowFoundationGen{args}")

    def _constraints(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}constraints{args}")

    def _numberer(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}numberer{args}")

    def _system(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}system{args}")

    def _test(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}test{args}")

    def _algorithm(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}algorithm{args}")

    def _integrator(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}integrator{args}")

    def _analysis(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}analysis{args}")

    def _eigen(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}eigen{args}")
        return None

    def _analyze(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}analyze{args}")
        return None

    def _modalProperties(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}modalProperties{args}")
        return None

    def _responseSpectrumAnalysis(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}responseSpectrumAnalysis{args}")

    def _recorder(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}recorder{args}")

    def _record(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}record{args}")

    def _print(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}printModel{args}")

    def _printA(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}printA{args}")

    def _logFile(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}logFile{args}")

    def _remove(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}remove{args}")

    def _loadConst(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}loadConst{args}")

    def _wipeAnalysis(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}wipeAnalysis{args}")

    def _modalDamping(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}modalDamping{args}")

    def _database(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}database{args}")

    def _getTime(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getTime{args}")

    def _setTime(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setTime{args}")

    def _testUniaxialMaterial(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}testUniaxialMaterial{args}")

    def _setStrain(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setStrain{args}")

    def _getStrain(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getStrain{args}")

    def _getStress(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getStress{args}")

    def _getTangent(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getTangent{args}")

    def _getDampTangent(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getDampTangent{args}")

    def _reactions(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}reactions{args}")

    def _nodeReaction(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeReaction{args}")

    def _nodeEigenvector(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeEigenvector{args}")

    def _setCreep(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setCreep{args}")

    def _eleResponse(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}eleResponse{args}")

    def _reset(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}reset{args}")

    def _initialize(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}initialize{args}")

    def _getLoadFactor(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getLoadFactor{args}")

    def _build(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}build{args}")

    def _printGID(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}printGID{args}")

    def _getCTestNorms(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getCTestNorms{args}")

    def _getCTestIter(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getCTestIter{args}")

    def _save(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}save{args}")

    def _restore(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}restore{args}")

    def _eleForce(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}eleForce{args}")

    def _eleDynamicalForce(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}eleDynamicalForce{args}")

    def _nodeUnbalance(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeUnbalance{args}")

    def _nodeDisp(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeDisp{args}")

    def _setNodeDisp(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setNodeDisp{args}")

    def _nodeVel(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeVel{args}")

    def _setNodeVel(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setNodeVel{args}")

    def _nodeAccel(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeAccel{args}")

    def _setNodeAccel(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setNodeAccel{args}")

    def _nodeResponse(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeResponse{args}")

    def _nodeCoord(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeCoord{args}")

    def _setNodeCoord(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setNodeCoord{args}")

    def _updateElementDomain(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}updateElementDomain{args}")

    def _getNDMM(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getNDM{args}")

    def _getNDFF(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getNDF{args}")

    def _eleNodes(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}eleNodes{args}")

    def _eleType(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}eleType{args}")

    def _nodeDOFs(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeDOFs{args}")

    def _nodeMass(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeMass{args}")

    def _nodePressure(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodePressure{args}")

    def _setNodePressure(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setNodePressure{args}")

    def _nodeBounds(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}nodeBounds{args}")

    def _startTimer(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}start{args}")

    def _stopTimer(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}stop{args}")

    def _modalDampingQ(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}modalDampingQ{args}")

    def _setElementRayleighDampingFactors(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setElementRayleighDampingFactors{args}")

    def _setPrecision(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setPrecision{args}")

    def _searchPeerNGA(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}searchPeerNGA{args}")

    def _domainChange(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}domainChange{args}")

    def _defaultUnits(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}defaultUnits{args}")

    def _stripXML(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}stripXML{args}")

    def _convertBinaryToText(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}convertBinaryToText{args}")

    def _convertTextToBinary(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}convertTextToBinary{args}")

    def _getEleTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getEleTags{args}")

    def _getCrdTransfTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getCrdTransfTags{args}")

    def _getNodeTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getNodeTags{args}")

    def _getParamTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getParamTags{args}")

    def _getParamValue(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getParamValue{args}")

    def _sectionForce(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sectionForce{args}")

    def _sectionDeformation(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sectionDeformation{args}")

    def _sectionStiffness(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sectionStiffness{args}")

    def _sectionFlexibility(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sectionFlexibility{args}")

    def _sectionLocation(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sectionLocation{args}")

    def _sectionWeight(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sectionWeight{args}")

    def _sectionTag(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sectionTag{args}")

    def _sectionDisplacement(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sectionDisplacement{args}")

    def _cbdiDisplacement(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}cbdiDisplacement{args}")

    def _basicDeformation(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}basicDeformation{args}")

    def _basicForce(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}basicForce{args}")

    def _basicStiffness(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}basicStiffness{args}")

    def _InitialStateAnalysis(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}InitialStateAnalysis{args}")

    def _totalCPU(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}totalCPU{args}")

    def _solveCPU(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}solveCPU{args}")

    def _accelCPU(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}accelCPU{args}")

    def _numFact(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}numFact{args}")

    def _numIter(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}numIter{args}")

    def _systemSize(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}systemSize{args}")

    def _version(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}version{args}")

    def _setMaxOpenFiles(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setMaxOpenFiles{args}")

    def _limitCurve(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}limitCurve{args}")

    def _setElementRayleighFactors(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setElementRayleighFactors{args}")

    def _mesh(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}mesh{args}")

    def _remesh(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}remesh{args}")

    def _parameter(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}parameter{args}")

    def _addToParameter(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}addToParameter{args}")

    def _updateParameter(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}updateParameter{args}")

    def _setParameter(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setParameter{args}")

    def _getPID(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getPID{args}")

    def _getNP(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getNP{args}")

    def _barrier(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}barrier{args}")

    def _send(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}send{args}")

    def _recv(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}recv{args}")

    def _Bcast(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}Bcast{args}")

    def _computeGradients(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}computeGradients{args}")

    def _sensitivityAlgorithm(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sensitivityAlgorithm{args}")

    def _sensNodeDisp(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sensNodeDisp{args}")

    def _sensNodeVel(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sensNodeVel{args}")

    def _sensNodeAccel(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sensNodeAccel{args}")

    def _sensLambda(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sensLambda{args}")

    def _sensSectionForce(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sensSectionForce{args}")

    def _sensNodePressure(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sensNodePressure{args}")

    def _getNumElements(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getNumElements{args}")

    def _getEleClassTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getEleClassTags{args}")

    def _getEleLoadClassTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getEleLoadClassTags{args}")

    def _getEleLoadTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getEleLoadTags{args}")

    def _getEleLoadData(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getEleLoadData{args}")

    def _getNodeLoadTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getNodeLoadTags{args}")

    def _getNodeLoadData(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getNodeLoadData{args}")

    def _randomVariable(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}randomVariable{args}")

    def _getRVTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getRVTags{args}")

    def _getRVMean(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getRVMean{args}")

    def _getRVStdv(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getRVStdv{args}")

    def _getRVPDF(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getRVPDF{args}")

    def _getRVCDF(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getRVCDF{args}")

    def _getRVInverseCDF(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getRVInverseCDF{args}")

    def _addCorrelate(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}addCorrelate{args}")

    def _correlate(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}correlate{args}")

    def _performanceFunction(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}performanceFunction{args}")

    def _gradPerformanceFunction(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}gradPerformanceFunction{args}")

    def _transformUtoX(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}transformUtoX{args}")

    def _wipeReliability(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}wipeReliability{args}")

    def _updateMaterialStage(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}updateMaterialStage{args}")

    def _sdfResponse(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}sdfResponse{args}")

    def _probabilityTransformation(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}probabilityTransformation{args}")

    def _startPoint(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}startPoint{args}")

    def _randomNumberGenerator(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}randomNumberGenerator{args}")

    def _reliabilityConvergenceCheck(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}reliabilityConvergenceCheck{args}")

    def _searchDirection(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}searchDirection{args}")

    def _meritFunctionCheck(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}meritFunctionCheck{args}")

    def _stepSizeRule(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}stepSizeRule{args}")

    def _rootFinding(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}rootFinding{args}")

    def _functionEvaluator(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}functionEvaluator{args}")

    def _gradientEvaluator(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}gradientEvaluator{args}")

    def _runFOSMAnalysis(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}runFOSMAnalysis{args}")

    def _findDesignPoint(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}findDesignPoint{args}")

    def _runFORMAnalysis(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}runFORMAnalysis{args}")

    def _getLSFTags(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getLSFTags{args}")

    def _runImportanceSamplingAnalysis(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}runImportanceSamplingAnalysis{args}")

    def _IGA(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}IGA{args}")

    def _NDTest(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}NDTest{args}")

    def _getNumThreads(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}getNumThreads{args}")

    def _setNumThreads(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setNumThreads{args}")

    def _setStartNodeTag(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}setStartNodeTag{args}")

    def _hystereticBackbone(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}hystereticBackbone{args}")

    def _stiffnessDegradation(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}stiffnessDegradation{args}")

    def _strengthDegradation(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}strengthDegradation{args}")

    def _unloadingRule(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}unloadingRule{args}")

    def _partition(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}partition{args}")

    def _pc(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}pressureConstraint{args}")

    def _domainCommitTag(self, *args):
        args = _remove_commit(args)
        args = tuple([_type_convert(i) for i in args])
        self.contents.append(f"{self.prefix}domainCommitTag{args}")

    @staticmethod
    def _display(*args):
        print(f"This <display {args}> function will be ignored!")

    @staticmethod
    def _prp(*args):
        print(f"This display <prp {args}> function will be ignored!")

    @staticmethod
    def _vup(*args):
        print(f"This display <vup {args}> function will be ignored!")

    @staticmethod
    def _vpn(*args):
        print(f"This display <vpn {args}> function will be ignored!")

    @staticmethod
    def _vrp(*args):
        print(f"This display <vrp {args}> function will be ignored!")

    def _createcommand(self):
        self.interp.createcommand("commits___", self._commits)
        self.interp.createcommand("puts", self._puts)
        self.interp.createcommand("wipe", self._wipe)
        self.interp.createcommand("model", self._model)
        self.interp.createcommand("node", self._node)
        self.interp.createcommand("fix", self._fix)
        self.interp.createcommand("fixX", self._fixX)
        self.interp.createcommand("fixY", self._fixY)
        self.interp.createcommand("fixZ", self._fixZ)
        self.interp.createcommand("equalDOF", self._equalDOF)
        self.interp.createcommand("equalDOF_Mixed", self._equalDOF_Mixed)
        self.interp.createcommand("rigidDiaphragm", self._rigidDiaphragm)
        self.interp.createcommand("rigidLink", self._rigidLink)
        self.interp.createcommand("element", self._element)
        self.interp.createcommand("timeSeries", self._timeSeries)
        # self.interp.createcommand('Series', _timeSeries)
        self.interp.createcommand("pattern", self._pattern)
        self.interp.createcommand("load", self._load)
        self.interp.createcommand("eleLoad", self._eleLoad)
        self.interp.createcommand("sp", self._sp)
        self.interp.createcommand("groundMotion", self._groundMotion)
        self.interp.createcommand("imposedMotion", self._imposedMotion)
        self.interp.createcommand("imposedSupportMotion", self._imposedMotion)
        self.interp.createcommand("mass", self._mass)
        self.interp.createcommand("uniaxialMaterial", self._uniaxialMaterial)
        self.interp.createcommand("nDMaterial", self._nDMaterial)
        self.interp.createcommand("beamIntegration", self._beamIntegration)
        self.interp.createcommand("section", self._section)
        self.interp.createcommand("fiber", self._fiber)
        self.interp.createcommand("patch", self._patch)
        self.interp.createcommand("layer", self._layer)
        self.interp.createcommand("frictionModel", self._frictionModel)
        self.interp.createcommand("geomTransf", self._geomTransf)
        self.interp.createcommand("region", self._region)
        self.interp.createcommand("rayleigh", self._rayleigh)
        self.interp.createcommand("block2D", self._block2D)
        self.interp.createcommand("block2d", self._block2D)
        self.interp.createcommand("block3D", self._block3D)
        self.interp.createcommand("block3d", self._block3D)
        self.interp.createcommand("ShallowFoundationGen", self._ShallowFoundationGen)
        self.interp.createcommand("constraints", self._constraints)
        self.interp.createcommand("numberer", self._numberer)
        self.interp.createcommand("system", self._system)
        self.interp.createcommand("test", self._test)
        self.interp.createcommand("algorithm", self._algorithm)
        self.interp.createcommand("integrator", self._integrator)
        self.interp.createcommand("analysis", self._analysis)
        self.interp.createcommand("eigen", self._eigen)
        self.interp.createcommand("analyze", self._analyze)
        self.interp.createcommand("modalProperties", self._modalProperties)
        self.interp.createcommand("responseSpectrumAnalysis", self._responseSpectrumAnalysis)
        self.interp.createcommand("record", self._record)
        self.interp.createcommand("recorder", self._recorder)
        self.interp.createcommand("print", self._print)
        self.interp.createcommand("printA", self._printA)
        self.interp.createcommand("logFile", self._logFile)
        self.interp.createcommand("remove", self._remove)
        self.interp.createcommand("loadConst", self._loadConst)
        self.interp.createcommand("wipeAnalysis", self._wipeAnalysis)
        self.interp.createcommand("modalDamping", self._modalDamping)
        self.interp.createcommand("database", self._database)
        self.interp.createcommand("getTime", self._getTime)
        self.interp.createcommand("setTime", self._setTime)
        self.interp.createcommand("testUniaxialMaterial", self._testUniaxialMaterial)
        self.interp.createcommand("setStrain", self._setStrain)
        self.interp.createcommand("getStrain", self._getStrain)
        self.interp.createcommand("getStress", self._getStress)
        self.interp.createcommand("getTangent", self._getTangent)
        self.interp.createcommand("getDampTangent", self._getDampTangent)
        self.interp.createcommand("reactions", self._reactions)
        self.interp.createcommand("nodeReaction", self._nodeReaction)
        self.interp.createcommand("remove", self._remove)
        self.interp.createcommand("nodeEigenvector", self._nodeEigenvector)
        self.interp.createcommand("setCreep", self._setCreep)
        self.interp.createcommand("eleResponse", self._eleResponse)
        self.interp.createcommand("reset", self._reset)
        self.interp.createcommand("initialize", self._initialize)
        self.interp.createcommand("getLoadFactor", self._getLoadFactor)
        self.interp.createcommand("build", self._build)
        self.interp.createcommand("printGID", self._printGID)
        self.interp.createcommand("testNorm", self._getCTestNorms)
        self.interp.createcommand("testIter", self._getCTestIter)
        self.interp.createcommand("save", self._save)
        self.interp.createcommand("restore", self._restore)
        self.interp.createcommand("eleForce", self._eleForce)
        self.interp.createcommand("eleDynamicalForce", self._eleDynamicalForce)
        self.interp.createcommand("nodeUnbalance", self._nodeUnbalance)
        self.interp.createcommand("nodeDisp", self._nodeDisp)
        self.interp.createcommand("setNodeDisp", self._setNodeDisp)
        self.interp.createcommand("nodeVel", self._nodeVel)
        self.interp.createcommand("setNodeVel", self._setNodeVel)
        self.interp.createcommand("nodeAccel", self._nodeAccel)
        self.interp.createcommand("setNodeAccel", self._setNodeAccel)
        self.interp.createcommand("nodeResponse", self._nodeResponse)
        self.interp.createcommand("nodeCoord", self._nodeCoord)
        self.interp.createcommand("setNodeCoord", self._setNodeCoord)
        self.interp.createcommand("updateElementDomain", self._updateElementDomain)
        self.interp.createcommand("getNDM", self._getNDMM)
        self.interp.createcommand("getNDF", self._getNDFF)
        self.interp.createcommand("eleNodes", self._eleNodes)
        self.interp.createcommand("eleType", self._eleType)
        self.interp.createcommand("nodeDOFs", self._nodeDOFs)
        self.interp.createcommand("nodeMass", self._nodeMass)
        self.interp.createcommand("nodePressure", self._nodePressure)
        self.interp.createcommand("setNodePressure", self._setNodePressure)
        self.interp.createcommand("nodeBounds", self._nodeBounds)
        self.interp.createcommand("start", self._startTimer)
        self.interp.createcommand("stop", self._stopTimer)
        self.interp.createcommand("modalDampingQ", self._modalDampingQ)
        self.interp.createcommand("setElementRayleighDampingFactors", self._setElementRayleighDampingFactors)
        self.interp.createcommand("setPrecision", self._setPrecision)
        self.interp.createcommand("searchPeerNGA", self._searchPeerNGA)
        self.interp.createcommand("domainChange", self._domainChange)
        self.interp.createcommand("defaultUnits", self._defaultUnits)
        self.interp.createcommand("stripXML", self._stripXML)
        self.interp.createcommand("convertBinaryToText", self._convertBinaryToText)
        self.interp.createcommand("convertTextToBinary", self._convertTextToBinary)
        self.interp.createcommand("getEleTags", self._getEleTags)
        self.interp.createcommand("getCrdTransfTags", self._getCrdTransfTags)
        self.interp.createcommand("getNodeTags", self._getNodeTags)
        self.interp.createcommand("getParamTags", self._getParamTags)
        self.interp.createcommand("getParamValue", self._getParamValue)
        self.interp.createcommand("sectionForce", self._sectionForce)
        self.interp.createcommand("sectionDeformation", self._sectionDeformation)
        self.interp.createcommand("sectionStiffness", self._sectionStiffness)
        self.interp.createcommand("sectionFlexibility", self._sectionFlexibility)
        self.interp.createcommand("sectionLocation", self._sectionLocation)
        self.interp.createcommand("sectionWeight", self._sectionWeight)
        self.interp.createcommand("sectionTag", self._sectionTag)
        self.interp.createcommand("sectionDisplacement", self._sectionDisplacement)
        self.interp.createcommand("cbdiDisplacement", self._cbdiDisplacement)
        self.interp.createcommand("basicDeformation", self._basicDeformation)
        self.interp.createcommand("basicForce", self._basicForce)
        self.interp.createcommand("basicStiffness", self._basicStiffness)
        self.interp.createcommand("InitialStateAnalysis", self._InitialStateAnalysis)
        self.interp.createcommand("totalCPU", self._totalCPU)
        self.interp.createcommand("solveCPU", self._solveCPU)
        self.interp.createcommand("accelCPU", self._accelCPU)
        self.interp.createcommand("numFact", self._numFact)
        self.interp.createcommand("numIter", self._numIter)
        self.interp.createcommand("systemSize", self._systemSize)
        self.interp.createcommand("version", self._version)
        self.interp.createcommand("setMaxOpenFiles", self._setMaxOpenFiles)
        self.interp.createcommand("limitCurve", self._limitCurve)

        self.interp.createcommand("equalDOF_Mixed", self._equalDOF_Mixed)
        self.interp.createcommand("setElementRayleighFactors", self._setElementRayleighFactors)
        self.interp.createcommand("mesh", self._mesh)
        self.interp.createcommand("remesh", self._remesh)
        self.interp.createcommand("parameter", self._parameter)
        self.interp.createcommand("addToParameter", self._addToParameter)
        self.interp.createcommand("updateParameter", self._updateParameter)
        self.interp.createcommand("setParameter", self._setParameter)
        self.interp.createcommand("getPID", self._getPID)
        self.interp.createcommand("getNP", self._getNP)
        self.interp.createcommand("barrier", self._barrier)
        self.interp.createcommand("send", self._send)
        self.interp.createcommand("recv", self._recv)
        self.interp.createcommand("Bcast", self._Bcast)
        self.interp.createcommand("computeGradients", self._computeGradients)
        self.interp.createcommand("sensitivityAlgorithm", self._sensitivityAlgorithm)
        self.interp.createcommand("sensNodeDisp", self._sensNodeDisp)
        self.interp.createcommand("sensNodeVel", self._sensNodeVel)
        self.interp.createcommand("sensNodeAccel", self._sensNodeAccel)
        self.interp.createcommand("sensLambda", self._sensLambda)
        self.interp.createcommand("sensSectionForce", self._sensSectionForce)
        self.interp.createcommand("sensNodePressure", self._sensNodePressure)
        self.interp.createcommand("getNumElements", self._getNumElements)
        self.interp.createcommand("getEleClassTags", self._getEleClassTags)
        self.interp.createcommand("getEleLoadClassTags", self._getEleLoadClassTags)
        self.interp.createcommand("getEleLoadTags", self._getEleLoadTags)
        self.interp.createcommand("getEleLoadData", self._getEleLoadData)
        self.interp.createcommand("getNodeLoadTags", self._getNodeLoadTags)
        self.interp.createcommand("getNodeLoadData", self._getNodeLoadData)
        self.interp.createcommand("randomVariable", self._randomVariable)
        self.interp.createcommand("getRVTags", self._getRVTags)
        self.interp.createcommand("getMean", self._getRVMean)
        self.interp.createcommand("getStdv", self._getRVStdv)
        self.interp.createcommand("getPDF", self._getRVPDF)
        self.interp.createcommand("getCDF", self._getRVCDF)
        self.interp.createcommand("getInverseCDF", self._getRVInverseCDF)
        self.interp.createcommand("correlate", self._correlate)
        self.interp.createcommand("performanceFunction", self._performanceFunction)
        self.interp.createcommand("gradPerformanceFunction", self._gradPerformanceFunction)
        self.interp.createcommand("transformUtoX", self._transformUtoX)
        self.interp.createcommand("wipeReliability", self._wipeReliability)
        self.interp.createcommand("updateMaterialStage", self._updateMaterialStage)
        self.interp.createcommand("sdfResponse", self._sdfResponse)
        self.interp.createcommand("probabilityTransformation", self._probabilityTransformation)
        self.interp.createcommand("startPoint", self._startPoint)
        self.interp.createcommand("randomNumberGenerator", self._randomNumberGenerator)
        self.interp.createcommand("reliabilityConvergenceCheck", self._reliabilityConvergenceCheck)
        self.interp.createcommand("searchDirection", self._searchDirection)
        self.interp.createcommand("meritFunctionCheck", self._meritFunctionCheck)
        self.interp.createcommand("stepSizeRule", self._stepSizeRule)
        self.interp.createcommand("rootFinding", self._rootFinding)
        self.interp.createcommand("functionEvaluator", self._functionEvaluator)
        self.interp.createcommand("gradientEvaluator", self._gradientEvaluator)
        self.interp.createcommand("runFOSMAnalysis", self._runFOSMAnalysis)
        self.interp.createcommand("findDesignPoint", self._findDesignPoint)
        self.interp.createcommand("runFORMAnalysis", self._runFORMAnalysis)
        self.interp.createcommand("getLSFTags", self._getLSFTags)
        self.interp.createcommand("runImportanceSamplingAnalysis", self._runImportanceSamplingAnalysis)
        self.interp.createcommand("IGA", self._IGA)
        self.interp.createcommand("NDTest", self._NDTest)
        self.interp.createcommand("getNumThreads", self._getNumThreads)
        self.interp.createcommand("setNumThreads", self._setNumThreads)
        self.interp.createcommand("setStartNodeTag", self._setStartNodeTag)
        self.interp.createcommand("hystereticBackbone", self._hystereticBackbone)
        self.interp.createcommand("stiffnessDegradation", self._stiffnessDegradation)
        self.interp.createcommand("strengthDegradation", self._strengthDegradation)
        self.interp.createcommand("unloadingRule", self._unloadingRule)
        self.interp.createcommand("partition", self._partition)
        self.interp.createcommand("pressureConstraint", self._pc)
        self.interp.createcommand("domainCommitTag", self._domainCommitTag)
        # ------------------------------------------------------------
        self.interp.createcommand("display", self._display)
        self.interp.createcommand("prp", self._prp)
        self.interp.createcommand("vup", self._vup)
        self.interp.createcommand("vpn", self._vpn)
        self.interp.createcommand("vrp", self._vrp)

    def eval(self, contents):
        self._createcommand()
        self.interp.eval(contents)

    def get_interp(self):
        return self.interp

    def get_opspy_cmds(self):
        return self.contents
