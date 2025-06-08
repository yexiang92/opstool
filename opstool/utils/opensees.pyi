"""code hints for openseespy.

Author: Yexiang Yan
Mail: yexiang_yan@outlook.com
"""

from typing import Union

def model(*args) -> None:
    """``model('basic', '-ndm', ndm, '-ndf', ndf)``

    Set the default model dimensions and number of dofs.
    See https://openseespydoc.readthedocs.io/en/latest/src/model.html

    * ndm (int)  :   number of dimensions (1,2,3)
    * ndf (int)  :   number of dofs (optional)
    """
    pass

def node(*args) -> None:
    """
    ``node(nodeTag, *crds, '-ndf', ndf, '-mass', *mass, '-disp', *disp, '-vel', *vel, '-accel', *accel)``

    Create a OpenSees node.
    See https://openseespydoc.readthedocs.io/en/latest/src/node.html

    * nodeTag (int)  :   node tag.
    * crds (list (float))  :   nodal coordinates.
    * ndf (float)  :   nodal ndf. (optional)
    * mass (list (float))  :   nodal mass. (optional)
    * vel (list (float))  :   nodal velocities. (optional)
    * accel (list (float))  :   nodal accelerations. (optional)
    """
    pass

def mass(*args) -> None:
    """``mass(nodeTag, *massValues)``

    This command is used to set the mass at a node, replacing any previously defined mass at the node.

    * nodeTag (int): integer tag identifying node whose mass is set
    * massValues (list (float)): ndf nodal mass values corresponding to each DOF
    """
    pass

def region(*args) -> None:
    """``region(regTag, '-ele', *eles, '-eleOnly', *eles, '-eleRange', startEle, endEle, '-eleOnlyRange',
    startEle, endEle, '-node', *nodes, '-nodeOnly', *nodes, '-nodeRange', startNode, endNode, '-nodeOnlyRange',
    startNode, endNode, '-rayleigh', alphaM, betaK, betaKinit, betaKcomm)``

    See https://openseespydoc.readthedocs.io/en/latest/src/region.html

    The region command is used to label a group of nodes and elements.
    This command is also used to assign rayleigh damping parameters to the nodes and elements in this region.
    The region is specified by either elements or nodes, not both.
    If elements are defined, the region includes these elements and the all connected nodes,
    unless the -eleOnly option is used, in which case only elements are included.
    If nodes are specified, the region includes these nodes and all elements of which
    all nodes are prescribed to be in the region, unless the -nodeOnly option is used
    in which case only the nodes are included.

    * regTag (int) : unique integer tag
    * eles (list (int)) : tags of selected elements in domain to be included in region (optional)
    * nodes (list (int)) : tags of selected nodes in domain to be included in region (optional)
    * startEle (int) : tag for start element (optional)
    * endEle (int) : tag for end element (optional)
    * startNode (int) : tag for start node (optional)
    * endNode (int) : tag for end node (optional)
    * alphaM (float) : factor applied to elements or nodes mass matrix (optional)
    * betaK (float) : factor applied to elements current stiffness matrix (optional)
    * betaKinit (float) : factor applied to elements initial stiffness matrix (optional)
    * betaKcomm (float) : factor applied to elements committed stiffness matrix (optional)

    Note:
    * The user cannot prescribe the region by BOTH elements and nodes.
    """
    pass

def rayleigh(*args) -> None:
    """``rayleigh(alphaM, betaK, betaKinit, betaKcomm)``

    See https://openseespydoc.readthedocs.io/en/latest/src/reyleigh.html

    This command is used to assign damping to all previously defined elements and nodes.
    When using rayleigh damping in OpenSees, the damping matrix for an element or node,
    D is specified as a combination of stiffness and mass-proportional damping matrices:

    D=αM∗M+βK∗Kcurr+βKinit∗Kinit+βKcomm∗Kcommit

    * alphaM (float): factor applied to elements or nodes mass matrix
    * betaK (float): factor applied to elements current stiffness matrix.
    * betaKinit (float): factor applied to element initial stiffness matrix.
    * betaKcomm (float): factor applied to element committed stiffness matrix.
    """
    pass

def block2D(*args) -> None:
    """``block2D(numX, numY, startNode, startEle, eleType, *eleArgs, *crds)``

    Create mesh of quadrilateral elements, see https://openseespydoc.readthedocs.io/en/latest/src/block2D.html
    """
    pass

def block3D(*args) -> None:
    """``block3D(numX, numY, numZ, startNode, startEle, eleType, *eleArgs, *crds)``

    The block3D command generates three-dimensional meshes of eight-node brick solid element.
    See https://openseespydoc.readthedocs.io/en/latest/src/block3D.html
    """
    pass

def beamIntegration(*args) -> None:
    """``beamIntegration(type, tag, *args)``

    See https://openseespydoc.readthedocs.io/en/latest/src/beamIntegration.html

    A wide range of numerical integration options are available in OpenSees
    to represent distributed plasticity or non-prismatic section details in Beam-Column Elements,
    i.e., across the entire element domain [0, L].

    Integration Methods for Distributed Plasticity.
    Distributed plasticity methods permit yielding at any integration point along the element length.

    ``beamIntegration('Lobatto', tag, secTag, N)`` ------> prismatic section

    ``beamIntegration('Lobatto', tag, N, *secTags)`` -----> non-prismatic sections.

    ``beamIntegration('Legendre', tag, secTag, N)`` ------> prismatic section

    ``beamIntegration('Legendre', tag, N, *secTags)`` -----> non-prismatic sections.

    Plastic Hinge Integration Methods. Plastic hinge integration methods confine material yielding
    to regions of the element of specified length while the remainder of the element is linear elastic.
    A summary of plastic hinge integration methods is found in (Scott and Fenves 2006).

    ``beamIntegration('HingeRadau', tag, secI, lpI, secJ, lpJ, secE)``

    ``beamIntegration('HingeMidpoint', tag, secI, lpI, secJ, lpJ, secE)``

    ``beamIntegration('HingeRadauTwo', tag, secI, lpI, secJ, lpJ, secE)``
    """
    pass

def uniaxialMaterial(*args) -> None:
    """``uniaxialMaterial(matType, matTag, *matArgs)``

    This command is used to construct a UniaxialMaterial object which represents uniaxial
    stress-strain (or force-deformation) relationships.
    See https://openseespydoc.readthedocs.io/en/latest/src/uniaxialMaterial.html

    * matType (str): material type
    * matTag (int): material tag.
    * matArgs (list): a list of material arguments must be preceded with *.
    """
    pass

def hystereticBackbone(*args) -> None:
    """backbone function, see https://openseespydoc.readthedocs.io/en/latest/src/Backbone.html"""
    pass

def stiffnessDegradation(*args) -> None:
    """``stiffnessDegradation(type, tag, *args)``"""
    pass

def strengthDegradation(*args) -> None:
    """``strengthDegradation(type, tag, *args)``"""
    pass

def strengthControl(*args) -> None:
    """``strengthControl(type, tag, *args)``

    Same as strengthDegradation.
    """
    pass

def unloadingRule(*args) -> None:
    """``unloadingRule(type, tag, *args)``"""
    pass

def limitCurve(*args) -> None:
    """``limitCurve(type, arg1, arg2, ...)``

    ``limitCurve("Axial", tag, eleTag, Fsw, Kdeg, Fres, defType, forType, <ndI, ndJ, dof, perpDirn, delta, eleRemove>)``

    ``limitCurve("Shear", tag, eleTag, rho, fc, b, h, d, Fsw, Kdeg, Fres, defType, forType, <ndI, ndJ, dof, perpDirn, delta>)``

    ``limitCurve("ThreePoint", tag, eleTag, x1, y1, x2, y2, x3, y3, Kdeg, Fres, defType, forType, <ndI, ndJ, dof, perpDirn>)``

    ``limitCurve("RotationShearCurve", crvTag, eleTag, ndI, ndJ, rotAxis, Vn, Vr, Kdeg, rotLim)``

    ``limitCurve("RotationShearCurve", crvTag, eleTag, ndI, ndJ, rotAxis, Vn, Vr, Kdeg, defType,
                 b, d, h, L, st, As, Acc, ld, db, rhot, fc, fy, fyt, delta)``

    Construct a failure curve for a limit state material.
    See https://opensees.berkeley.edu/wiki/index.php/Limit_Curve
    """
    pass

def nDMaterial(*args) -> None:
    """``nDMaterial(matType, matTag, *matArgs)``

    This command is used to construct an NDMaterial object which represents the stress-strain relationship
    at the gauss-point of a continuum element.
    See https://openseespydoc.readthedocs.io/en/latest/src/ndMaterial.html


    * matType (str): material type
    * matTag (int): material tag.
    * matArgs (list): a list of material arguments must be preceded with *.
    """
    pass

def section(*args) -> None:
    """``section(secType, secTag, *secArgs)``

    This command is used to construct a SectionForceDeformation object, hereto referred to as Section, which represents
    force-deformation (or resultant stress-strain) relationships at beam-column and plate sample points.

    See https://openseespydoc.readthedocs.io/en/latest/src/section.html

    ``section('Elastic', secTag, E_mod, A, Iz, <G_mod, alphaY>)`` for 2D

    ``section('Elastic', secTag, E_mod, A, Iz, Iy, G_mod, Jxx, <alphaY, alphaZ>)`` for 3D

    ``section('Fiber', secTag, '-GJ', GJ)``

    ``section('Fiber', secTag, '-torsion', torsionMatTag)``

    ``section('Aggregator', secTag, *mats, '-section', sectionTag)``

    ``section('ElasticMembranePlateSection', secTag, E_mod, nu, h, rho, <Ep_modifier>)``

    ``section('PlateFiber', secTag, matTag, h)``

    ``section('LayeredShell', sectionTag, nLayers, *mats)``
    """
    pass

def fiber(*args) -> None:
    """``fiber(yloc, zloc, A, matTag)``

    This command allows the user to construct a single fiber and add it to the enclosing FiberSection or NDFiberSection.
    See https://openseespydoc.readthedocs.io/en/latest/src/fiber.html

    * yloc (float) : y coordinate of the fiber in the section (local coordinate system)
    * zloc (float) : z coordinate of the fiber in the section (local coordinate system)
    * A (float) : cross-sectional area of fiber
    * matTag (int) : material tag associated with this fiber (UniaxialMaterial tag for a FiberSection and NDMaterial tag for use in an NDFiberSection).
    """
    pass

def patch(*args) -> None:
    """
    ``patch('quad', matTag, numSubdivIJ, numSubdivJK, *crdsI, *crdsJ, *crdsK, *crdsL)``

    ``patch('rect', matTag, numSubdivY, numSubdivZ, *crdsI, *crdsJ)``

    ``patch('circ', matTag, numSubdivCirc, numSubdivRad, *center, *rad, *ang)``

    See https://openseespydoc.readthedocs.io/en/latest/src/patch.html
    """
    pass

def layer(*args) -> None:
    """
    ``layer('straight', matTag, numFiber, areaFiber, *start, *end)``

    ``layer('circ', matTag,numFiber,areaFiber,*center,radius,*ang=[0.0,360.0-360/numFiber])``

    See https://openseespydoc.readthedocs.io/en/latest/src/layer.html
    """
    pass

def frictionModel(*args) -> None:
    """``frictionModel(frnType, frnTag, *frnArgs)``

    The frictionModel command is used to construct a friction model object,
    which specifies the behavior of the coefficient of friction in terms of the absolute sliding
    velocity and the pressure on the contact area. The command has at least one argument, the friction model type.
    See https://openseespydoc.readthedocs.io/en/latest/src/frictionModel.html

    * frnType (str) : frictionModel type
    * frnTag (int) : frictionModel tag.
    * frnArgs (list) : a list of frictionModel arguments, must be preceded with *.

    ``frictionModel('Coulomb', frnTag, mu)``

    ``frictionModel('VelDependent', frnTag, muSlow, muFast, transRate)``

    ``frictionModel('VelNormalFrcDep', frnTag, aSlow, nSlow, aFast, nFast, alpha0, alpha1, alpha2, maxMuFact)``

    ``frictionModel('VelPressureDep', frnTag, muSlow, muFast0, A, deltaMu, alpha, transRate)``

    ``frictionModel('VelDepMultiLinear', frnTag, '-vel', *velPoints, '-frn', *frnPoints)``
    """
    pass

def geomTransf(*args) -> None:
    """``geomTransf(transfType, transfTag, *transfArgs)``

    The geometric-transformation command is used to construct a coordinate-transformation (CrdTransf) object,
    which transforms beam element stiffness and resisting force from the basic system to the global-coordinate system.
    The command has at least one argument, the transformation type.
    See https://openseespydoc.readthedocs.io/en/latest/src/geomTransf.html

    * transfType (str): geomTransf type
    * transfTag (int): geomTransf tag.
    * transfArgs (list): a list of geomTransf arguments must be preceded with *.

    ``geomTransf('Linear', transfTag, '-jntOffset', *dI, *dJ)`` for 2D

    ``geomTransf('Linear', transfTag, *vecxz, '-jntOffset', *dI, *dJ)`` for 3D

    ``geomTransf('PDelta', transfTag, *vecxz, '-jntOffset', *dI, *dJ)`` for 3D

    ``geomTransf('Corotational', transfTag, *vecxz, '-jntOffset', *dI, *dJ)`` for 3D

    Note:
        Currently the ``Corotational`` transformation does not deal with element loads and
        will ignore any that are applied to the element.
    """
    pass

def fix(*args) -> None:
    """``fix(nodeTag, *constrValues)``

    Create a homogeneous SP constriant.

    * nodeTag (int): tag of node to be constrained
    * constrValues (list (int)): a list of constraint values (0 or 1), must be preceded with *.
        0--free; 1--fixed
    """
    pass

def fixX(*args) -> None:
    """``fixX(x, *constrValues, '-tol', tol=1e-10)``

    Create homogeneous SP constriants.

    * x (float): x-coordinate of nodes to be constrained
    * constrValues (list (int)): a list of constraint values (0 or 1), must be preceded with *.
        0 free 1 fixed
    * tol (float): user-defined tolerance (optional)
    """
    pass

def fixY(*args) -> None:
    """``fixY(y, *constrValues, '-tol', tol=1e-10)``

    Create homogeneous SP constriants.

    * y (float): y-coordinate of nodes to be constrained
    * constrValues (list (int)): a list of constraint values (0 or 1), must be preceded with *.
        0 free 1 fixed
    * tol (float): user-defined tolerance (optional)
    """
    pass

def fixZ(*args) -> None:
    """``fixZ(z, *constrValues, '-tol', tol=1e-10)``

    Create homogeneous SP constriants.

    * z (float) : z-coordinate of nodes to be constrained
    * constrValues (list (int)) : a list of constraint values (0 or 1), must be preceded with *.
        0 free 1 fixed
    * tol (float) : user-defined tolerance (optional)
    """
    pass

def equalDOF(*args) -> None:
    """``equalDOF(rNodeTag, cNodeTag, *dofs)``

    Create a multi-point constraint between nodes.

    * rNodeTag (int) : integer tag identifying the retained, or master node.
    * cNodeTag (int) : integer tag identifying the constrained, or slave node.
    * dofs (list (int)) : nodal degrees-of-freedom that are constrained at the cNode to be the same as those at the rNo
    """
    pass

def equalDOF_Mixed(*args) -> None:
    """``equalDOF_Mixed(rNodeTag, cNodeTag, numDOF, *rcdofs)``

    Create a multi-point constraint between nodes.

    * rNodeTag (int) : integer tag identifying the retained, or master node.
    * cNodeTag (int) : integer tag identifying the constrained, or slave node.
    * numDOF (int) : number of dofs to be constrained
    * rcdofs (list (int)) : nodal degrees-of-freedom that are constrained at the cNode
        to be the same as those at the rNode Valid range is from 1 through ndf, the number of nodal degrees-of-freedom.
        rcdofs = [rdof1, cdof1, rdof2, cdof2, ...]
    """
    pass

def rigidDiaphragm(*args) -> None:
    """``rigidDiaphragm(perpDirn, rNodeTag, *cNodeTags)``

    Create a multi-point constraint between nodes. These objects will constrain
    certain degrees-of-freedom at the listed secondary nodes to move as if in a rigid plane with
    the primary (retained) node. To enforce this constraint, ``Transformation`` constraint handler is recommended.

    * perpDirn (int) : direction perpendicular to the rigid plane (i.e. direction 3 corresponds to the 1-2 plane)
    * rNodeTag (int) : integer tag identifying the retained (primary) node
    * cNodeTags (list (int)) : integar tags identifying the constrained (secondary) nodes
    """
    pass

def rigidLink(*args) -> None:
    """``rigidLink(type, rNodeTag, cNodeTag)``

    Create a multi-point constraint between nodes.

    * type (str) : string-based argument for rigid-link type:
        (1) 'bar': only the translational degree-of-freedom will be constrained
            to be exactly the same as those at the master node
        (2) 'beam': both the translational and rotational degrees of freedom are constrained.
    * rNodeTag (int) : integer tag identifying the master node
    * cNodeTag (int) : integar tag identifying the slave node
    """
    pass

def pressureConstraint(*args) -> None:
    """``pressureConstraint(nodeTag, pNodeTag)``

    Create a pressure constraint for incompressible flow.
    See https://openseespydoc.readthedocs.io/en/latest/src/pc.html

    * nodeTag (int) : tag of node to be constrained.
    * pNodeTag (int) : tag of extra pressure node, which must exist before calling this command.
    """
    pass

def timeSeries(*args) -> None:
    """``timeSeries(tsType, tsTag, *tsArgs)``

    This command is used to construct a TimeSeries object which represents the relationship between the time in the domain, t
    , and the load factor applied to the loads, λ, in the load pattern with which the TimeSeries
    object is associated, i.e. λ=F(t).
    See https://openseespydoc.readthedocs.io/en/latest/src/timeSeries.html

    * tsType (str) : time series type.
    * tsTag (int) : time series tag.
    * tsArgs (list) : a list of time series arguments

    ``timeSeries('Path', tag, '-dt', dt=0.0, '-values', *values, '-time', *time, '-filePath', filePath='',
    '-fileTime', fileTime='', '-factor', factor=1.0, '-startTime', startTime=0.0, '-useLast', '-prependZero')``

    ``timeSeries('Linear', tag, '-factor', factor=1.0, '-tStart', tStart=0.0)``

    ``timeSeries('Constant', tag, '-factor', factor=1.0)``

    ``timeSeries('Trig', tag, tStart, tEnd, period, '-factor', factor=1.0, '-shift', shift=0.0, '-zeroShift', zeroShift=0.0)``

    ``timeSeries('Triangle', tag, tStart, tEnd, period, '-factor', factor=1.0, '-shift', shift=0.0, '-zeroShift', zeroShift=0.0)``

    ``timeSeries('Rectangular', tag, tStart, tEnd, '-factor', factor=1.0)``

    ``timeSeries('Pulse', tag, tStart, tEnd, period, '-width', width=0.5,
    '-shift', shift=0.0, '-factor', factor=1.0, '-zeroShift', zeroShift=0.0)``
    """
    pass

def pattern(*args) -> None:
    """``pattern(patternType, patternTag, *patternArgs)``

    The pattern command is used to construct a LoadPattern and add it to the Domain. Each LoadPattern in OpenSees
    has a TimeSeries associated with it. In addition it may contain ElementLoads,
    NodalLoads and SinglePointConstraints. Some of these SinglePoint constraints may be associated with GroundMotions.
    See https://openseespydoc.readthedocs.io/en/latest/src/pattern.html
    * patternType (str) : pattern type.
    * patternTag (int) : pattern tag.
    * patternArgs (list) : a list of pattern arguments.

    The following contain information about available patternType:

    ``pattern('Plain', patternTag, tsTag, '-fact', fact)``

    ``pattern('UniformExcitation', patternTag, dir, '-disp', dispSeriesTag,
    '-vel', velSeriesTag, '-accel', accelSeriesTag, '-vel0', vel0, '-fact', fact)``

    ``pattern('MultipleSupport', patternTag)``
    """
    pass

def load(*args) -> None:
    """``load(nodeTag, *loadValues)``

    This command is used to construct a NodalLoad object and add it to the enclosing LoadPattern.
    see https://openseespydoc.readthedocs.io/en/latest/src/load.html

    * nodeTag (int) : tag of node to which load is applied.
    * loadValues (list (float)) : ndf reference load values.

    Note:

        The load values are reference loads values. It is the time series that provides the load factor.
        The load factor times the reference values is the load that is actually applied to the node.
    """
    pass

def eleLoad(*args) -> None:
    """
    ``eleLoad('-ele', *eleTags, '-range', eleTag1, eleTag2, '-type',
    '-beamUniform', Wy, <Wz>, Wx=0.0, '-beamPoint', Py, <Pz>, xL, Px=0.0, '-beamThermal', *tempPts)``

    The eleLoad command is used to construct an ElementalLoad object and add it to the enclosing LoadPattern.
    See https://openseespydoc.readthedocs.io/en/latest/src/eleload.html

    * eleTags (list (int)) : tag of PREVIOUSLY DEFINED element
    * eleTag1 (int) : element tag
    * eleTag2 (int) : element tag
    * Wx (float) : mag of uniformily distributed ref load acting in direction along member length. (optional)
    * Wy (float) : mag of uniformily distributed ref load acting in local y direction of element
    * Wz (float) : mag of uniformily distributed ref load acting in local z direction of element. (required only for 3D)
    * Px (float) : mag of ref point load acting in direction along member length. (optional)
    * Py (float) : mag of ref point load acting in local y direction of element
    * Pz (float) : mag of ref point load acting in local z direction of element. (required only for 3D)
    * xL (float) : location of point load relative to node I, prescribed as fraction of element length
    * tempPts (list (float)) : temperature points: temPts = [T1, y1, T2, y2, ..., T9, y9].
        Each point (T1, y1) define a temperature and location. This command may accept 2,5 or 9 temperature points.

    Note:
    * The load values are reference load values, it is the time series that provides the load factor.
        The load factor times the reference values is the load that is actually applied to the element.
    * At the moment, eleLoads do not work with 3D beam-column elements if Corotational geometric transformation is used.
    """
    pass

def sp(*args) -> None:
    """``sp(nodeTag, dof, dofValue)``

    This command is used to construct a single-point constraint object and add it to the enclosing LoadPattern.
    See https://openseespydoc.readthedocs.io/en/latest/src/sp.html

    * nodeTag (int) : tag of node to which load is applied.
    * dof (int) : the degree-of-freedom at the node to which constraint is applied (1 through ndf)
    * dofValue (float) : reference constraint value.

    Note:
    * The dofValue is a reference value, it is the time series that provides the load factor.
    The load factor times the reference value is the constraint that is actually applied to the node.
    """
    pass

def groundMotion(*args) -> None:
    """``groundMotion(gmTag, 'Plain', '-disp', dispSeriesTag, '-vel',
    velSeriesTag, '-accel', accelSeriesTag, '-int', tsInt='Trapezoidal', '-fact', factor=1.0)``
    ----> Plain Ground Motion, see https://openseespydoc.readthedocs.io/en/latest/src/groundMotion.html

    ``groundMotion(gmTag, 'Interpolated', *gmTags, '-fact', facts)``
    ----> Interpolated Ground Motion, see https://openseespydoc.readthedocs.io/en/latest/src/interpolatedGroundMotion.html
    """
    pass

def imposedMotion(*args) -> None:
    """``imposedMotion(nodeTag, dof, gmTag)``

    This command is used to construct an ImposedMotionSP constraint which
    is used to enforce the response of a dof at a node in the model.
    The response enforced at the node at any give time is obtained from the GroundMotion object associated with the constraint.
    See https://openseespydoc.readthedocs.io/en/latest/src/imposedMotion.html

    * nodeTag (int) : tag of node on which constraint is to be placed.
    * dof (int) : dof of enforced response. Valid range is from 1 through ndf at node.
    * gmTag (int) : pre-defined GroundMotion object tag.
    """
    pass

def element(*args) -> None:
    """``element(eleType, eleTag, *eleNodes, *eleArgs)``

    Create a OpenSees element,
    see https://openseespydoc.readthedocs.io/en/latest/src/element.html.

    * eleType (str) : element type.
    * eleTag (int) : element tag.
    * eleNodes (list (int)) : a list of element nodes, must be preceded with *.
    * eleArgs (list) : a list of element arguments, must be preceded with *.

    ``element('zeroLength', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dirs,
    <'-doRayleigh', rFlag=0>, <'-orient', *vecx, *vecyp>)``

    ``element('zeroLengthSection', eleTag, *eleNodes, secTag, <'-orient', *vecx, *vecyp>, <'-doRayleigh', rFlag>)``

    ``element('twoNodeLink', eleTag, *eleNodes, '-mat', *matTags, '-dir', *dir, <'-orient', *vecx, *vecyp>,
    <'-pDelta', *pDeltaVals>, <'-shearDist', *shearDist>, <'-doRayleigh'>, <'-mass', m>)``

    ``element('Truss', eleTag, *eleNodes, A, matTag, <'-rho', rho>, <'-cMass', cFlag>, <'-doRayleigh', rFlag>)``

    ``element('TrussSection', eleTag, *eleNodes, secTag, <'-rho', rho>, <'-cMass', cFlag>, <'-doRayleigh', rFlag>)``

    ``element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, Iz, transfTag,
    <'-mass', mass>, <'-cMass'>, <'-release', releaseCode>)`` For 2D

    ``element('elasticBeamColumn', eleTag, *eleNodes, secTag, transfTag,
    <'-mass', mass>, <'-cMass'>, <'-release', releaseCode>)`` For 2D

    ``element('elasticBeamColumn', eleTag, *eleNodes, Area, E_mod, G_mod, Jxx, Iy, Iz, transfTag,
    <'-mass', mass>, <'-cMass'>)`` For 3D

    ``element('elasticBeamColumn', eleTag, *eleNodes, secTag, transfTag,
    <'-mass', mass>, <'-cMass'> <'-releasez', releaseCode>, <'-releasey', releaseCode>)`` For 3D

    ``element('dispBeamColumn', eleTag, *eleNodes, transfTag, integrationTag, '-cMass', '-mass', mass=0.0)``

    ``element('forceBeamColumn', eleTag, *eleNodes, transfTag, integrationTag, '-iter', maxIter=10, tol=1e-12, '-mass', mass=0.0)``
    """
    pass

def constraints(*args) -> None:
    """``constraints(constraintType, *constraintArgs)``

    This command is used to construct the ConstraintHandler object. The ConstraintHandler object determines how the constraint equations are enforced in the analysis. Constraint equations enforce a specified value for a DOF, or a relationship between DOFs.
    See https://openseespydoc.readthedocs.io/en/latest/src/constraints.html

    * constraintType (str) : constraints type
    * constraintArgs (list) : a list of constraints arguments

    ``constraints('Plain')``

    ``constraints('Lagrange', alphaS=1.0, alphaM=1.0)``

    ``constraints('Penalty', alphaS=1.0, alphaM=1.0)``

    ``constraints('Transformation')``
    """
    pass

def numberer(*args) -> None:
    """``numberer(numbererType, *numbererArgs)``

    This command is used to construct the DOF_Numberer object.
    The DOF_Numberer object determines the mapping between equation numbers and
    degrees-of-freedom – how degrees-of-freedom are numbered.
    See https://openseespydoc.readthedocs.io/en/latest/src/numberer.html

    * numbererType (str) : numberer type
    * numbererArgs (list) : a list of numberer arguments

    ``numberer('Plain')``

    ``numberer('RCM')``

    ``numberer('AMD')``

    ``numberer('ParallelPlain')``

    ``numberer('ParallelRCM')``
    """
    pass

def system(*args) -> None:
    """``system(systemType, *systemArgs)``

    This command is used to construct the LinearSOE and LinearSolver
    objects to store and solve the system of equations in the analysis.
    See https://openseespydoc.readthedocs.io/en/latest/src/system.html

    * systemType (str) : system type
    * systemArgs (list) : a list of system arguments

    ``system('BandGen')``

    ``system('BandSPD')``

    ``system('ProfileSPD')``

    ``system('SuperLU')``

    ``system('UmfPack')``

    ``system('SparseSYM')``
    """
    pass

def test(*args) -> None:
    """``test(testType, *testArgs)``

    This command is used to construct the LinearSOE and LinearSolver objects to store and solve the test of equations in the analysis.
    See https://openseespydoc.readthedocs.io/en/latest/src/test.html

    * testType (str) : test type
    * testArgs (list) : a list of test arguments

    ``test('NormUnbalance', tol, iter, pFlag=0, nType=2, maxIncr=maxIncr)``

    ``test('NormDispIncr', tol, iter, pFlag=0, nType=2)``

    ``test('EnergyIncr', tol, iter, pFlag=0, nType=2)``
    """
    pass

def algorithm(*args) -> None:
    """``algorithm(algoType, *algoArgs)``

    This command is used to construct a SolutionAlgorithm object, which determines the
    sequence of steps taken to solve the non-linear equation.
    See https://openseespydoc.readthedocs.io/en/latest/src/algorithm.html

    * algoType (str) : algorithm type
    * algoArgs (list) : a list of algorithm arguments

    ``algorithm("Linear", <"-secant">, <"-initial">, <"-factorOnce">)``

    ``algorithm("Newton", <"-Initial",> <"-intialThenCurrent",> <"-Hall", iFactor(0.1), cFactor(0.9)>)``

    ``algorithm("ModifiedNewton", <"-secant",> <"-factoronce",> <"-initial",> <"-Hall", iFactor(0.1), cFactor(0.9)>)``

    ``algorithm("NewtonLineSearch", <"-type", typeSearch("InitialInterpolated"),> <"-tol", tol(0.8),>
    <"-maxIter", maxIter(10),> <"-minEta", minEta(0.1),> <"-maxEta", maxEta(10.0)>, <"-pFlag", pFlag(1))``

    ``algorithm("KrylovNewton", <"-iterate", tangIter("current"),> <"-increment", tangIncr("current"),> <"-maxDim", maxDim(3)>)``

    ``algorithm("SecantNewton", <"-iterate", tangIter("current"),> <"-increment", tangIncr("current"),> <"-maxDim", maxDim(3),>
    <"-numTerms", numTerms(2),> <"-cutOut", R1(3.5), R2(0.3)>)``

    ``algorithm("BFGS", <"-initial",> <"-count", count(10)>)``

    ``algorithm("Broyden", <"-initial",> <"-count", count=10>)``

    ``algorithm("ExpressNewton", <nIter(2)>, <kMultiplier(1.0)>, <"-initialTangent">, <"-currentTangent">, <"-factorOnce">)``
    """

    pass

def integrator(*args) -> None:
    """``integrator(intType, *intArgs)``
    This command is used to construct the Integrator object.
    The Integrator object determines the meaning of the terms in the system of equation object Ax=B.
    See https://openseespydoc.readthedocs.io/en/latest/src/integrator.html

    * intType (str) : integrator type
    * intArgs (list) : a list of integrator arguments

    Static integrator objects
    -------------------------
    ``integrator('LoadControl', incr, numIter=1, minIncr=incr, maxIncr=incr)``

    ``integrator('DisplacementControl', nodeTag, dof, incr, numIter=1, dUmin=incr, dUmax=incr)``

    ``integrator('ParallelDisplacementControl', nodeTag, dof, incr, numIter=1, dUmin=incr, dUmax=incr)``

    ``integrator('MinUnbalDispNorm', dlambda1, Jd=1, minLambda=dlambda1, maxLambda=dlambda1, det=False)``

    ``integrator('ArcLength', s, alpha)``

    Transient integrator objects
    ------------------------------
    ``integrator('CentralDifference')``

    ``integrator('Newmark', gamma, beta, '-form', form)``

    ``integrator('HHT', alpha, gamma=1.5-alpha, beta=(2-alpha)^2/4)``

    ``integrator('GeneralizedAlpha', alphaM, alphaF, gamma=0.5+alphaM-alphaF, beta=(1+alphaM-alphaF)^2/4)``

    ``integrator('ExplicitDifference')``
    """
    pass

def analysis(analysisType: str) -> None:
    """``analysis(analysisType)``

    See https://openseespydoc.readthedocs.io/en/latest/src/analysis.html.
    This command is used to construct the Analysis object, which defines what type of analysis is to be performed.

    1. determine the predictive step for time t+dt
    2. specify the tangent matrix and residual vector at any iteration
    3. determine the corrective step based on the displacement increment dU

    * analysisType (str) : char string identifying type of analysis object to be constructed.
        Currently 3 valid options:

        ``'Static'`` - for static analysis;

        ``'Transient'`` - for transient analysis constant time step;

        ``'VariableTransient'`` - for transient analysis with variable time step;

        ``'PFEM'`` - for PFEM analysis.

    Note:

    If the component objects are not defined before hand,
    the command automatically creates default component objects and issues warning messages to this effect.
    The number of warning messages depends on the number of component objects that are undefined.
    """
    pass

def eigen(*args) -> list:
    """``eigen(<solver='-genBandArpack'>, numEigenvalues)``

    Eigen value analysis. Return a list of eigen values.
    See https://opensees.berkeley.edu/wiki/index.php/Eigen_Command

    * numEigenvalues (int) : number of eigenvalues required
    * solver (str) : optional string detailing type of solver: '-genBandArpack', '-fullGenLapack', (optional)

    Note

    * The eigenvectors are stored at the nodes and can be printed out using a Node Recorder,
        the nodeEigenvector command, or the Print command.
    * The default eigensolver is able to solve only for N-1 eigenvalues, where N is
        the number of inertial DOFs. When running into this limitation the -fullGenLapack
        solver can be used instead of the default Arpack solver.
    * The -fullGenLapack option is VERY SLOW for moderate to large models
    """
    pass

def analyze(*args) -> int:
    """``analyze(numIncr=1, dt=0.0, dtMin=0.0, dtMax=0.0, Jd=0)``

    Perform the analysis. Return 0 if successful, <0 if NOT successful
    See https://openseespydoc.readthedocs.io/en/latest/src/analyze.html

    * numIncr (int) : Number of analysis steps to perform. (required except for PFEM analysis)
    * dt (float) : Time-step increment. (required for Transient analysis and VariableTransient analysis.`)
    * dtMin (float) : Minimum time steps. (required for VariableTransient analysis)
    * dtMax (float) : Maximum time steps (required for VariableTransient analysis)
    * Jd (float) : Number of iterations user would like performed at each step.
        The variable transient analysis will change current time step if last analysis step
        took more or less iterations than this to converge (required for VariableTransient analysis)
    """
    pass

def modalProperties(*args) -> dict:
    """``modalProperties(<'-print'>, <'-file', reportFileName>, <'-unorm'>, <'-return'>)``

    See https://openseespydoc.readthedocs.io/en/latest/src/modalProperties.html

    * '-print' : (str) : Optional. If included, a report of the modal properties is printed to the console.
    * '-file' : (str) : Optional. If included, a report of the modal properties is printed to the file reportFileName.
    * reportFileName : (str) : Optional, but mandatory if the -file option is included. Indicates the filename for the report.
        If the file does not exist, it will be created. If the file exists, it will be overwritten.
    * '-unorm' : (str) : Optional. If included, the computation of the modal properties will be carried out using
        a displacement-normalized version of the eigenvectors.
    * '-return' : (str) : Optional. If included, a report of the modal properties will be returned as a dict object to Python.
    """
    pass

def responseSpectrumAnalysis(*args) -> None:
    """This command is used to perform a response spectrum analysis.
    The response spectrum analysis performs N linear analysis steps, where N is the number of eigenvalues
    requested in a previous call to eigen command.
    For each analysis step, it computes the modal displacements. When the i-th analysis step is complete,
    all previously defined recorders will be called, so they will record all the results requested by the user,
    pertaining to the current modal displacements.
    The modal combination of these modal displacements (and derived results such as beam forces) is up to the user, and
    can be easily done via Python scripting.

    The command can be called in two different ways, depending on how you store the Tn/Sa (response spectrum function) values.
    They can be either stored in a timeSeries …

    See https://openseespydoc.readthedocs.io/en/latest/src/responseSpectrumAnalysis.html

    ``responseSpectrumAnalysis(tsTag, direction, <'-scale', scale>, <'-mode', mode>)``

    or in two lists

    ``responseSpectrumAnalysis(direction, '-Tn', Tn, '-Sa', Sa, <'-scale ', scale>, <'-mode', mode>)``
    """
    pass

def wipe() -> None:
    """This command is used to destroy all constructed objects, i.e. all components of the model,
    all components of the analysis and all recorders.

    This command is used to start over without having to exit and restart the interpreter.
    It causes all elements, nodes, constraints, loads to be removed from the domain.
    In addition it deletes all recorders, analysis objects and all material objects created by the model builder.
    """
    pass

def wipeAnalysis() -> None:
    """This command is used to destroy all components of the Analysis object, i.e.
    any objects created with system, numberer, constraints, integrator, algorithm, and analysis commands.
    """
    pass

# ------------- OUTPUT------------
def basicDeformation(eleTag: int) -> list:
    """Returns the deformation of the basic system for a beam-column element.

    * eleTag (int) : element tag.
    """
    pass

def basicForce(eleTag: int) -> list:
    """Returns the forces of the basic system for a beam-column element.

    * eleTag (int) : element tag.
    """
    pass

def basicStiffness(eleTag: int) -> list:
    """Returns the stiffness of the basic system for a beam-column element.
    A list of values in row order will be returned.

    * eleTag (int) : element tag.
    """
    pass

def eleDynamicalForce(eleTag: int, dof: int = -1) -> Union[list, float]:
    """``eleDynamicalForce(eleTag, dof)``

    Returns the elemental dynamic force.

    * eleTag (int) : element tag.
    * dof (int) : specific dof at the element, (optional), if no dof is provided, a list of values for all dofs is returned.
    """
    pass

def eleForce(eleTag: int, dof: int = -1) -> Union[list, float]:
    """Returns the elemental resisting force.

    * eleTag (int) : element tag.
    * dof (int) : specific dof at the element, (optional), if no dof is provided, a list of values for all dofs is returned.
    """
    pass

def eleNodes(eleTag: int) -> list:
    """Get nodes in an element.

    * eletag (int) : element tag.
    """
    pass

def eleResponse(eleTag: int, *args) -> list:
    """This command is used to obtain the same element quantities as those obtained from the element recorder at a particular time step.

    * eletag (int) : element tag.
    * args (list) : same arguments as those specified in element recorder.
        These arguments are specific to the type of element being used.
    """
    pass

def getEleTags(*args) -> list:
    """``getEleTags('-mesh', mtag)``

    Get all elements in the domain or in a mesh.

    * mtag (int) : mesh tag. (optional)
    """
    pass

def getLoadFactor(patternTag: int) -> float:
    """
    Returns the load factor λ for the pattern with patternTag

    * patternTag (int) : pattern tag.
    """
    pass

def getNodeTags(*args) -> list:
    """getNodeTags('-mesh', mtag)
    Get all nodeTags in the domain or in a mesh.

    * mtag (int) : mesh tag. (optional)
    """
    pass

def getTime() -> float:
    """Returns the current time in the domain."""
    pass

def nodeAccel(nodeTag: int, dof: int = -1) -> Union[list, float]:
    """Returns the current acceleration at a specified node.

    * nodeTag (int) : node tag.
    * dof (int) : specific dof at the node (1 through ndf), (optional),
    if no dof is provided, a list of values for all dofs is returned.
    """
    pass

def nodeVel(nodeTag: int, dof: int = -1) -> Union[list, float]:
    """Returns the current velocity at a specified node.

    * nodeTag (int) : node tag.
    * dof (int) : specific dof at the node (1 through ndf), (optional),
    if no dof is provided, a list of values for all dofs is returned.
    """
    pass

def nodeDisp(nodeTag: int, dof: int = -1) -> Union[list, float]:
    """Returns the current displacement at a specified node.

    * nodeTag (int) : node tag.
    * dof (int) : specific dof at the node (1 through ndf), (optional),
    if no dof is provided, a list of values for all dofs is returned.
    """
    pass

def nodeBounds() -> list:
    """Get the boundary of all nodes. Return a list of boundary values"""
    pass

def nodeCoord(*args) -> list:
    """``nodeCoord(nodeTag, dim=-1)``
    Returns the coordinates of a specified node.

    * nodeTag (int) : node tag.
    * dim (int) : specific dimension at the node (1 through ndm), (optional),
        if no dim is provided, a list of values for all dimensions is returned.
    """
    pass

def nodeEigenvector(*args) -> Union[list, float]:
    """``nodeEigenvector(nodeTag, eigenvector, dof=-1)``

    Returns the eigenvector at a specified node.

    * nodeTag (int) : node tag.
    * eigenvector (int) : mode number of eigenvector to be returned
    * dof (int) : specific dof at the node (1 through ndf), (optional), if no dof is provided, a list of values for all dofs is returned.
    """
    pass

def nodeDOFs(nodeTag: int) -> list:
    """Returns the DOF numbering of a node.

    * nodeTag (int) : node tag.
    """
    pass

def nodeMass(nodeTag: int, dof: int = -1) -> Union[list, float]:
    """``nodeMass(nodeTag, dof=-1)``

    Returns the mass at a specified node.

    * nodeTag (int) : node tag.
    * dof (int) : specific dof at the node (1 through ndf), (optional), if no dof is provided,
        a list of values for all dofs is returned.
    """
    pass

def nodePressure(nodeTag: int) -> list:
    """
    Returns the fluid pressures at a specified node if this is a fluid node.

    * nodeTag (int) : node tag.
    """
    pass

def nodeReaction(nodeTag: int, dof: int = -1) -> Union[list, float]:
    """Returns the reactions at a specified node.
    Must call reactions() command before this command.

    * nodeTag (int) : node tag.
    * dof (int) : specific dof at the node (1 through ndf), (optional),
        if no dof is provided, a list of values for all dofs is returned.
    """
    pass

def nodeResponse(nodeTag: int, dof: int, responseID: int) -> float:
    """Returns the responses at a specified node.
    To get reactions (id=6), must call the reactions command before this command.

    * nodeTag (int) : node tag.
    * dof (int) : specific dof of the response
    * responseID (int) : the id of responses:

    1. Disp = 1
    2. Vel = 2
    3. Accel = 3
    4. IncrDisp = 4
    5. IncrDeltaDisp = 5
    6. Reaction = 6
    7. Unbalance = 7
    8. RayleighForces = 8
    """
    pass

def nodeUnbalance(nodeTag: int, dof: int = -1) -> Union[list, float]:
    """Returns the unbalanced force at a specified node.

    * nodeTag (int) : node tag.
    * dof (int) : specific dof at the node (1 through ndf), (optional),
        if no dof is provided, a list of values for all dofs is returned.
    """
    pass

def numFact() -> int:
    """Return the number of factorizations."""
    pass

def numIter() -> int:
    """Return the number of iterations."""
    pass

def printA(*args) -> Union[list, None]:
    """``printA('-file', filename, '-ret')``

    print the contents of a FullGeneral system that the integrator creates to the screen or a file if the '-file' option is used.
    If using a static integrator, the resulting matrix is the stiffness matrix.
    If a transient integrator, it will be some combination of mass and stiffness matrices.
    The printA command can only be issued after an analyze command.

    * filename (str) : name of file to which output is sent, by default, print to the screen. (optional)
    * '-ret' (str) : return the A matrix as a list. (optional)
    """
    pass

def printB(*args) -> Union[list, None]:
    """``printA('-file', filename, '-ret')``

    print the right hand side of a FullGeneral system that the integrator creates to the screen or a file if the '-file' option is used.

    * filename (str) : name of file to which output is sent, by default, print to the screen. (optional)
    * '-ret' (str) : return the A matrix as a list. (optional)
    """
    pass

def printModel(*args) -> None:
    """``printModel('-JSON', '-file', filename, '-node', '-flag', flag, *nodes=[], *eles=[])``

    This command is used to print output to screen or file.

    * filename (str) : name of file to which output is sent, by default, print to the screen. (optional)
    * '-JSON' (str) : print to a JSON file. (optional)
    * '-node' (str) : print node information. (optional)
    * flag (int) : integer flag to be sent to the print() method, depending on the node and element type (optional)
    * nodes (list (int)) : a list of nodes tags to be printed, default is to print all, (optional)
    * eles (list (int)) : a list of element tags to be printed, default is to print all, (optional)

    Note

    * This command was called print in Tcl. Since print is a built-in function in Python, it is renamed to printModel.
    """
    pass

def record() -> None:
    """This command is used to cause all the recorders to do a record on the current state of the model.

    Note

    A record is issued after every successfull static or transient analysis step.
    Sometimes the user may need the record to be issued on more occasions than this,
    for example if the user is just looking to record the eigenvectors after an eigen command or
    for example the user wishes to include the state of the model at time 0.0 before any analysis has been completed.
    """
    pass

def recorder(*args) -> int:
    """``recorder(recorderType, *recorderArgs)``

    This command is used to generate a recorder object which is to monitor what is happening
    during the analysis and generate output for the user.
    See https://openseespydoc.readthedocs.io/en/latest/src/recorder.html

    Return
    -------

    * >0 an integer tag that can be used as a handle on the recorder for the ``remove`` recorder commmand.
    * -1 recorder command failed if integer -1 returned.

    Args
    -----
    recorderType (str) : recorder type
    recorderArgs (list) : a list of recorder arguments

    ``recorder('Node', '-file', filename, '-xml', filename, '-binary', filename,
    '-tcp', inetAddress, port, '-precision', nSD=6, '-timeSeries', tsTag, '-time', '-dT', deltaT=0.0,
    '-closeOnWrite', '-node', *nodeTags=[], '-nodeRange', startNode, endNode,
    '-region', regionTag, '-dof', *dofs=[], respType)``

    ``recorder('EnvelopeNode', '-file', filename, '-xml', filename, '-precision', nSD=6,
    '-timeSeries', tsTag, '-time', '-dT', deltaT=0.0, '-closeOnWrite', '-node', *nodeTags=[],
    '-nodeRange', startNode, endNode, '-region', regionTag, '-dof', *dofs=[], respType)``

    ``recorder('Element', '-file', filename, '-xml', filename, '-binary', filename, '-precision', nSD=6,
    '-timeSeries', tsTag, '-time', '-dT', deltaT=0.0, '-closeOnWrite', '-ele', *eleTags=[],
    '-eleRange', startEle, endEle, '-region', regionTag, *args)``

    ``recorder('EnvelopeElement', '-file', filename, '-xml', filename, '-binary', filename, '-precision', nSD=6,
    '-timeSeries', tsTag, '-time', '-dT', deltaT=0.0, '-closeOnWrite', '-ele', *eleTags=[],
    '-eleRange', startEle, endEle, '-region', regionTag, *args)``

    ``recorder('PVD', filename, '-precision', precision=10, '-dT', dT=0.0, *res)``

    ``recorder('BgPVD', filename, '-precision', precision=10, '-dT', dT=0.0, *res)``

    ``recorder('Collapse', '-node', nodeTag, '-file_infill', fileNameinf, '-checknodes', nTagbotn, nTagmidn, nTagtopn,
    '-global_gravaxis', globgrav, '-secondary', '-eles', *eleTags, '-eleRage', start, end,
    '-region', regionTag, '-time', '-dT', dT, '-file', fileName, '-mass', *massValues,
    '-g', gAcc, gDir, gPat, '-section', *secTags, '-crit', critType, critValue)``
    """
    pass

def sectionForce(eleTag: int, secNum: int, dof: int = -1) -> Union[list, float]:
    """Returns the section force for a beam-column element.
    The dof of the section depends on the section type. Please check with the section manual.

    * eleTag (int) : element tag.
    * secNum (int) : section number, i.e. the Gauss integratio number
    * dof (int) : the dof of the section
    """
    pass

def sectionDeformation(eleTag: int, secNum: int, dof: int = -1) -> Union[list, float]:
    """Returns the section deformation for a beam-column element.
    The dof of the section depends on the section type. Please check with the section manual.

    * eleTag (int) : element tag.
    * secNum (int) : section number, i.e. the Gauss integratio number
    * dof (int) : the dof of the section
    """
    pass

def sectionStiffness(eleTag: int, secNum: int) -> list:
    """Returns the section stiffness matrix for a beam-column element.
    A list of values in the row order will be returned.

    * eleTag (int) : element tag.
    * secNum (int) : section number, i.e. the Gauss integratio number
    """
    pass

def sectionFlexibility(eleTag: int, secNum: int) -> list:
    """Returns the section flexibility matrix for a beam-column element.
    A list of values in the row order will be returned.

    * eleTag (int) : element tag.
    * secNum (int) : section number, i.e. the Gauss integratio number
    """
    pass

def sectionLocation(eleTag: int, secNum: int = 0) -> Union[list, float]:
    """Returns the locations of integration points of a section for a beam-column element.

    * eleTag (int) : element tag.
    * secNum (int) : section number, i.e. the Gauss integratio number.
        If 0, return all sections.
    """
    pass

def sectionWeight(eleTag: int, secNum: int = 0) -> Union[list, float]:
    """Returns the weights of integration points of a section for a beam-column element.

    * eleTag (int): element tag.
    * secNum (int): section number, i.e., the Gauss integratio number.
        If 0, return all sections.
    """
    pass

def systemSize() -> int:
    """Return the size of the system."""
    pass

def testIter() -> int:
    """Returns the number of iterations the convergence test took in the last analysis step"""
    pass

def testNorm() -> list:
    """Returns the norms from the convergence test for the last analysis step.

    Note
    ------
    The size of norms will be equal to the max number of iterations specified.
    The first testIter of these will be non-zero, the remaining ones will be zero.
    """
    pass

def testNorms() -> list:
    """Returns the norms from the convergence test for the last analysis step.

    Note
    ------
    The size of norms will be equal to the max number of iterations specified.
    The first testIter of these will be non-zero, the remaining ones will be zero.
    """
    pass

def version() -> str:
    """Return the current OpenSees version."""
    pass

def testUniaxialMaterial(tag: int) -> None:
    """
    Define the test uniaxial material.

    Parameters
    -----------
    tag: int, the uniaxial material tag defined previously.
    """
    pass

def setStrain(strain: float, strainRate: float = 0.0) -> None:
    """Set the stain of the test uniaxial material."""
    pass

def getStrain() -> float:
    """Get the stain of the test uniaxial material."""
    pass

def getStress() -> float:
    """Get the stress of the test uniaxial material."""
    pass

def getTangent() -> float:
    """Get the tangent of the test uniaxial material."""
    pass

def getDampTangent() -> float:
    """Get the damp tangent of the test uniaxial material."""
    pass

def getPatterns() -> list:
    """Return all pattern tags."""
    pass

def getFixedNodes() -> list:
    """Return all fixed node tags."""
    pass

def getFixedDOFs(nodeTag: int) -> list:
    """Return fixed dofs of the node."""
    pass

def getConstrainedNodes(*args) -> list:
    """``getConstrainedNodes(<rNodeTag>)``

    * rNodeTag: int, retained node tag, optional,
        if not input, all constrained nodes tags will return.
    """
    pass

def getConstrainedDOFs(*args) -> list:
    """``getConstrainedDOFs(cNode, <rNode>, <rDOF>)``"""
    pass

def getRetainedNodes(*args) -> list:
    """``getRetainedNodes(<cNodeTag>)``"""
    pass

def getRetainedDOFs(*args) -> list:
    """``getRetainedDOFs(rNode, <cNode>, <cDOF>)``"""
    pass

def updateElementDomain() -> None:
    """ """
    pass

def updateMaterialStage(*args) -> None:
    """``updateMaterialStage('-material', matTag, '-stage', value, <'-parameter', paramTag>)``

    This function is used in geotechnical modeling to maintain elastic nDMaterial response during the application of gravity loads.
    The material is then updated to allow for plastic strains during additional static loads or earthquakes.
    See https://openseespydoc.readthedocs.io/en/latest/src/updateMaterialStage.html

    * matTag (int) : tag of nDMaterial.
    * value (int) : stage value.
    *paramTag (int) : tag of parameter (optional).
    """
    pass

def getNDM(*args) -> Union[list, int]:
    """``getNDM(<nodeTag>)``

    * nodeTag: int, optional.
    """
    pass

def getNDF(*args) -> Union[list, int]:
    """``getNDM(<nodeTag>)``

    * nodeTag: int, optional.
    """
    pass

def eleType(eleTag: int) -> str:
    """ """
    pass

def getCrdTransfTags() -> list:
    """getCrdTransfTags"""
    pass

def getNumElements() -> int:
    """getNumElements"""
    pass

def getEleClassTags(*args) -> Union[list, int]:
    """``getEleClassTags(<eleTag>)``

    * eleTag: int, optional
    """
    pass

def getEleLoadClassTags(*args) -> list:
    """``getEleLoadClassTags(<patternTag>)``

    *patternTag: int, optional
    """
    pass

def getEleLoadTags(*args) -> list:
    """``getEleLoadTags(<patternTag>)``

    *patternTag: int, optional
    """
    pass

def getEleLoadData(*args) -> list:
    """``getEleLoadData(<patternTag>)``

    *patternTag: int, optional
    """
    pass

def getNodeLoadTags(*args) -> list:
    """``getNodeLoadTags(<patternTag>)``

    *patternTag: int, optional
    """
    pass

def getNodeLoadData(*args) -> list:
    """``getNodeLoadData(<patternTag>)``

    *patternTag: int, optional
    """
    pass

# ------------
def loadConst(*args) -> None:
    """``loadConst('-time', pseudoTime)``

    This command is used to set the loads constant in the domain and to also set the time in the domain.
    When setting the loads constant,
    the procedure will invoke setLoadConst() on all LoadPattern
    objects which exist in the domain at the time the command is called.

    pseudoTime (float) : Time domain is to be set to (optional)

    Note
    -----
    Load Patterns added afer this command is invoked are not set to constant.
    """
    pass

def modalDamping(*args) -> None:
    """``modalDamping(*factor)``

    Set modal damping factor. The eigen() must be called before.

    * factor (list) : damping factor.
    """
    pass

def reactions(*args) -> None:
    """``reactions('-dynamic', '-rayleigh')``

    Calculate the reactions. Call this command before the nodeReaction().

    * '-dynamic' (str) : Include dynamic effects.
    *'-rayleigh' (str) : Include rayleigh damping.
    """
    pass

def remove(*args) -> None:
    """``remove(type, tag)``

    This commmand is used to remove components from the model.
    See https://openseespydoc.readthedocs.io/en/latest/src/remove.html

    * type (str) : type of the object, 'ele', 'loadPattern', 'parameter', 'node', 'timeSeries', 'sp', 'mp'.
    * tag (int) : tag of the object

    ``remove('recorders')`` ----> Remove all recorder objects.

    ``remove('sp', nodeTag, dofTag, patternTag)`` ----> Remove a sp object based on node
    """
    pass

def reset() -> None:
    """This command is used to set the state of the domain to its original state.

    Note
    -----
    It iterates over all components of the domain telling them to set their state back to the initial state.
    This is not always the same as going back to the state of the model after initial model generation, e.g.
    if elements have been removed
    """
    pass

def sdfResponse(*args) -> list:
    """``sdfResponse(m, zeta, k, Fy, alpha, dtF, filename, dt[, uresidual, umaxprev])``

    It is a command that computes bilinear single degree of freedom response in C++,
    and is much quicker than using the OpenSees model builder.
    The command implements Newmark’s method with an inner Newton loop.

    See https://openseespydoc.readthedocs.io/en/latest/src/sdfResponse.html

    Args
    -----
    * m (float) : mass
    * zeta (float) : damping ratio
    * k (float) : stiffness
    * Fy (float) : yielding strength
    * alpha (float) : strain-hardening ratio
    * dtF (float) : time step for input data
    * filename (str) : input data file, one force per line
    * dt (float) : time step for analysis
    * uresidual (float) : residual displacement at the end of previous analysis (optional, default=0)
    * umaxprev (float) : previous displacement (optional, default=0),

    The command returns a list of five response quantities.

    Returns
    -----------
    * umax (float) : maximum displacement during analysis
    * u (float) : displacement at end of analysis
    * up (float) : permanent residual displacement at end of analysis
    * amax (float) : maximum acceleration during analysis
    * tamax (float) : time when maximum accleration occurred
    """
    pass

def database(type: str, dbName: str) -> None:
    """Create a database.

    * type (str) : database type:

        'File' - outputs database into a file;

        'MySQL' - creates a SQL database;

        'BerkeleyDB' - creates a BerkeleyDB database.

    * dbName (str) : database name.
    """
    pass

def restore(commitTag: int) -> None:
    """Restore data from database, which should be created through database().

    * commitTag (int) : a tag identify the commit
    """
    pass

def save(commitTag: int) -> None:
    """Save current state to database, which should be created through database().

    * commitTag (int) : a tag identify the commit
    """
    pass

def InitialStateAnalysis(flag: str) -> None:
    """Set the initial state analysis to 'on' or 'off'

    * flag (str) : 'on' or 'off'.
    """
    pass

def setTime(pseudoTime: float) -> None:
    """This command is used to set the time in the Domain.

    * pseudoTime (float) : Time domain to be set.
    """
    pass

def setNodeCoord(nodeTag: int, dim: int, value: float) -> None:
    """set the nodal coodinate at the specified dimension.

    * nodeTag (int) : node tag.
    * dim (int) : the dimension of the coordinate to be set.
    * value (float) : coordinate value
    """
    pass

def setNodeDisp(*args) -> None:
    """``setNodeDisp(nodeTag, dof, value, <'-commit'>)``

    set the nodal displacement at the specified DOF.

    * nodeTag (int) : node tag.
    * dof (int) : the DOF of the displacement to be set.
    * value (float) : displacement value
    * '-commit' (str) : commit nodal state. (optional)
    """
    pass

def setNodeVel(*args) -> None:
    """``setNodeVel(nodeTag, dof, value, <'-commit'>)``

    set the nodal velocity at the specified DOF.

    * nodeTag (int) : node tag.
    * dof (int) : the DOF of the velocity to be set.
    * value (float) : velocity value
    * '-commit' (str) : commit nodal state. (optional)
    """
    pass

def setNodeAccel(*args) -> None:
    """``setNodeAccel(nodeTag, dof, value, <'-commit'>)``

    set the nodal acceleration at the specified DOF.

    * nodeTag (int) : node tag.
    * dof (int) : the DOF of the acceleration to be set.
    * value (float) : acceleration value
    * '-commit' (str) : commit nodal state. (optional)
    """
    pass

def setPrecision(precision: int) -> None:
    """Set the precision for screen output.

    * precision (int) : the precision number.
    """
    pass

def setElementRayleighDampingFactors(eleTag: int, alphaM: float, betaK: float, betaK0: float, betaKc: float) -> None:
    """Set the rayleigh() damping for an element.

    * eleTag (int) : element tag.
    * alphaM (float) : factor applied to elements or nodes mass matrix
    * betaK (float) : factor applied to elements current stiffness matrix.
    * betaK0 (float) : factor applied to elements initial stiffness matrix.
    * betaKc (float) : factor applied to elements committed stiffness matrix.
    """
    pass

def start() -> None:
    """Start the timer."""
    pass

def stop() -> None:
    """Stop the timer and print timing information."""
    pass

def stripXML(inputml: str, outputdata: str, outputxml: str) -> None:
    """Strip a xml file to a data file and a descriptive file.

    * inputxml (str) : input xml file name.
    * outputdata (str) : output data file name.
    * outputxml (str) : output xml file name.
    """
    pass

def setNumThreads(num: int) -> None:
    """set the number of threads to be used in the multi-threaded environment.

    * num (int) : number of threades
    """
    pass

def getNumThreads() -> int:
    """return the total number of threads available."""
    pass

def convertBinaryToText(inputfile: str, outputfile: str) -> None:
    """Convert binary file to text file.

    * inputfile (str) : input file name.
    * outputfile (str) : output file name.
    """
    pass

def convertTextToBinary(inputfile: str, outputfile: str) -> None:
    """Convert text file to binary file.

    * inputfile (str) : input file name.
    * outputfile (str) : output file name.
    """
    pass

def mesh(*args) -> None:
    """``mesh(type, tag, *args)``

    Create a mesh object. See below for available mesh types.
    See https://openseespydoc.readthedocs.io/en/latest/src/mesh.html

    ``mesh('line', tag, numnodes, *ndtags, id, ndf, meshsize, eleType='', *eleArgs=[])``

    ``mesh('tri', tag, numlines, *ltags, id, ndf, meshsize, eleType='', *eleArgs=[])``

    ``mesh('quad', tag, numlines, *ltags, id, ndf, meshsize, eleType='', *eleArgs=[])``

    ``mesh('tet', tag, nummesh, *mtags, id, ndf, meshsize, eleType='', *eleArgs=[])``

    ``mesh('part', tag, type, *pArgs, eleType='', *eleArgs=[], '-vel', *vel0, '-pressure', p0)``

    ``mesh('bg', basicsize, *lower, *upper, '-tol', tol, '-meshtol', meshtol,
    '-wave', wavefilename, numl, *locations, '-numsub', numsub, '-structure', id, numnodes, *snodes,
    '-largeSize', level, *llower, *lupper)``
    """
    pass

def remesh(alpha: float = -1.0) -> None:
    """See https://openseespydoc.readthedocs.io/en/latest/src/remesh.html"""
    pass

# ----------- Sensitivity Commands ---------------
def parameter(tag: int, *args) -> None:
    """``parameter(tag, <specific parameter args>)``

    See https://openseespydoc.readthedocs.io/en/latest/src/parameter.html

    In DDM-based FE response sensitivity analysis, the sensitivity parameters can be material, geometry or discrete loading parameters.

    * tag (int) : integer tag identifying the parameter.
    * <specific parameter args> : depend on the object in the FE model encapsulating the desired parameters.

    Note
    -----
    Each parameter must be unique in the FE domain, and all parameter tags must be numbered sequentially starting from 1.

    Examples
    ---------
    To identify the elastic modulus, E, of the material 1 at section 3 of element 4, the <specific object arguments> string becomes:

    ``parameter(1, 'element', 4, 'section', 3, 'material', 1, 'E')``

    To identify the elastic modulus, E, of elastic section 3 of element 4 (for elastic section,
    no specific material need to be defined), the <specific object arguments> string becomes:

    ``parameter(1, 'element', 4, 'section', 3, 'E')``

    To parameterize E for element 4 with material 1 (no section need to be defined), the <specific object arguments> string simplifies as:

    ``parameter(1, 'element', 4, 'material', 1, 'E')``

    Note
    ------
    Notice that the format of the <specific object arguments> is different for each considered element/section/material.
    The specific set of parameters and the relative <specific object arguments> format will be added in the future.
    """
    pass

def addToParameter(tag: int, *args) -> None:
    """``addToParameter(tag, <specific parameter args>)``

    In case that more objects (e.g., element, section) are mapped to an existing parameter,
    the command can be used to relate these additional objects to the specific parameter.

    * tag (int) : integer tag identifying the parameter.
    * <specific parameter args> : depend on the object in the FE model encapsulating the desired parameters.
    """
    pass

def updateParameter(tag: int, newValue: float) -> None:
    """Once the parameters in FE model are defined, their value can be updated.
    See https://openseespydoc.readthedocs.io/en/latest/src/updateParameter.html

    * tag (int) : integer tag identifying the parameter.
    * newValue (float) : the updated value to which the parameter needs to be set.
    """
    pass

def setParameter(*args) -> None:
    """``setParameter('-val', newValue, <'-ele', *eleTags>, <'-eleRange', start, end>, <*args>)``

    set value for an element parameter

    * newValue (float) : the updated value to which the parameter needs to be set.
    * eleTags (list (int)) : a list of element tags
    * start (int) : start element tag
    * end (int) : end element tag
    * args (list (str)) : a list of strings for the element parameter
    """
    pass

def getParamTags() -> list:
    """Return a list of tags for all parameters."""
    pass

def getParamValue(paramTag: int) -> float:
    """Return the value of a parameter.

    * paramTag (int) : integer tag identifying the parameter.
    """
    pass

def computeGradients() -> None:
    """This command is used to perform a sensitivity analysis.
    If the user wants to call this command, then the ``'-computeByCommand'`` should be set in the ``sensitivityAlgorithm`` command.
    """
    pass

def sensitivityAlgorithm(type: str) -> None:
    """This command is used to create a sensitivity algorithm.

    * type (str) : the type of the sensitivity algorithm,

    ``'-computeAtEachStep'`` automatically compute at the end of each step

    ``'-compuateByCommand'`` compute by calling computeGradients.
    """
    pass

def sensNodeDisp(nodeTag: int, dof: int, paramTag: int) -> float:
    """Returns the current displacement sensitivity to a parameter at a specified node.

    * nodeTag (int) : node tag
    * dof (int) : specific dof at the node (1 through ndf)
    * paramTag (int) : parameter tag
    """
    pass

def sensNodeVel(nodeTag: int, dof: int, paramTag: int) -> float:
    """Returns the current velocity sensitivity to a parameter at a specified node.

    * nodeTag (int) : node tag
    * dof (int) : specific dof at the node (1 through ndf)
    * paramTag (int) : parameter tag
    """
    pass

def sensNodeAccel(nodeTag: int, dof: int, paramTag: int) -> float:
    """Returns the current velocity acceleration to a parameter at a specified node.

    * nodeTag (int) : node tag
    * dof (int) : specific dof at the node (1 through ndf)
    * paramTag (int) : parameter tag
    """
    pass

def sensLambda(patternTag: int, paramTag: int) -> float:
    """Returns the current load factor sensitivity to a parameter in a load pattern.

    * patternTag (int) : load pattern tag
    * paramTag (int) : parameter tag
    """
    pass

def sensSectionForce(*args) -> float:
    """``sensSectionForce(eleTag, <secNum>, dof, paramTag)``

    Returns the current section force sensitivity to a parameter at a specified element and section.

    * eleTag (int) : element tag
    * secNum (int) : section number (optional)
    * dof (int) : specific dof at the element (1 through element force ndf)
    * paramTag (int) : parameter tag
    """
    pass

def sensNodePressure(nodeTag: int, paramTag: int) -> float:
    """Returns the current pressure sensitivity to a parameter at a specified node.

    * nodeTag (int): node tag
    * paramTag (int): parameter tag
    """
    pass

# -------------Reliability Commands---------------
def randomVariable(*args) -> None:
    """``randomVariable(tag, dist, '-mean', mean, '-stdv', stdv,
    '-startPoint', startPoint, '-parameters', *params)``

    Create a random variable with user specified distribution.
    See https://openseespydoc.readthedocs.io/en/latest/src/randomVariable.html
    """
    pass

def getRVTags() -> list:
    """Get the tags of all random variables."""
    pass

def getRVParamTag(rvTag: int) -> int:
    """Get parameter tag for random variable rvTag."""
    pass

def getRVValue(rvTag: int) -> float:
    """Get the current value for random variable rvTag."""
    pass

def getMean(rvTag: int) -> float:
    """Get the mean for random variable rvTag."""
    pass

def getStdv(rvTag: int) -> float:
    """Get the standard deviation for random variable rvTag."""
    pass

def getPDF(rvTag: int, X: float) -> float:
    """Get the probability density function value for random variable rvTag at X."""
    pass

def getCDF(rvTag: int, X: float) -> float:
    """Get the cumulative distribution function value for random variable rvTag at X."""
    pass

def getInverseCDF(rvTag: int, p: float) -> float:
    """Get the inverse CDF value for random variable rvTag at probability p."""
    pass

def correlate(rvTag1: int, rvTag2: int, rho: float) -> None:
    """Add the correlation coefficient rho between rvTag1 and rvTag2."""
    pass

def functionEvaluator(type_: str, *args) -> None:
    """
    ``functionEvaluator('Python', <'-file', filename>) ``
    """
    pass

def gradientEvaluator(type_: str, *args) -> None:
    """
    ``functionEvaluator('FiniteDifference', '-pert', perturbationFactor=1000.0)``

    ``functionEvaluator('Implicit')``
    """
    pass

def performanceFunction(tag: int, *args) -> None:
    """
    ``performanceFunction(tag: int, <lsf: str>)``
    Add performance function lsf with tag to the reliability domain.
    """
    pass

def gradPerformanceFunction(lsfTag: int, rvTag: int, expr: str) -> None:
    """
    Add gradient of LSF with tag lsfTag for random variable rvTag.

    Parameters
    -----------
    lsfTag: int, the tag of limit state function.
    rvTag: int, random variable with tag rvTag.
    expr: str, gradient expression.
    """
    pass

def transformUtoX(*args) -> list:
    """
    ``transformUtoX(u1, u2, ... , un, <rvTag1, rvTag2, ... , rvTagn>)``

    The value of the standard normal distribution ``u`` is converted into
    the value of each random variable ``X`` according to the ``probabilityTransformation``.
    """
    pass

def wipeReliability() -> None:
    """Wipe the Reliability domin."""
    pass

def probabilityTransformation(type_: str, *args) -> None:
    """Create probability transformation.

    ``probabilityTransformation('Nataf', <'-print', printTag(0)>)``

    ``probabilityTransformation('AllIndependent', <'-print', printTag(0)>)``
    """
    pass

def startPoint(type_: str, *args) -> None:
    """Set the start point.

    ``startPoint('Mean')``

    ``startPoint('Zero')``

    ``startPoint('Origin')``

    ``startPoint('-file', filename)``
    """
    pass

def randomNumberGenerator(type_: str) -> None:
    """
    ``randomNumberGenerator('CStdLib')``
    """
    pass

def reliabilityConvergenceCheck(type_: str, *args) -> None:
    """Set Standard or Optimality convergence check.

    ``reliabilityConvergenceCheck('Standard',
    <<'-e1', e1>, <'-e2', e2>, <'-scaleValue', scaleValue>, <'-print', print>>)``

    ``reliabilityConvergenceCheck('OptimalityCondition',
    <<'-e1', e1>, <'-e2', e2>, <'-scaleValue', scaleValue>, <'-print', print>>)``
    """
    pass

def searchDirection(type_: str, *args) -> None:
    """Set the type of search direction.

    ``searchDirection('iHLRF')``

    ``searchDirection('PolakHe', <<'-gamma', gamma>, <'-delta', delta>>)``

    ``searchDirection('GradientProjection')``

    ``searchDirection('SQP', <<'-c_bar', c_bar>, <'-e_bar', e_bar>>)``
    """
    pass

def meritFunctionCheck(type_: str, *args) -> None:
    """Set the type of merit function check.

    ``meritFunctionCheck('AdkZhang', <<'-multi', multi>, <'-add', add>, <'-factor', factor>>)``

    ``meritFunctionCheck('PolakHe', <'-factor', factor(0.5)>)``

    ``meritFunctionCheck('SQP', <'-factor', factor(0.5)>)``
    """
    pass

def stepSizeRule(type_: str, *args) -> None:
    """Set the type of step size rule.

    ``stepSizeRule('Armijo', <<'-print', printFlag(0)>, <'-maxNum', maxNumReductions(10)>,
    <'-base', base(0.5)>, <'-initial', b0(1.0), numberOfShortSteps(2)>,
    <'-sphere', radius(50.0), surfaceDistance(0.1), evolution(0.5)>>)``

    ``stepSizeRule('Fixed', <'-stepSize', stepSize(1.0)>)``
    """
    pass

def rootFinding(type_: str, *args) -> None:
    """Set the type of rootFinding.

    ``rootFinding('Secant', <<'-maxIter', maxIter(50)>, <'-tol', tol(1e-3)>,
    <'-maxStepLength', maxStepLength(1.0)>>)``
    """
    pass

def findDesignPoint(type_: str, *args) -> None:
    """Set the type of findDesignPoint.

    ``findDesignPoint('StepSearch', <<'-maxNumIter', maxIter(100)>,
    <'-printAllPointsX', fileName>, <'-printAllPointsY', fileName>,
    <'-printDesignPointX', fileName>,<'-printDesignPointY', fileName>,
    <'-printCurrentPointX', fileName>,<'-printCurrentPointY', fileName>,>)``
    """
    pass

def runFOSMAnalysis(filename: str) -> None:
    """Run FOSM Analysis."""
    pass

def runFORMAnalysis(filename: str, *args) -> None:
    """Run FORM Analysis.

    ``runFORMAnalysis(filename, <'-relSens', relSensTag(0)>)``
    """
    pass

def getLSFTags() -> list:
    """Get all the tags of Limit State Functions."""
    pass

def runImportanceSamplingAnalysis(filename: str, *args) -> None:
    """
    ``runImportanceSamplingAnalysis(filename,
    <'-type', <'failureProbability'>,<'outCrossingFailureProbability'>,
    'responseStatistics'>, 'saveGvalues'>>, <'-variance', samplingVariance(1.0)>,
    '-maxNum', maxNum(1000)>,<'-targetCOV', targetCOV(0.05)>,
    <'-print', printFlag(0)>)``
    """
    pass

# -----------Parallel Commands--------------------
def getPID() -> int:
    """Get the processor ID of the calling processor."""
    pass

def getNP() -> int:
    """Get total number of processors."""
    pass

def barrier() -> None:
    """Set a barrier for all processors, i.e., faster processors
    will pause here to wait for all processors to reach to this point.
    """
    pass

def send(*args) -> None:
    """``send('-pid', pid, *data)``

    Send information to another processor.

    * pid (int) : ID of processor where data is sent to
    * data (list (int)) : can be a list of integers;
    * data (list (float)) : can be a list of floats;
    * data (str) : can be a string

    Note
    -----
    ``send`` command and ``recv`` command must match and the order of calling both commands matters.
    """
    pass

def recv(*args) -> None:
    """``recv('-pid', pid)``

    Receive information from another processor.

    * pid (int) : ID of processor where data is received from
    * pid (str) : if pid is 'ANY', the processor can receive data from any processor.

    Note
    -----
    ``send`` command and ``recv`` command must match and the order of calling both commands matters.
    """
    pass

def Bcast(*data) -> None:
    """Broadcast information from processor 0 to all processors.

    * data (list (int)) : can be a list of integers
    * data (list (float)) : can be a list of floats
    * data (str) : can be a string

    Note
    -----
    Run the same command to receive data sent from pid = 0.
    """
    pass

def setStartNodeTag(ndtag: int) -> None:
    """Set the starting node tag for the mesh command. The purpose of this command is to control
    the node tags generated by the mesh command. Some nodes are shared by processors,
    which must have same tags. Nodes which are unique to a processor must have uniques
    tags across all processors.

    * ndtag (int) : starting node tag for the next call of mesh command.
    """
    pass

def domainChange() -> None:
    """Mark the domain has changed manually.
    This is used to notify processors whose domain is not changed,
    but the domain in other processors have changed.
    """
    pass

def partition(*args) -> None:
    """``partition('-ncuts', ncuts, '-niter', niters, '-ufactor', ufactor, '-info')``

    In a parallel environment, this command partitions the model.
    It requires that all processors have the exact same model to be partitioned.

    * ncuts (int) : Specifies the number of different partitionings that it will compute.
        The final partitioning is the one that achieves the best edge cut or communication volume.
        (Optional default is 1).
    * niters (int) : Specifies the number of iterations for the refinement algorithms at
        each stage of the uncoarsening process. (Optional default is 10).
    * ufactor (int) : Specifies the maximum allowed load imbalance among the partitions.
        (Optional default is 30, indicating a load imbalance of 1.03).
    * '-info' (str) : print information. (optional)
    """
    pass

def domainCommitTag(commitTag: int) -> int:
    """ """
    pass

def IGA(type_: str, *args) -> None:
    """Isogeometric Analysis.

    See https://www.sciencedirect.com/science/article/abs/pii/S0010448523000490

    ``IGA('Patch', tag, nodeStartTag, P, Q, noPtsX, noPtsY,
    <'-type', ['KLShell' or 'KLShell_BendingStrip'>,
    <'-nonLinearGeometry', [0 or 1]>,
    <'-planeStressMatTags', *[list of tags]>,
    <'-gFact', gx, gy, gz>,
    <'-theta', *[list of thetas]>,
    <'-thickness', *[list of layer thicknesses]>,
    <'-uKnot', *[list of uKnots]>,
    <'-vKnot', *[list of vKnots]>,
    <'-controlPts', *[list coordinates of control points, u-direction first]>,
    <'-sectionTag', sectionTag>,
    <'-nodeStartTag', nodeStartTag>,
    <'-elementStartTag', elementStartTag>)``

    ``IGA('SurfacePatch', tag, nodeStartTag, P, Q, noPtsX, noPtsY,
    <'-type', ['KLShell' or 'KLShell_BendingStrip'>,
    <'-nonLinearGeometry', [0 or 1]>,
    <'-planeStressMatTags', *[list of tags]>,
    <'-gFact', gx, gy, gz>,
    <'-theta', *[list of thetas]>,
    <'-thickness', *[list of layer thicknesses]>,
    <'-uKnot', *[list of uKnots]>,
    <'-vKnot', *[list of vKnots]>,
    <'-controlPts', *[list coordinates of control points, u-direction first]>,
    <'-sectionTag', sectionTag>,
    <'-nodeStartTag', nodeStartTag>,
    <'-elementStartTag', elementStartTag>)``
    """
    pass

def NDTest(type_: str, *args) -> Union[list, None]:
    """Test the nD material.

    Parameters
    -----------
    type_: str, optional "SetStrain", "CommitState", "PrintStress", "PrintStrain",
        "GetStrain", "GetStress", "GetTangentStiffness", "UpdateIntegerParameter",
        "UpdateDoubleParameter"
    args: args for type_.

    ``NDTest("SetStrain", tag, *strains)``

    ``NDTest("CommitState", tag)``

    ``NDTest("PrintStress", tag)``

    ``NDTest("PrintStrain", tag)``

    ``NDTest("GetStrain", tag)``

    ``NDTest("GetStress", tag)``

    ``NDTest("GetTangentStiffness", tag)``

    ``NDTest("UpdateIntegerParameter", tag, responseID, theNewIntegerParameterValue)``

    ``NDTest("UpdateDoubleParameter", tag, responseID, theNewDoubleParameterValue)``
    """
    pass
