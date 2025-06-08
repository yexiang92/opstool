Preprocessing
===============

Fiber Section Mesh
--------------------
Fiber section mesh is a tool to create a mesh of fiber sections for OpenSeesPy models. It can be used to generate fiber sections from a given geometry and material properties.

.. toctree::
   :maxdepth: 1

   pre.section.rst


Tcl to Python
---------------
Converts Tcl commands to Python commands for OpenSeesPy. 
This is useful if you have a model defined in Tcl and want to convert it to Python for use with OpenSeesPy.

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :recursive:
   
   opstool.pre.tcl2py

Gmsh to OpenSeesPy
--------------------
Building the OpenSeesPy model from a Gmsh mesh file or model.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:
   
   opstool.pre.Gmsh2OPS

Unit System
-------------
A class to handle unit systems in OpenSeesPy. It allows for the conversion of units and provides a way to manage different unit systems within the OpenSeesPy framework.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:
   
   opstool.pre.UnitSystem

Loads Transform and Processing
--------------------------------
This module provides functions to apply load distributions, create gravity loads, and transform various types of loads (uniform, point, surface) from global system into a local system for OpenSeesPy.

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :recursive:
   
   opstool.pre.create_gravity_load
   opstool.pre.gen_grav_load
   opstool.pre.transform_beam_uniform_load
   opstool.pre.transform_beam_point_load
   opstool.pre.transform_surface_uniform_load
   opstool.pre.apply_load_distribution

Model Data
----------------
Return the model data from the OpenSeesPy model.

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :recursive:
   
   opstool.pre.get_node_coord
   opstool.pre.get_node_mass
   opstool.pre.get_mck


Model Mass 
-----------

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:
   
   opstool.pre.ModelMass

Utilities
---------------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :recursive:
   
   opstool.pre.find_void_nodes
   opstool.pre.remove_void_nodes