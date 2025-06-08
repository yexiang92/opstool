.. This theme is based on the `sphinx-needs`.

.. grid::
   :gutter: 2 3 3 3
   :margin: 4 4 1 2
   :class-container: architecture-bg
   :class-row: sd-w-100

   .. grid-item::
      :columns: auto
      :child-align: justify
      :class: sd-fs-3

      .. div:: sd-font-weight-bold
         
         Making structural simulation with OpenSees and OpenSeesPy effortless and straightforward

      .. div:: sd-fs-5 sd-font-italic

         Welcome to the documentation for opstool, a thoughtfully crafted preprocessing and postprocessing package 
         designed for OpenSeesPy. 
         This tool simplifies the creation, analysis, and visualization of structural models and results, 
         greatly enhancing both the efficiency and user experience of OpenSeesPy.

      .. grid:: 1 1 2 2
         :gutter: 2 2 3 3
         :margin: 0
         :padding: 0

         .. grid-item::
            :columns: auto

            .. button-ref:: quick_start
               :ref-type: doc
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               Get Started
      
         .. grid-item::
            :columns: auto

            .. button-link:: https://opensees.berkeley.edu/
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               OpenSees
         
         .. grid-item::
            :columns: auto

            .. button-link:: https://openseespydoc.readthedocs.io/en/latest/index.html#
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               OpenSeesPy
         
         .. grid-item::
            :columns: auto

            .. button-link:: https://portwooddigital.com/
               :outline:
               :color: primary
               :class: sd-rounded-pill sd-px-4 sd-fs-5

               Portwood Digital

----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: :octicon:`shield-check;1.5em;sd-mr-1 fill-primary` Preprocessing Capabilities

      - Fiber Section Meshing and Property Calculation Using `sectionproperties <https://sectionproperties.readthedocs.io/en/stable/>`__
      - Easily translate Tcl-based OpenSees models into Python scripts
      - Unified Unit Conversion System
      - Integration with `Gmsh <https://gmsh.info/>`__ Meshing
      - Features in Development
  
   .. grid-item-card:: :octicon:`checkbox;1.5em;sd-mr-1 fill-primary` Elegant Visualization

      - Powered by `PyVista <https://docs.pyvista.org/>`__ and `Plotly <https://plotly.com/python/>`__
      - Supporting most commonly used OpenSees element types
      - Nearly identical APIs are tailored for both engines

   .. grid-item-card:: :octicon:`workflow;1.5em;sd-mr-1 fill-primary` Data Processing

      Analysis results are saved and processed using `xarray <https://docs.xarray.dev/en/stable/index.html#>`__, 
      enabling easy retrieval and interpretation of analysis outcomes.
   
   .. grid-item-card:: :octicon:`dependabot;1.5em;sd-mr-1 fill-primary` Analysis Assistance

      - Smart Analysis, features include automatic step size division, algorithm switching, and other advanced functionalities
      - Moment-Curvature Analysis of Sections
      - Linar Buckling Analysis

----------------

Citing OPSTOOL
---------------

.. _cards-clickable:

OPSTOOL was published in

.. card:: :octicon:`project-roadmap;1.5em;sd-mr-1 fill-primary` OPSTOOL Publication
    :link: https://www.sciencedirect.com/science/article/pii/S2352711025000937

    Yexiang Yan and Yazhou Xie. *"opstool: A Python library for OpenSeesPy analysis automation, streamlined pre-and post-processing, and enhanced data visualization."* SoftwareX 30 (2025): 102126. DOI: `https://doi.org/10.1016/j.softx.2025.102126 <https://www.sciencedirect.com/science/article/pii/S2352711025000937>`_.


If you used OPSTOOL in your scientific publication, we would very appreciated if you include the adequate citation:


  Bibtex entry::

    @article{YAN2025102126,
      title = {opstool: A Python library for OpenSeesPy analysis automation, streamlined pre- and post-processing, and enhanced data visualization},
      author = {Yexiang Yan and Yazhou Xie},
      journal = {SoftwareX},
      volume = {30},
      pages = {102126},
      year = {2025},
      issn = {2352-7110},
      doi = {https://doi.org/10.1016/j.softx.2025.102126},
      }


Why Choose opstool?
--------------------

- **Efficiency**: Streamlines complex workflows, reducing time spent on repetitive tasks.
- **Flexibility**: Provides nearly identical interfaces for different visualization engines.
- **Accessibility**: Makes advanced structural analysis tools like OpenSeesPy more approachable to users of all levels.

``opstool`` is actively evolving, with continuous additions of new features planned for the future.
With ``opstool``, you can focus on what matters most: 
understanding and solving your structural engineering challenges. 
Whether you are building models, visualizing results, or interpreting data, 
``opstool`` is your go-to solution for OpenSeesPy workflows.


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   installation.rst
   known_issues.rst
   quick_start.rst

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   src/pre/index
   src/post/index
   src/vis/index.ipynb
   src/analysis/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index


.. toctree::
   :maxdepth: 1
   :caption: API Reference
   
   src/api/global
   src/api/pre
   src/api/post
   src/api/vis
   src/api/analysis
