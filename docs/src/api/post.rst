Postprocessing
===============
The post-processing module mainly handles the saving and reading of model and response data. 
Most responses are returned in the `DataArray <https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataarray>`_ and `DataSet <https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataset>`_ formats of ``xarray`` for easy retrieval.

Utilities
----------
.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :recursive:

   opstool.post.set_odb_path
   opstool.post.update_unit_system
   opstool.post.reset_unit_system

Data Saving
------------
Model, Eigenvalue and Linear Buckling Analysis Response Preservation:
**********************************************************************
.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :recursive:

   opstool.post.save_model_data
   opstool.post.save_eigen_data
   opstool.post.save_linear_buckling_data

The built-in step response data by opstool is saved
**********************************************************************
.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   opstool.post.CreateODB

Result Reading
--------------
Retrive the response data:

.. autosummary::
   :toctree: _autosummary
   :template: custom-function-template.rst
   :recursive:

   opstool.post.get_model_data
   opstool.post.get_eigen_data
   opstool.post.get_linear_buckling_data
   opstool.post.get_nodal_responses
   opstool.post.get_element_responses
   opstool.post.get_sensitivity_responses