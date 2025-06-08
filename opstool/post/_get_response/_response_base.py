from abc import ABC, abstractmethod

import numpy as np


class ResponseBase(ABC):
    @abstractmethod
    def initialize(self):
        """initialize response data."""

    @abstractmethod
    def reset(self):
        """Reset response data."""

    @abstractmethod
    def add_data_one_step(self, *args):
        """Add data at each analysis step."""

    @abstractmethod
    def get_data(self):
        """Get responses data"""

    @abstractmethod
    def get_track(self):
        """Get track tag."""

    @abstractmethod
    def save_file(self, *args):
        """Save responses data."""

    @abstractmethod
    def read_file(self, *args):
        """Read response data from a file."""


def _expand_to_uniform_array(array_list, dtype=None):
    """
    Convert a list of NumPy arrays with varying shapes into a single 2D/3D NumPy array,
    padding with NaN where dimensions do not match.

    Parameters:
    array_list (list): List of NumPy arrays with varying shapes.

    Returns:
    np.ndarray: A padded NumPy array with uniform shape.
    """
    # Find the maximum shape along each dimension
    max_shape = np.max([array.shape for array in array_list], axis=0)

    # Create a result array filled with NaN
    result = np.full((len(array_list), *tuple(max_shape)), np.nan)

    # Copy data into the result array
    for i, arr in enumerate(array_list):
        slices = tuple(slice(0, dim) for dim in arr.shape)
        result[i][slices] = arr
    if dtype is not None:
        result = result.astype(dtype)
    return result
