import numpy as np

def indep_roll(arr: np.ndarray, shifts: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Apply an independent roll for each dimensions of a single axis.
    
    Parameters
    ----------
    arr : np.ndarray
        Array of any shape
    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.
    axis : int, optional
        Axis along which elements are shifted. Defaults to 1.

    Returns
    -------
    arr : np.ndarray
        shifted array
    """
    arr = np.swapaxes(arr, axis, -1)  # Move the target axis to the last position
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]# Create grid indices
    shifts[shifts < 0] += arr.shape[-1]  # Convert to a positive shift
    new_indices = all_idcs[-1] - shifts[:, np.newaxis]
    result = arr[tuple(all_idcs[:-1]) + (new_indices,)]
    arr = np.swapaxes(result, -1, axis)
    return arr