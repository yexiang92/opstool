import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import scipy.sparse as sp


def get_spy_matrix(numberer: str = "RCM", plot: bool = True) -> tuple:
    """
    Plot the sparsity pattern of the system matrix in OpenSeesPy.
    This function constructs a sparse matrix representing the connectivity of the system
    and visualizes it using a spy plot. The bandwidth of the matrix is also computed.
    The bandwidth is defined as the maximum distance between non-zero entries in the same row.
    The function returns the sparse matrix and its bandwidth.

    Parameters:
    -----------
    numberer : str, default="RCM"
        The numberer type to use for the analysis. Options include "RCM", "Plain", etc.
    plot : bool, default=True
        If True, the sparsity pattern will be plotted using matplotlib.

    Returns:
    --------
    tuple: A tuple containing:
        - SpyMatrix (scipy.sparse.csr_matrix): The sparse matrix representing the system.
        - bw (int): The bandwidth of the matrix.
    """
    ops.constraints("Plain")  # Ensure constraints are set to plain
    ops.algorithm("Linear")  # Use linear algorithm
    ops.system("BandGeneral")  # Use banded system solver
    ops.numberer(numberer)  # Set the numberer type
    ops.test("NormDispIncr", 1.0e-6, 10, 0)  # Set convergence test
    ops.integrator("LoadControl", 0.0)  # Use load control integrator
    ops.analysis("Static")  # Set analysis type to static

    ops.analyze(1)  # Warm-up run if needed
    Neqn = ops.systemSize()

    # Use sparse LIL matrix for efficient incremental construction
    SpyMatrix = sp.lil_matrix((Neqn, Neqn), dtype=np.uint8)

    for e in ops.getEleTags():
        dofs = []
        for nd in ops.eleNodes(e):
            dofs.extend(d for d in ops.nodeDOFs(nd) if d >= 0)  # Skip constrained DOFs

        for i in dofs:
            for j in dofs:
                SpyMatrix[i, j] = 1

    # Compute bandwidth efficiently
    row, col = SpyMatrix.nonzero()
    bw = max(col - row) + 1 if len(row) > 0 else 0

    if plot:
        # Convert to CSR for plotting
        SpyMatrix = SpyMatrix.tocsr()

        # Plot
        plt.figure(figsize=(6, 6))
        plt.spy(SpyMatrix, markersize=0.05)
        plt.title(f"Sparsity Pattern (Bandwidth={bw})")
        plt.xlabel("Equation Index")
        plt.ylabel("Equation Index")
        plt.tight_layout()
        plt.show()

    ops.wipeAnalysis()  # Clean up analysis to avoid memory issues

    return SpyMatrix, bw


if __name__ == "__main__":
    import opstool as opst

    opst.load_ops_examples("CableStayedBridge")
    get_spy_matrix("RCM", plot=True)  # Example usage with RCM numberer
