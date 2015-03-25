import sys

from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np


nSupport = 5
supportSpace = (0, 1)
nEval = 9
evalSpace = (-0.2, 1.2)
testfunction = lambda a:  a**5 - a**4 + a**3 - a**2 + 1 # [0, 1]

def basisfunction(radius):
    function = "gauss"

    if function == "gauss":
        shape = 20
        return np.exp( - (shape*radius)**2)
    elif function == "tps":
        return 0 if radius == 0 else radius**2 * np.log(radius)
    else:
        print("No Basisfunction selected.")
        sys.exit(-1)

def plot(supports, evals, interp, coeffs):
    sRange = np.linspace(min(supportSpace[0], evalSpace[0]), max(supportSpace[1], evalSpace[1]))  # Super range
    f, axes = plt.subplots(3, sharex=True)
    axes[0].plot(sRange, testfunction(sRange), "b") # Plot the original function
    axes[0].plot(supports, testfunction(supports), "bo", label = "Supports") # Plot the support points
    axes[0].plot(evals, interp, "ro-", label = "Evals") # Plot the evaluation points and values
    axes[0].legend()
    axes[0].grid()
    delta = [ interp[i] - testfunction(x) for i, x in enumerate(evals) ]
    axes[1].plot(evals, delta, "ro-", label = "Delta") # Plot error
    axes[1].legend()
    axes[1].grid()

    for c in zip(supports, coeffs):
        basis = basisfunction(abs(c[0]-sRange))*c[1]
        axes[2].plot(sRange, basis)

    axes[2].grid()
    plt.show()


def main():
    MPIrank = MPI.COMM_WORLD.Get_rank()
    MPIsize = MPI.COMM_WORLD.Get_size()
    print("MPI Rank = ", MPIrank)
    print("MPI Size = ", MPIsize)

    # plot()

    A = PETSc.Mat(); A.createDense( (nSupport, nSupport) )
    A.setName("A");  A.setUp()
    
    E = PETSc.Mat(); E.createDense( (nEval, nSupport) )
    E.setName("E");  E.setUp()
    
    c = A.createVecRight(); c.setName("c")
    b = A.createVecRight(); b.setName("b")
    interp = E.createVecLeft(); interp.setName("interp")

    supports = np.linspace(supportSpace[0], supportSpace[1], nSupport)

    for row, i in enumerate(supports):
        for col, j in enumerate(supports):
            A.setValue(row, col, basisfunction(abs(i-j)))
        b.setValue(row, testfunction(i))
    A.assemble()

    ksp = PETSc.KSP()
    ksp.create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, c)

    evals = np.linspace(evalSpace[0], evalSpace[1], nEval)

    for row, i in enumerate(evals):
        for col, j in enumerate(supports):
            E.setValue(row, col, basisfunction(abs(i - j)))
    E.assemble()

    E.mult(c, interp);
              
    plot(supports, evals, interp.array, c.array)


if __name__ == '__main__':
    main()


