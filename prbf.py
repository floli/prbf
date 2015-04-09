#!env python3

import ipdb

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np


eMesh = {1: [np.linspace(0, 1, 4) ],
         2: [np.linspace(0, 0.5, 2, False), # False == Do no include endpoint in range
             np.linspace(0.5,1, 2 )],
         4: [np.linspace(0, 0.1, 5, False),
             np.linspace(0.1, 0.3, 10, False),
             np.linspace(0.3, 0.7, 10, False),
             np.linspace(0.7,1 , 10, False), ]
         }


MPIrank = MPI.COMM_WORLD.Get_rank()
MPIsize = MPI.COMM_WORLD.Get_size()

nSupport = 9           # Number of support points
supportSpace = (0, 1)   # Range in which the support points are equally distributed

# Dimension of interpolation. Used for adding a polynomial to the matrix. Set to zero to deactivate polynomial
# f(x) = y with scalars x and y gives dimension = 2.
dimension = 0

testfunction = lambda a:  a**5 - a**4 + a**3 - a**2 + 1 # [0, 1]


def basisfunction(radius):
    function = "gauss"

    if function == "gauss":
        shape = 8
        return np.exp( - (shape*radius)**2)
    elif function == "tps":
        return 0 if radius == 0 else radius**2 * np.log(radius)
    else:
        print("No Basisfunction selected.")
        sys.exit(-1)

def partitions(lst):
    """ Partitions the list evenly through all domains. """
    MPIsize = (MPI.COMM_WORLD.Get_size())
    # lst = range(nSupport)
    division = len(lst) / float(MPIsize)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(MPIsize) ]

    
def plot(supports, interp, coeffs):
    """ Support Points, Evaluation Point, Interpolation Results, Coefficients"""
    evals =  np.concatenate( [i for i in eMesh[MPIsize]] )
    sRange = np.linspace(supportSpace[0]-0.2, supportSpace[1]+0.2, 1000)  # Range a bit larget than the support space
    f, axes = plt.subplots(3, sharex=True)
    
    axes[0].plot(sRange, testfunction(sRange), "b") # Plot the original function
    axes[0].plot(supports, testfunction(supports), "bo", label = "Supports") # Plot the support points
    axes[0].plot(evals, interp, "ro-", label = "Evals") # Plot the evaluation points and values
    axes[0].legend()
    axes[0].grid()

    # Calculate and plot error
    delta = [ interp[i] - testfunction(x) for i, x in enumerate(evals) ]
    rms = np.sqrt( np.mean(np.power(delta, 2)) )
    axes[1].plot(evals, delta, "ro-", label = "Delta")
    axes[1].set_title("RMS = " + str(rms) )
    axes[1].legend()
    axes[1].grid()

    # Plot the actual basisfunction
    for c in zip(supports, coeffs):
        basis = basisfunction(abs(c[0]-sRange))*c[1]
        axes[2].plot(sRange, basis)
    if dimension:
        poly = coeffs[nSupport] + coeffs[nSupport + 1] * sRange
        axes[2].plot(sRange, poly)
    axes[2].grid()

    if MPIsize > 1:
        # Plot a vertical line at domain boundaries of support points
        sParts = partitions(supports)
        middle = (min(sParts[1])-max(sParts[0])) / 2 # We assume that the all points are equidistant
        for i in sParts:
            axes[0].axvline(x = max(i) + middle, color='b', linestyle=':')  # Add a small eps to show which domain
            axes[1].axvline(x = max(i) + middle, color='b', linestyle=':')  # this point belongs to.

        # Plot a vertical line at domain boundaries of evaluation points
        middle = (min(eMesh[MPIsize][1])-max(eMesh[MPIsize][0])) / 2 # We assume that the all points are equidistant
        for i in eMesh[MPIsize]:
            axes[0].axvline(x = max(i) + middle, color='r', linestyle='--')  # Add a small eps to show which domain
            axes[1].axvline(x = max(i) + middle, color='r', linestyle='--')  # this point belongs to.

    plt.tight_layout()
    plt.show()

def Print(*s):
    out = " ".join( [ str(i) for i in s] )
    print("[%s] %s" % (MPIrank, out))
    MPI.COMM_WORLD.Barrier() # Just to keep the output together


def main():
    print("MPI Rank = ", MPIrank)
    print("MPI Size = ", MPIsize)

    supports = np.linspace(supportSpace[0], supportSpace[1], nSupport)
    sParts = partitions(supports)[MPIrank]
    
    eParts = eMesh[MPIsize][MPIrank] # np.array of positions to evaluate
    

    Print("sParts = ", sParts)
    Print("eParts = ", eParts)
    
    MPI.COMM_WORLD.Barrier() # Just to keep the output together
    
    A = PETSc.Mat(); A.createDense( size = ((len(sParts), PETSc.DETERMINE), (len(sParts), PETSc.DETERMINE)) )
    # A = PETSc.Mat(); A.createDense( (nSupport + dimension, nSupport + dimension) )
    A.setName("System Matrix");  A.setFromOptions(); A.setUp()
    A.assemble()
    # E = PETSc.Mat(); E.createDense( (nEval, nSupport + dimension) )
    E = PETSc.Mat(); E.createDense( size = ((len(eParts), PETSc.DETERMINE), (len(sParts), PETSc.DETERMINE)) )
    E.setName("Evaluation Matrix");  E.setFromOptions(); E.setUp()
    
    c = A.createVecRight(); c.setName("Coefficients")
    b = A.createVecRight(); b.setName("RHS Function Values")
    interp = E.createVecLeft(); interp.setName("interp")

    
    for row in range(*A.owner_range): # Rows are partioned
        for col in range(*A.owner_range): # Seperate the problem in purely local ones
            A.setValue(row, col, basisfunction(abs(supports[row]-supports[col])))
        b.setValue(row, testfunction(supports[row])) # Add the solution to the RHS

        # Add the polynomial
        if dimension:
            A.setValue(row, nSupport, 1) # Const part of the polynom
            A.setValue(nSupport, row, 1) # Ensure symmetricity
            for d in range(dimension-1):
                A.setValue(row, nSupport + 1 + d, i) # Value of support point
                A.setValue(nSupport + 1 + d, row, i)
            
    A.assemble()
    b.assemble()
    A.view()
    # A.view(PETSc.Viewer.DRAW().createDraw())
    # b.view()
    ksp = PETSc.KSP()
    ksp.create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, c)

    offset = E.owner_range[0]
    for row in range(*E.owner_range):
        for col in range(*E.owner_range):
            print(row, col, offset, eParts)
            E.setValue(row, col, basisfunction(abs(eParts[row-offset] - eParts[col-offset]))) # global to local

        # Add the polynomial
        if dimension:
            E.setValue(row, nSupport, 1)
            for d in range(dimension-1):
                E.setValue(row, nSupport + 1 + d, i)
    E.assemble()

    E.mult(c, interp);

    scatter, interp0 = PETSc.Scatter.toZero(interp)
    scatter.scatter(interp, interp0)
    scatter, c0 = PETSc.Scatter.toZero(c)
    scatter.scatter(c, c0)
    
    if MPIrank == 0:
        plot(supports, interp0.array, c0.array)
        
    sys.exit()



if __name__ == '__main__':
    main()


