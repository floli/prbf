#!env python3

import ipdb

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from phelper import *

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
supportSpace = (0, 1)  # Range in which the support points are equally distributed

# Dimension of interpolation. Used for adding a polynomial to the matrix. Set to zero to deactivate polynomial
# f(x) = y with scalars x and y gives dimension = 2.
dimension = 0


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
        plot(supports, eMesh, interp0.array, c0.array, dimension)
        
    sys.exit()



if __name__ == '__main__':
    main()


