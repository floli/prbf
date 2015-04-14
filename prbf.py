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
         2: [np.linspace(0, 0.5, 5, False), # False == Do no include endpoint in range
             np.linspace(0.5, 1, 8 )],
         4: [np.linspace(0, 0.25, 5, False),
             np.linspace(0.25, 0.5, 5, False),
             np.linspace(0.5, 0.75, 5, False),
             np.linspace(0.75, 1, 5, False), ]
         }


MPIrank = MPI.COMM_WORLD.Get_rank()
MPIsize = MPI.COMM_WORLD.Get_size()

nSupport = 5        # Number of support points
supportSpace = (-0.1, 1.1)  # Range in which the support points are equally distributed

# Dimension of interpolation. Used for adding a polynomial to the matrix. Set to zero to deactivate polynomial
dimension = 1
polyparams = dimension+1 if dimension else 0

def main():
    shuffle_mesh(eMesh)
    ePoints = eMesh[MPIsize][MPIrank] # np.array of positions to evaluate
    supports = np.linspace(supportSpace[0], supportSpace[1], nSupport)
    sPoints = partitions(supports)[MPIrank]

    A = PETSc.Mat(); A.create()
    E = PETSc.Mat(); E.create()
    if MPIrank == MPIsize-1 and dimension > 0: # The last rank gets the polynomial rows
        A.setSizes( size = ((len(sPoints)+polyparams, PETSc.DETERMINE), (len(sPoints)+polyparams, PETSc.DETERMINE)) )
        E.setSizes( size = ((len(ePoints), PETSc.DETERMINE), (len(sPoints)+polyparams, PETSc.DETERMINE)) )
    else:
        A.setSizes( size = ((len(sPoints), PETSc.DETERMINE), (len(sPoints), PETSc.DETERMINE)) )
        E.setSizes( size = ((len(ePoints), PETSc.DETERMINE), (len(sPoints), PETSc.DETERMINE)) )
    A.setName("System Matrix");  A.setFromOptions(); A.setUp()
    E.setName("Evaluation Matrix");  E.setFromOptions(); E.setUp()
    
    c = A.createVecRight(); c.setName("Coefficients")
    b = A.createVecRight(); b.setName("RHS Function Values")
    interp = E.createVecLeft(); interp.setName("interp")

    for row in range(*A.owner_range): # Rows are partioned
        if row >= len(supports): break # We are not setting the rows for the polynomial, this is done when setting each column.
        for col in range(nSupport):
            v = basisfunction(abs(supports[row]-supports[col]))
            if v != 0 or row == col: # Set 0 explicitly only on the main diagonal, petsc requirement
                A.setValue(row, col, v)
        b.setValue(row, testfunction(supports[row])) # Add the solution to the RHS

        # Add the polynomial
        if dimension:
            A.setValue(row, nSupport, 1) # Const part of the polynom
            A.setValue(nSupport, row, 1) # Ensure symmetricity
            for d in range(dimension):
                A.setValue(row, nSupport + 1 + d, supports[row]) # Value of support point
                A.setValue(nSupport + 1 + d, row, supports[row])

        
    A.assemble()
    b.assemble()
    A.view()
    A.view(PETSc.Viewer.DRAW().createDraw()) # Use command line -draw_pause <sec>.

    Print("polyparams= ", polyparams)
    Print("A Size =", A.getSize())
    Print("E Global Size = ", E.getSize())
    Print("E Local  Size = ", E.getLocalSize())
    Print("E Owner Range", E.owner_range)

    offset = E.owner_range[0]
    for row in range(*E.owner_range):
        for col in range(E.getSize()[1]-polyparams):
            E.setValue(row, col, basisfunction(abs(ePoints[row-offset] - supports[col])))
        
        # Add the polynomial
        if dimension:
            E.setValue(row, nSupport, 1)
            for d in range(dimension):
                E.setValue(row, nSupport + 1 + d, ePoints[row-offset])

    E.assemble()
    E.view(PETSc.Viewer.DRAW().createDraw()) # Use command line -draw_pause <sec>.

    ksp = PETSc.KSP()
    ksp.create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, c)

    E.mult(c, interp);

    scatter, interp0 = PETSc.Scatter.toZero(interp)
    scatter.scatter(interp, interp0)
    scatter, c0 = PETSc.Scatter.toZero(c)
    scatter.scatter(c, c0)
    
    if MPIrank == 0:
        plot(supports, eMesh, interp0.array, c0.array, dimension)
        


if __name__ == '__main__':
    main()


