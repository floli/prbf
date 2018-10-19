#include <iostream>
#include <map>

#include "petscksp.h"
#include "Petsc.hpp"
#include "chelper.hpp"
#include "prettyprint.hpp"

using std::cout;
using std::endl;

const int dimension = 1;
const int polyparams = dimension ? dimension+1 : 0;

const int nSupport = 10;
auto supports = linspace(-0.1, 1.1, nSupport);

std::map<int, std::vector<std::vector<double>>> eMesh = {
  {1, {linspace(0.0, 1.0, 4)} },
  {2, {linspace(0.0, 0.5, 5, false),
       linspace(0.5, 1.0, 8)} },
  {4, {linspace(0.0, 0.25, 5.0, false),
       linspace(0.25, 0.5, 5.0, false),
       linspace(0.5, 0.75, 5.0, false),
       linspace(0.75, 1.0, 5.0, false)} }
};


void RBF()
{
  int MPIrank, MPIsize;
  MPI_Comm_rank(PETSC_COMM_WORLD, &MPIrank);
  MPI_Comm_size(PETSC_COMM_WORLD, &MPIsize);

  auto sPoints = partition(supports, MPIsize)[MPIrank];
  auto ePoints = eMesh[MPIsize][MPIrank];

  petsc::Matrix A("System Matrix"), E("Evaluation Matrix");
  if ( (MPIrank == MPIsize-1) and (dimension > 0) ) {
    A.init( sPoints.size()+polyparams, sPoints.size()+polyparams, PETSC_DECIDE, PETSC_DECIDE );
    E.init( ePoints.size(), sPoints.size()+polyparams, PETSC_DECIDE, PETSC_DECIDE );
  }
  else {
    A.init( sPoints.size(), sPoints.size(), PETSC_DECIDE, PETSC_DECIDE );
    E.init( ePoints.size(), sPoints.size(), PETSC_DECIDE, PETSC_DECIDE );
  }
  petsc::Vector c(A, "Coefficients", petsc::Vector::RIGHT);
  petsc::Vector b(A, "RHS Function Values", petsc::Vector::RIGHT);
  petsc::Vector interp(E, "Interpolation Results", petsc::Vector::LEFT);

  PetscInt rangeStart, rangeEnd;
  std::tie(rangeStart, rangeEnd) = A.ownerRange();

  cout << "[" << MPIrank << "] Original Range Start / End = " << rangeStart << ", " << rangeEnd << endl;

  // Introduce some overlap for testing.
  rangeStart -= 2;
  rangeEnd += 2;
  if (rangeStart < 0) rangeStart = 0;
  if (rangeEnd > nSupport+polyparams) rangeEnd = nSupport+polyparams;
  
  cout << "[" << MPIrank << "] Range Start / End = " << rangeStart << ", " << rangeEnd << endl;
    
  for (PetscInt row = rangeStart; row < rangeEnd; row++) {
    if (row >= supports.size())
      break;
    for (PetscInt col = 0; col < nSupport; col++) {
      auto v = basisfunction( std::abs(supports[row]-supports[col]));
      if (v != 0)
        A.setValue(row, col, v);
    }
    b.setValue(row, testfunction(supports[row]));

    if (dimension) { // add the polynomial
      A.setValue(row, nSupport, 1);
      A.setValue(nSupport, row, 1); // Ensure symmetricity, might not be needed when sbaij is used
      for (PetscInt d = 0; d < dimension; d++ ) {
        A.setValue(row, nSupport + 1 + d, supports[row]);
        A.setValue(nSupport + 1 + d, row, supports[row]);
      }
    }
  }
  A.assemble(MAT_FLUSH_ASSEMBLY);
  petsc::Vector zeros(A); // Petsc requires that all diagonal entries are set, even if set to zero.
  MatDiagonalSet(A.matrix, zeros.vector, ADD_VALUES);
  A.assemble();
  b.assemble();
  // A.view();
  // A.viewDraw();  

  std::tie(rangeStart, rangeEnd) = E.ownerRange();
  for (PetscInt row = rangeStart; row < rangeEnd; row++) {
    for (PetscInt col = 0; col < E.getSize().second; col++)
      E.setValue(row, col, basisfunction( std::abs(ePoints[row-rangeStart]-supports[col])) );

    if (dimension) { // add the polynomail
      E.setValue(row, nSupport, 1);
      for (PetscInt d = 0; d < dimension; d++) {
        E.setValue(row, nSupport + 1 + d, ePoints[row-rangeStart]);
      }
    }
  }
  E.assemble();
  // E.view();
  // E.viewDraw();
  // b.view(); 
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, A.matrix, A.matrix);
  KSPSetFromOptions(ksp);

  KSPSolve(ksp, b.vector, c.vector);

  MatMult(E.matrix, c.vector, interp.vector);
  c.view();
  interp.view();
}




int main(int argc, char *argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  RBF();  
  PetscFinalize();  
  return 0;
}
