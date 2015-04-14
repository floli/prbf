#include "petnum.hpp"
#include "chelper.hpp"

const int dimensions = 1;
const int polyparams = dimensions ? dimensions+1 : 0;

const int nSupport = 5;
auto supports = linspace(-0.1, 1.1, nSupport);

void RBF()
{
  int MPIrank, MPIsize;
  MPI_Comm_rank(PETSC_COMM_WORLD, &MPIrank);
  MPI_Comm_size(PETSC_COMM_WORLD, &MPIrank);
  
  petsc::Matrix A(PETSC_COMM_WORLD, "System Matrix"), E(PETSC_COMM_WORLD, "Evaluation Matrix");
  if ( (MPIrank == MPIsize-1) and (dimensions > 0) ) {
    // A.init()
  }
  
}

int main(int argc, char *argv[])
{
  PetscInitialize(&argc, &argv, NULL, NULL);
  RBF();  
  PetscFinalize();  
  return 0;
}
