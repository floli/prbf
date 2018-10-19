#include "Petsc.hpp"
#include <string>
#include <limits>
#include <random>
#include "petscviewer.h"
#include "petscdraw.h"

namespace petsc {

void openViewer(PetscViewer & viewer, std::string filename, VIEWERFORMAT format, MPI_Comm comm)
{
  PetscErrorCode ierr = 0;
  if (format == ASCII) {
    ierr = PetscViewerASCIIOpen(comm, filename.c_str(), &viewer);
    CHKERRV(ierr);
  }
  else if (format == BINARY) {
    ierr = PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_WRITE, &viewer);
    CHKERRV(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE);
    CHKERRV(ierr);
  }
}

template<class T>
MPI_Comm getCommunicator(T obj)
{
  MPI_Comm comm;
  PetscObjectGetComm(reinterpret_cast<PetscObject>(obj), &comm);
  return comm;
}

template<class T>
void setName(T obj, std::string name)
{
  PetscErrorCode ierr = 0;
  ierr = PetscObjectSetName(reinterpret_cast<PetscObject>(obj), name.c_str()); CHKERRV(ierr);
}

template<class T>
std::string getName(T obj)
{
  const char *cstr;
  PetscObjectGetName(reinterpret_cast<PetscObject>(obj), &cstr);
  return cstr;
}


/////////////////////////////////////////////////////////////////////////

Vector::Vector(std::string name)
{
  int size;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  PetscErrorCode ierr = 0;
  ierr = VecCreate(PETSC_COMM_WORLD, &vector); CHKERRV(ierr);
  setName(vector, name);
}

Vector::Vector(Vec &v, std::string name)
{
  VecCopy(v, vector);
  setName(vector, name);
}

Vector::Vector(Vector &v, std::string name)
{
  PetscErrorCode ierr = 0;
  ierr = VecDuplicate(v.vector, &vector); CHKERRV(ierr);
  setName(vector, name);
}

Vector::Vector(Mat &m, std::string name, LEFTRIGHT type)
{
  // MatGetVecs is deprecated, we keep it due to the old PETSc version at the SuperMUC.
  PetscErrorCode ierr = 0;
  if (type == LEFTRIGHT::LEFT) {
    ierr = MatCreateVecs(m, nullptr, &vector); CHKERRV(ierr); // a vector with the same number of rows
  }
  else {
    ierr = MatCreateVecs(m, &vector, nullptr); CHKERRV(ierr); // a vector with the same number of cols
  }
  setName(vector, name);
}

Vector::Vector(Matrix &m, std::string name, LEFTRIGHT type) :
  Vector(m.matrix, name, type)
{}

Vector::~Vector()
{
  PetscErrorCode ierr = 0;
  PetscBool petscIsInitialized;
  PetscInitialized(&petscIsInitialized);
  if (petscIsInitialized) // If PetscFinalize is called before ~Vector
    ierr = VecDestroy(&vector); CHKERRV(ierr);
}

Vector::operator Vec&()
{
  return vector;
}

void Vector::init(PetscInt rows)
{
  PetscErrorCode ierr = 0;
  ierr = VecSetSizes(vector, PETSC_DECIDE, rows); CHKERRV(ierr);
  ierr = VecSetFromOptions(vector); CHKERRV(ierr);
}

int Vector::getSize()
{
  PetscInt size;
  VecGetSize(vector, &size);
  return size;
}

int Vector::getLocalSize()
{
  PetscInt size;
  VecGetLocalSize(vector, &size);
  return size;
}

void Vector::setValue(PetscInt row, PetscScalar value)
{
  PetscErrorCode ierr = 0;
  ierr = VecSetValue(vector, row, value, INSERT_VALUES); CHKERRV(ierr);
}

void Vector::arange(double start, double stop)
{
  PetscErrorCode ierr = 0;
  PetscScalar *a;
  PetscInt range_start, range_end, size;
  VecGetSize(vector, &size);
  VecGetOwnershipRange(vector, &range_start, &range_end);
  double step_size = (stop-start) / size;
  ierr = VecGetArray(vector, &a); CHKERRV(ierr); 
  for (PetscInt i = range_start; i < range_end; i++) {
    a[i - range_start] = (i + start) * step_size;
  }
  VecRestoreArray(vector, &a);
}

void Vector::fillWithRandoms()
{
  PetscErrorCode ierr = 0;
  PetscRandom rctx;

  std::random_device rd;
  std::uniform_real_distribution<double> dist(0, 1);

  PetscRandomCreate(getCommunicator(vector), &rctx);
  PetscRandomSetType(rctx, PETSCRAND48);
  PetscRandomSetSeed(rctx, dist(rd));
  PetscRandomSeed(rctx);     
  ierr = VecSetRandom(vector, rctx); CHKERRV(ierr);
  PetscRandomDestroy(&rctx);
}

void Vector::sort() 
{
  PetscErrorCode ierr = 0;
  PetscInt size;
  PetscReal *a;
  ierr = VecGetArray(vector, &a); CHKERRV(ierr);
  ierr = VecGetSize(vector, &size);
  ierr = PetscSortReal(size, a); CHKERRV(ierr);
  ierr = VecRestoreArray(vector, &a); CHKERRV(ierr);
}

void Vector::assemble()
{
  PetscErrorCode ierr = 0;
  ierr = VecAssemblyBegin(vector); CHKERRV(ierr); 
  ierr = VecAssemblyEnd(vector); CHKERRV(ierr); 
}


std::pair<PetscInt, PetscInt> Vector::ownerRange()
{
  PetscInt range_start, range_end;
  VecGetOwnershipRange(vector, &range_start, &range_end);
  return std::make_pair(range_start, range_end);
}
  
void Vector::write(std::string filename, VIEWERFORMAT format)
{
  PetscErrorCode ierr = 0;
  PetscViewer viewer;
  openViewer(viewer, filename, format, getCommunicator(vector));
  VecView(vector, viewer); CHKERRV(ierr);
  PetscViewerDestroy(&viewer);
}

void Vector::read(std::string filename, VIEWERFORMAT format)
{
   PetscErrorCode ierr = 0;
   PetscViewer viewer;
   openViewer(viewer, filename, format, getCommunicator(vector));
   VecLoad(vector, viewer); CHKERRV(ierr); CHKERRV(ierr);
   PetscViewerDestroy(&viewer);
}

void Vector::view()
{
  PetscErrorCode ierr;
  ierr = VecView(vector, PETSC_VIEWER_STDOUT_WORLD); CHKERRV(ierr);
}

/////////////////////////////////////////////////////////////////////////

Matrix::Matrix(std::string name)
{
  PetscErrorCode ierr = 0;
  ierr = MatCreate(PETSC_COMM_WORLD, &matrix); CHKERRV(ierr);
  setName(matrix, name);
}


Matrix::~Matrix()
{
  PetscErrorCode ierr = 0;
  PetscBool petscIsInitialized;
  PetscInitialized(&petscIsInitialized);
  if (petscIsInitialized) // If PetscFinalize is called before ~Matrix
    ierr = MatDestroy(&matrix); CHKERRV(ierr);
}

Matrix::operator Mat&()
{
  return matrix;
}

void Matrix::assemble(MatAssemblyType type)
{
  PetscErrorCode ierr = 0;
  ierr = MatAssemblyBegin(matrix, type); CHKERRV(ierr);
  ierr = MatAssemblyEnd(matrix, type); CHKERRV(ierr);
}

void Matrix::init(PetscInt localRows, PetscInt localCols, PetscInt globalRows, PetscInt globalCols,
                  MatType type, bool doSetup)
{
  PetscErrorCode ierr = 0;
  if (type != nullptr) {
    ierr = MatSetType(matrix, type); CHKERRV(ierr);
  }
  ierr = MatSetSizes(matrix, localRows, localCols, globalRows, globalCols); CHKERRV(ierr);
  ierr = MatSetFromOptions(matrix); CHKERRV(ierr);
  if (doSetup)
    ierr = MatSetUp(matrix); CHKERRV(ierr);
}

void Matrix::reset()
{
  PetscErrorCode ierr = 0;
  std::string name = getName(matrix);
  ierr = MatDestroy(&matrix); CHKERRV(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &matrix); CHKERRV(ierr);
  setName(matrix, name);
}

MatInfo Matrix::getInfo(MatInfoType flag)
{
  MatInfo info;
  MatGetInfo(matrix, flag, &info);
  return info;
}

void Matrix::setValue(PetscInt row, PetscInt col, PetscScalar value)
{
  PetscErrorCode ierr = 0;
  ierr = MatSetValue(matrix, row, col, value, INSERT_VALUES); CHKERRV(ierr);
}

void Matrix::fillWithRandoms()
{
  PetscErrorCode ierr = 0;
  PetscRandom rctx;

  std::random_device rd;
  std::uniform_real_distribution<double> dist(0, 1);

  PetscRandomCreate(getCommunicator(matrix), &rctx);
  PetscRandomSetType(rctx, PETSCRAND48);
  PetscRandomSetSeed(rctx, dist(rd));
  PetscRandomSeed(rctx);     
  ierr = MatSetRandom(matrix, rctx); CHKERRV(ierr);
  PetscRandomDestroy(&rctx);
}

void Matrix::setColumn(Vector &v, int col)
{
  PetscErrorCode ierr = 0;
  const PetscScalar *vec;
  PetscInt range_start, range_end;
  VecGetOwnershipRange(v.vector, &range_start, &range_end);
  std::vector<PetscInt> irow(range_end - range_start);
  std::iota(irow.begin(), irow.end(), range_start);
      
  ierr = VecGetArrayRead(v.vector, &vec); CHKERRV(ierr);
  ierr = MatSetValues(matrix, range_end - range_start, irow.data(), 1, &col, vec, INSERT_VALUES); CHKERRV(ierr);
  ierr = VecRestoreArrayRead(v.vector, &vec); CHKERRV(ierr);
  ierr = MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY); CHKERRV(ierr); 
  ierr = MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY); CHKERRV(ierr); 
}

std::pair<PetscInt, PetscInt> Matrix::getSize()
{
  PetscInt m, n;
  MatGetSize(matrix, &m, &n);
  return std::make_pair(m, n);
}

std::pair<PetscInt, PetscInt> Matrix::getLocalSize()
{
  PetscInt m, n;
  MatGetLocalSize(matrix, &m, &n);
  return std::make_pair(m, n);
}

std::pair<PetscInt, PetscInt> Matrix::ownerRange()
{
  PetscInt range_start, range_end;
  MatGetOwnershipRange(matrix, &range_start, &range_end);
  return std::make_pair(range_start, range_end);
}

std::pair<PetscInt, PetscInt> Matrix::ownerRangeColumn()
{
  PetscInt range_start, range_end;
  MatGetOwnershipRangeColumn(matrix, &range_start, &range_end);
  return std::make_pair(range_start, range_end);
}

PetscInt Matrix::blockSize() const
{
  PetscErrorCode ierr = 0;
  PetscInt bs;
  ierr = MatGetBlockSize(matrix, &bs); CHKERRQ(ierr);
  return bs;
}

void Matrix::write(std::string filename, VIEWERFORMAT format)
{
  PetscErrorCode ierr = 0;
  PetscViewer viewer;
  openViewer(viewer, filename, format, getCommunicator(matrix));
  ierr = MatView(matrix, viewer); CHKERRV(ierr);
  PetscViewerDestroy(&viewer);
}

void Matrix::read(std::string filename)
{
   PetscErrorCode ierr = 0;
   PetscViewer viewer;
   openViewer(viewer, filename, BINARY, getCommunicator(matrix));
   ierr = MatLoad(matrix, viewer); CHKERRV(ierr);
   PetscViewerDestroy(&viewer);
}

void Matrix::view()
{
  PetscErrorCode ierr = 0;
  PetscViewer viewer;
  ierr = PetscViewerCreate(getCommunicator(matrix), &viewer); CHKERRV(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII); CHKERRV(ierr); 
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE); CHKERRV(ierr);
  ierr = MatView(matrix, viewer); CHKERRV(ierr);
  ierr = PetscViewerPopFormat(viewer); CHKERRV(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRV(ierr); 
}

void Matrix::viewDraw()
{
  PetscErrorCode ierr = 0;
  PetscViewer viewer;
  PetscDraw draw;
  ierr = PetscViewerCreate(getCommunicator(matrix), &viewer); CHKERRV(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERDRAW); CHKERRV(ierr); 
  ierr = MatView(matrix, viewer); CHKERRV(ierr);
  ierr = PetscViewerDrawGetDraw(viewer, 0, &draw); CHKERRV(ierr);
  ierr = PetscDrawSetPause(draw, -1); CHKERRV(ierr); // Wait for user
  ierr = PetscViewerDestroy(&viewer); CHKERRV(ierr);
}


/////////////////////////////////////////////////////////////////////////

KSPSolver::KSPSolver(std::string name)
{
  PetscErrorCode ierr = 0;
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRV(ierr);
  setName(ksp, name);
}


KSPSolver::~KSPSolver()
{
  PetscErrorCode ierr = 0;
  PetscBool petscIsInitialized;
  PetscInitialized(&petscIsInitialized);
  if (petscIsInitialized) // If PetscFinalize is called before ~KSPSolver
    ierr = KSPDestroy(&ksp); CHKERRV(ierr);
}

KSPSolver::operator KSP&()
{
  return ksp;
}

void KSPSolver::reset()
{
  PetscErrorCode ierr = 0;
  ierr = KSPReset(ksp); CHKERRV(ierr);
}

bool KSPSolver::solve(Vector &b, Vector &x)
{
  PetscErrorCode ierr = 0;
  KSPConvergedReason convReason;
  KSPSolve(ksp, b, x);
  ierr = KSPGetConvergedReason(ksp, &convReason); CHKERRQ(ierr);
  return (convReason > 0);
}


/////////////////////////////////////////////////////////////////////////

void destroy(ISLocalToGlobalMapping * IS)
{
  PetscErrorCode ierr = 0;
  PetscBool petscIsInitialized;
  PetscInitialized(&petscIsInitialized);
  
  if (IS and petscIsInitialized) {
    ierr = ISLocalToGlobalMappingDestroy(IS); CHKERRV(ierr);
  }
}


}
