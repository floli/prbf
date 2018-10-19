#pragma once

#include <string>
#include <utility>

#include "petscvec.h"
#include "petscmat.h"
#include "petscksp.h"
#include "petscis.h"

namespace petsc {

enum VIEWERFORMAT { ASCII, BINARY };

class Matrix;

class Vector
{
public:
  Vec vector = nullptr;

  enum LEFTRIGHT { LEFT, RIGHT };
  
  /// Creates a new vector on the given MPI communicator.
  explicit Vector(std::string name = "");

  /// Use Vec v as vector.
  Vector(Vec &v, std::string name = "");

  /// Duplicates type, row layout etc. (not values) of v.
  Vector(Vector &v, std::string name = "");  

  /// Constructs a vector with the same number of rows (default) or columns.
  Vector(Mat &m, std::string name = "", LEFTRIGHT type = LEFT);

  /// Constructs a vector with the same number of rows (default) or columns.
  Vector(Matrix &m, std::string name = "", LEFTRIGHT type = LEFT);

  /// Delete copy and assignment constructor
  /** Copying and assignement of this class would involve copying the pointer to
      the PETSc object and finallly cause double destruction of it.
   */
  Vector(const Vector&) = delete;
  Vector& operator=(const Vector&) = delete;

  ~Vector();

  /// Enables implicit conversion into a reference to a PETSc Vec type
  operator Vec&();

  /// Sets the size and calls VecSetFromOptions
  void init(PetscInt rows);

  int getSize();

  int getLocalSize();
  
  void setValue(PetscInt row, PetscScalar value);

  void arange(double start, double stop);

  void fillWithRandoms();

  /// Sorts the LOCAL partion of the vector
  void sort();

  void assemble();

  /// Returns a pair that mark the beginning and end of the vectors ownership range. Use first und second to access.
  std::pair<PetscInt, PetscInt> ownerRange();

  /// Writes the vector to file.
  void write(std::string filename, VIEWERFORMAT format = ASCII);

  /// Reads the vector from file.
  void read(std::string filename, VIEWERFORMAT format = ASCII);

  void view();
};

  
class Matrix
{
public:
  Mat matrix = nullptr;

  /// Delete copy and assignment constructor
  /** Copying and assignment of this class would involve copying the pointer to
      the PETSc object and finallly cause double destruction of it.
  */
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  explicit Matrix(std::string name = "");

  /// Move constructor, use the implicitly declared.
  Matrix(Matrix&& other) = default;

  ~Matrix();

  /// Enables implicit conversion into a reference to a PETSc Mat type
  operator Mat&();

  void assemble(MatAssemblyType type = MAT_FINAL_ASSEMBLY);
    
  /// Initializes matrix of given size and type
  /** @param[in] localRows,localCols The number of rows/cols that are local to the processor
      @param[in] globalRows,globalCols The number of global rows/cols.
      @param[in] type PETSc type of the matrix
      @param[in] doSetup Call MatSetup(). Not calling MatSetup can have performance gains when using preallocation
  */
  void init(PetscInt localRows, PetscInt localCols, PetscInt globalRows, PetscInt globalCols,
            MatType type = nullptr, bool doSetup = true);

  /// Destroys and recreates the matrix on the same communicator
  void reset();
  
  /// Get the MatInfo struct for the matrix.
  /** See http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatInfo.html for description of fields. */
  MatInfo getInfo(MatInfoType flag);
  
  void setValue(PetscInt row, PetscInt col, PetscScalar value);
  
  void fillWithRandoms();
  
  void setColumn(Vector &v, int col);

  /// Returns (rows, cols) global size
  std::pair<PetscInt, PetscInt> getSize();

  /// Returns (rows, cols) local size
  std::pair<PetscInt, PetscInt> getLocalSize();
  
  /// Returns a pair that mark the beginning and end of the matrix' ownership range.
  std::pair<PetscInt, PetscInt> ownerRange();
  
  /// Returns a pair that mark the beginning and end of the matrix' column ownership range.
  std::pair<PetscInt, PetscInt> ownerRangeColumn();

  /// Returns the block size of the matrix
  PetscInt blockSize() const;
  
  /// Writes the matrix to file.
  void write(std::string filename, VIEWERFORMAT format = ASCII);

  /// Reads the matrix from file, stored in PETSc binary format
  void read(std::string filename);

  /// Prints the matrix
  void view();

  void viewDraw();

};

class KSPSolver
{
public:
  KSP ksp;
  
  /// Delete copy and assignment constructor
  /** Copying and assignment of this class would involve copying the pointer to
      the PETSc object and finallly cause double destruction of it.
  */
  KSPSolver(const KSPSolver&) = delete;
  KSPSolver& operator=(const KSPSolver&) = delete;

  explicit KSPSolver(std::string name = "");

  /// Move constructor, use the implicitly declared.
  KSPSolver(KSPSolver&& other) = default;

  ~KSPSolver();
  
  /// Enables implicit conversion into a reference to a PETSc KSP type
  operator KSP&();
  
  /// Destroys and recreates the ksp on the same communicator
  void reset();

  /// Solves the linear system, returns false it not converged
  bool solve(Vector &b, Vector &x);
};


/// Destroys an KSP, if ksp is not null and PetscIsInitialized
void destroy(KSP * ksp);

/// Destroys an ISLocalToGlobalMapping, if IS is not null and PetscIsInitialized
void destroy(ISLocalToGlobalMapping * IS);


}
