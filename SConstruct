import os

env = Environment(ENV=os.environ)

env.Append(CPPPATH = ['/home/florian/software/petsc/include',
                      '/home/florian/software/petsc/arch-linux2-c-debug/include'])

env.Append(LIBPATH = ['/home/florian/software/petsc/arch-linux2-c-debug/lib'])
env.Append(LIBS = ['petsc'])

env.Append(CXXFLAGS = ['-O0','-g3', '-Wall', '-std=c++11'])
env.Replace(CXX = "/opt/mpich/bin/mpic++")
env["ENV"]["OMPI_CXX"] = "clang++"

env.Program(['petsctest.cpp', 'Petsc.cpp'])
env.Program(['prbf.cpp', 'Petsc.cpp'])

