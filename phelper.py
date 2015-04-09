from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

MPIrank = MPI.COMM_WORLD.Get_rank()
MPIsize = MPI.COMM_WORLD.Get_size()

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

    
def plot(supports, eMesh, interp, coeffs, dimension):
    """ Support Points, Evaluation Point, Interpolation Results, Coefficients"""
    evals =  np.concatenate( [i for i in eMesh[MPIsize]] )
    # sRange = np.linspace(supportSpace[0]-0.2, supportSpace[1]+0.2, 1000)  # Range a bit larget than the support space
    sRange = np.linspace(min(supports)-0.2, max(supports)+0.2, 1000)
    
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