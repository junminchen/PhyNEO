# Obtaining the code

The plugin is hosted [on Github](https://github.com/andysim/MPIDOpenMMPlugin) and can be checked out with
``` bash
git clone git@github.com:andysim/MPIDOpenMMPlugin
```

# Dependencies

The code needs OpenMM version 7.6 or later (tested with 7.7), which can be
installed as follows.

## Installation of Dependencies via Conda

Conda is *strongly* recommended for managing the environment and dependencies;
after [downloading](https://conda.io/docs/download.html) anaconda (make sure
you use `bash` or `zsh`).

To install the 7.7 verion of OpenMM into its own Conda environment called `mpid`, run
``` bash
conda create -n mpid openmm=7.7 cudatoolkit=10.2 swig mdtraj -c conda-forge
```
Make sure you request the version of the CUDA toolkit supported on your
cluster.  This example uses GCC to build; the speed of the C++ compiler is
irrelevant, because the CUDA code is the only fast code available in this
plugin.  Although the reference platform will run, it is very slow and
designed for correctness.

## Building the plugin

The plugin uses CMake for building, so that should be install locally; it can
be obtained from Conda if you do not have it available.  Once CMake is
installed, you can build the code using commands similar to the following (the
exact type of modules and mechanisms to load them will vary from system to
system)::

``` bash
conda activate mpid

export OPENMM_INSTALL_DIR=~/anaconda3/envs/mpid

module load cuda/10.2
module load cmake
module load gcc/8.2

# From the MPIDOpenMMPlugin top level directory
mkdir build
cd build
CXX=g++ cmake .. -DCMAKE_INSTALL_PREFIX=$OPENMM_INSTALL_DIR -DPYTHON_EXECUTABLE=`which python` -DOPENMM_DIR=$OPENMM_INSTALL_DIR -DCMAKE_CXX_FLAGS='-std=c++11'
make -j 4
make test
make install
make PythonInstall
```
Note that we use GCC in this example, but the nature of the C++ compiler is not
important, as the faster kernels are implemented in CUDA and only the slow
reference implementation is available on regular CPUs.

Before running the code, make sure you load the conda environment and all
modules used for building when using the plugin.
``` bash
conda activate mpid
```
