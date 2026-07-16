# What is this repository about?

It is a finite element shallow water solver for coastal applications based on the deal.ii library. It uses high order discontinuous finite elements and adaptive non-conforming quadrilateral meshes.

## Requirements

In order to compile and use the code you need to have the following programs installed:
- CMake (tested version 3.22 or later)
- a c++ compiler (tested gcc versions 11.2 and 11.4 and intel/oneapi-2021)
- mpi (tested openmpi version 5.0 and intelmpi/oneapi-2021)
- the finite element library deal.ii (version 9.5.1) with p4est for handling dynamic meshes across multiple processors. For both see https://www.dealii.org/current/readme.html.

## Quick compilation

Into the oceano repository, create a build directory:
```bash
cd oceano
mkdir build
cd build
ccmake ../
```
Set the `deal.II_DIR` variable to your deal.II install directory. Then generate the Makefile
and compile with:
```bash
make
```

## Travelling vortex

<img width="390" height="278" alt="travelling_vortex_contour" src="https://github.com/user-attachments/assets/920711b4-dd79-4c51-a4ba-30afd7147038" /> <img width="390" height="278" alt="travelling_vortex_msh" src="https://github.com/user-attachments/assets/cb19eefc-757f-4111-ac4b-0ecfeee0f580" />

We simulate a compactly supported travelling vortex that satisfies the shallow-water equations with a zero forcing term on the right-hand side.
Before running the code, some test-specific preprocessor definitions and parameters must be set. Open the file `/oceano/source/main.cpp` and make the following changes:

```cpp
#define ICBC_SHALLOWWATERVORTEX
```

which selects the initial conditions, boundary conditions, and source terms associated with this test;

```cpp
#define HPOCEANO_ERRORVORTICITY
```

which selects the vorticity-based refinement indicator;

```cpp
fe_degree = 1;
```

which selects the polynomial degree (r=1). At this point you need to recompile the code.

The directory `tests/shallowWaterVortex` contains the coarse-mesh file and the parameter file `shallowWaterVortex.prm`. The latter specifies the mesh and (hp)-adaptivity settings, time-integration parameters, physical constants, and output options. To run the test, execute:

```bash
cd tests/shallowWaterVortex
mpirun -np 4 oceano -i shallowWaterVortex.prm
```

The figure shows contour plots of the free-surface elevation (left) and the adapted mesh (right) at the final simulation time, using four levels of mesh refinement.
