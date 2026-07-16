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

Running with 4 MPI processes
Vectorization over 2 doubles = 128 bits (SSE2)
Number of quadrature points along a line   :    3
Number of quadrature points in a cell      :    4
Number of quadrature points for mass-matrix:    9
Reading mesh file: shallowWaterVortex_20x10cells.msh
Initial number of cells:      200
Number of cells after global refinement:      200
Number of cells after local  refinement:    5.528
Initial number of degrees of freedom: 66.336, 3 [vars], 5.528 [cells], 4 [dofs/cell/var]
Time step size: 0.00028269, initial minimal h: 0.00625, initial transport scaling: 0.0006282

Time:       0, cells:     5528, dt:  0.00028, error free_surface:  6.291e-16, hu:   4.93e-15
Time:  0.0252, cells:     6224, dt:  0.00028, error free_surface:  1.302e-05, hu:  0.0001207
Time:  0.0501, cells:     6248, dt:  0.00028, error free_surface:   1.35e-05, hu:  0.0001457
Time:  0.0753, cells:     6368, dt:  0.00028, error free_surface:  1.457e-05, hu:  0.0001809
Time:     0.1, cells:     6398, dt:  0.00028, error free_surface:  1.569e-05, hu:  0.0002199
Time:   0.125, cells:     6449, dt:  0.00028, error free_surface:  1.707e-05, hu:  0.0002612
Time:    0.15, cells:     6572, dt:  0.00028, error free_surface:  1.869e-05, hu:  0.0003048
Time:   0.167, cells:     6614, dt:  0.00028, error free_surface:  1.983e-05, hu:  0.0003338

+-------------------------------------------------+------------------+------------+------------------+
| Total wallclock time elapsed                    |     298.8s     3 |     298.8s |     298.8s     0 |
|                                                 |                  |                               |
| Section                             | no. calls |   min time  rank |   avg time |   max time  rank |
+-------------------------------------------------+------------------+------------+------------------+
| amr - remesh + remap                |        59 |     47.32s     2 |     47.42s |      47.5s     3 |
| compute errors                      |         8 |   0.09286s     1 |    0.1018s |    0.1168s     0 |
| compute initial solution            |         1 |    0.9383s     2 |    0.9384s |    0.9385s     0 |
| compute transport speed             |       119 |     1.516s     2 |     1.696s |     1.818s     3 |
| output solution                     |         8 |     1.518s     1 |     1.527s |     1.542s     0 |
| p-adaptation + remap                |         1 |   0.03032s     0 |   0.03032s |   0.03032s     2 |
| rk time stepping total              |       589 |     247.5s     3 |     247.7s |       248s     2 |
| rk_stage hydro - integrals L_h      |      1767 |     89.89s     2 |      92.4s |     94.12s     3 |
| rk_stage hydro - inv mass + vec upd |      1767 |       151s     3 |       153s |     155.8s     2 |
+-------------------------------------------------+------------------+------------+------------------+
```

The figure shows contour plots of the free-surface elevation (left) and the adapted mesh (right) at the final simulation time, using four levels of mesh refinement.
