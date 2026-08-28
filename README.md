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

## Travelling vortex with AMR

<img width="350" height="250" alt="travelling_vortex_contour" src="https://github.com/user-attachments/assets/920711b4-dd79-4c51-a4ba-30afd7147038" /> <img width="350" height="250" alt="travelling_vortex_msh" src="https://github.com/user-attachments/assets/cb19eefc-757f-4111-ac4b-0ecfeee0f580" />

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
Time:  0.0252, cells:     6173, dt:  0.00028, error free_surface:  1.301e-05, hu:  0.0001206
Time:  0.0501, cells:     6290, dt:  0.00028, error free_surface:   1.35e-05, hu:  0.0001457
Time:  0.0753, cells:     6350, dt:  0.00028, error free_surface:  1.457e-05, hu:  0.0001809
Time:     0.1, cells:     6398, dt:  0.00028, error free_surface:  1.569e-05, hu:    0.00022
Time:   0.125, cells:     6488, dt:  0.00028, error free_surface:  1.707e-05, hu:  0.0002613
Time:    0.15, cells:     6578, dt:  0.00028, error free_surface:  1.869e-05, hu:  0.0003049
Time:   0.167, cells:     6605, dt:  0.00028, error free_surface:  1.983e-05, hu:  0.0003338

+-------------------------------------------------+------------------+------------+------------------+
| Total wallclock time elapsed                    |     309.9s     3 |     309.9s |     309.9s     0 |
|                                                 |                  |                               |
| Section                             | no. calls |   min time  rank |   avg time |   max time  rank |
+-------------------------------------------------+------------------+------------+------------------+
| amr - remesh + remap                |        87 |     67.34s     2 |     67.43s |     67.53s     3 |
| compute errors                      |         8 |   0.07998s     2 |    0.1268s |     0.151s     3 |
| compute initial solution            |         1 |    0.9676s     0 |     0.969s |    0.9707s     2 |
| compute transport speed             |       119 |     1.577s     2 |     1.694s |     1.778s     0 |
| output solution                     |         8 |     1.557s     2 |     1.604s |     1.628s     3 |
| p-adaptation + remap                |         1 |   0.04552s     0 |   0.04611s |   0.04658s     3 |
| rk time stepping total              |       589 |     238.4s     3 |     238.5s |     238.8s     2 |
| rk_stage hydro - integrals L_h      |      1767 |     88.34s     2 |     90.02s |     90.84s     3 |
| rk_stage hydro - inv mass + vec upd |      1767 |     145.3s     3 |     146.3s |     148.2s     2 |
+-------------------------------------------------+------------------+------------+------------------+
```

The figure shows contour plots of the free-surface elevation (left) and the adapted mesh (right) at the final simulation time, using four levels of mesh refinement.

## Channel flow with sub-grid bathymetry

<img width="350" height="250" alt="plotOverLine_solution008_proje_zeta" src="https://github.com/user-attachments/assets/e310c49f-aba3-4f86-8fbc-6ac07c9ecd93" /> <img width="350" height="250" alt="plotOverLine_solution008_proje_u" src="https://github.com/user-attachments/assets/04438080-3605-4526-aaca-84af68226bef" />

We now examine the capability of the scheme to represent bathymetric obstacles that are not fully resolved at the grid scale. For this purpose, we consider a steady state solution of the one-dimensional shallow water equations with constant discharge, varying topography and friction. We consider a very coarse two-dimensional mesh with 10 elements along the x-direction. For such resolution,
the obstacles are unresolved at the grid scale.

Before running the code, the test-specific preprocessor and parameters must be set. Open the file `/oceano/source/main.cpp` and make the following changes:

```cpp
#define ICBC_CHANNELFLOW
```

which selects the initial conditions, boundary conditions, and source terms associated with this test;
```cpp
fe_degree = 3;
```

which selects the polynomial degree (r=3). At this point you need to recompile the code.

The directory `tests/channelFlow` contains the mesh file and the parameter file `channelFlow.prm`. The latter specifies the mesh, time-integration parameters, physical constants, and output options. To run the test, execute:

```bash
cd tests/channelFlow
mpirun -np 4 oceano -i channelFlow.prm

Running with 4 MPI processes
Vectorization over 2 doubles = 128 bits (SSE2)
Number of quadrature points along a line   :    5
Number of quadrature points in a cell      :   25
Number of quadrature points for mass-matrix:   25
Reading mesh file: channelFlow_10x2cells.msh
Initial number of cells:       20
Number of cells after global refinement:       20
Number of cells after local  refinement:       20
Initial number of degrees of freedom: 960, 3 [vars], 20 [cells], 16 [dofs/cell/var]
Time step size: 0.00531646, initial minimal h: 0.5, initial transport scaling: 0.0708861

Time:       0, cells:       20, dt:   0.0053, error free_surface:      1.523, hu:  1.631e-14
Time:      60, cells:       20, dt:   0.0053, error free_surface:     0.2469, hu:     0.4998
Time:     120, cells:       20, dt:   0.0053, error free_surface:     0.1731, hu:     0.1837
Time:     180, cells:       20, dt:   0.0053, error free_surface:     0.1758, hu:     0.1131
Time:     240, cells:       20, dt:   0.0053, error free_surface:     0.1744, hu:      0.109
Time:     300, cells:       20, dt:   0.0053, error free_surface:     0.1746, hu:     0.1087
Time:     360, cells:       20, dt:   0.0053, error free_surface:     0.1746, hu:     0.1087
Time:     420, cells:       20, dt:   0.0053, error free_surface:     0.1746, hu:     0.1087
Time:     480, cells:       20, dt:   0.0053, error free_surface:     0.1746, hu:     0.1087

+-------------------------------------------------+------------------+------------+------------------+
| Total wallclock time elapsed                    |     514.1s     2 |     514.1s |     514.1s     1 |
|                                                 |                  |                               |
| Section                             | no. calls |   min time  rank |   avg time |   max time  rank |
+-------------------------------------------------+------------------+------------+------------------+
| compute errors                      |         9 |   0.04119s     3 |    0.1508s |    0.3824s     0 |
| compute initial solution            |         1 |   0.01355s     1 |   0.01355s |   0.01355s     3 |
| compute transport speed             |     18192 |     4.974s     1 |     5.868s |     6.729s     2 |
| output solution                     |         9 |     30.99s     1 |     30.99s |     30.99s     0 |
| p-adaptation + remap                |        33 |    0.4252s     1 |     0.427s |    0.4287s     2 |
| rk time stepping total              |     90951 |     475.5s     2 |     476.3s |     477.2s     1 |
| rk_stage hydro - integrals L_h      |    272853 |     180.4s     1 |     190.5s |     199.2s     0 |
| rk_stage hydro - inv mass + vec upd |    272853 |     249.6s     0 |     258.2s |     269.1s     1 |
+-------------------------------------------------+------------------+------------+------------------+
```
The figure shows the free-surface elevation (left) and the velocity (right) which are close to the exact solution, in spite of the undersampling of the bathymetry at the grid scale.

## Thacker parabolic oscillations

<img width="350" height="290" alt="plotOverLine_zeta_low_N100" src="https://github.com/user-attachments/assets/ef65e918-85f8-436d-b6a9-96c814e73b8f" />
<img width="350" height="290" alt="plot_integralHistory_h" src="https://github.com/user-attachments/assets/e5ab1c42-ec81-4ce5-b055-74f808a2042b" />

We consider a periodic solution of the shallow water equations with a wet-dry transition. The free surface consists of a radially symmetric, oscillating paraboloid. Before running the code, the test-specific preprocessor definitions and parameters must be set. Open the file `/oceano/source/main.cpp` and make the following changes:

```cpp
#define ICBC_THACKEROSCILLATIONS2D
```

which selects the initial conditions, boundary conditions, and source terms associated with this test;

```cpp
#define OCEANO_WITH_MASSCONSERVATIONCHECK
```

which activates the mass-conservation diagnostics;

```cpp
fe_degree = 1;
```

which selects the polynomial degree (r=1). The use of higher-order polynomials does not provide significant advantages due to the $\mathcal{C}^1$ regularity of the velocity. After making these changes, recompile the code.

The directory `tests/thackerOscillations2d` contains the mesh file and the parameter file `thacker2d.prm`. In particular, the latter specifies the coarse mesh file, the mesh-refinement level (resulting in 100 elements in each direction), and the time-integration parameters. It also sets two wet-dry constants: the depth threshold used for polynomial-degree coarsening, set to $h_{\mathrm{lim}} = 10^{-2}\mathrm{m}$, and the depth threshold used for velocity desingularization, set to $\epsilon = 10^{-2}\mathrm{m}$.

To run the test, execute:

```bash
cd tests/thackerOscillations2d
mpirun -np 4 ./oceano -i thacker2d.prm
```

A successful run should produce output similar to the following:

```text
Running with 4 MPI processes
Vectorization over 2 doubles = 128 bits (SSE2)
Number of quadrature points along a line   :    3
Number of quadrature points in a cell      :    4
Number of quadrature points for mass-matrix:    9
Reading mesh file: thacker2d_25x25cells.msh
Initial number of cells:      400
Number of cells after global refinement:    6.400
Number of cells after local  refinement:    6.400
Initial number of degrees of freedom: 33.240, 3 [vars], 6.400 [cells], 4 [dofs/cell/var]
Time step size: 0.00846518, initial minimal h: 0.0375, initial transport scaling: 0.0338607

Time:       0, cells:     6400, dt:   0.0085, error free_surface:  0.0005621, hu:          0
Time:   0.562, cells:     6400, dt:   0.0091, error free_surface:  0.0008937, hu:  0.0006428
Time:    1.13, cells:     6400, dt:    0.011, error free_surface:   0.001208, hu:  0.0009634
Time:    1.69, cells:     6400, dt:   0.0092, error free_surface:   0.002473, hu:   0.001638
Time:    2.25, cells:     6400, dt:   0.0086, error free_surface:   0.003354, hu:   0.002181
Time:    2.81, cells:     6400, dt:   0.0093, error free_surface:   0.002124, hu:   0.003267
Time:    3.37, cells:     6400, dt:     0.01, error free_surface:    0.00483, hu:  0.0007625
Time:    3.93, cells:     6400, dt:   0.0092, error free_surface:    0.00235, hu:   0.003968
Time:    4.49, cells:     6400, dt:   0.0088, error free_surface:   0.007314, hu:   0.002176
Time:    5.05, cells:     6400, dt:   0.0094, error free_surface:   0.002075, hu:   0.005563
Time:    5.62, cells:     6400, dt:     0.01, error free_surface:   0.007614, hu:   0.000776
Time:    6.17, cells:     6400, dt:   0.0093, error free_surface:   0.002658, hu:   0.006011
Time:    6.73, cells:     6400, dt:   0.0089, error free_surface:    0.01032, hu:   0.001684

+-------------------------------------------------+------------------+------------+------------------+
| Total wallclock time elapsed                    |     735.9s     2 |     735.9s |     735.9s     0 |
|                                                 |                  |                               |
| Section                             | no. calls |   min time  rank |   avg time |   max time  rank |
+-------------------------------------------------+------------------+------------+------------------+
| compute errors                      |        13 |    0.1015s     1 |    0.1355s |     0.174s     3 |
| compute initial solution            |         1 |    0.5697s     3 |    0.5699s |    0.5703s     0 |
| compute transport speed             |       146 |     1.934s     0 |     1.947s |     1.959s     2 |
| output solution                     |        13 |     2.206s     1 |      2.24s |     2.278s     3 |
| p-adaptation + remap                |       722 |     477.4s     3 |     477.4s |     477.4s     2 |
| rk time stepping total              |       721 |     252.8s     2 |       253s |     253.2s     3 |
| rk_stage hydro - check mass         |      2163 |     7.481s     1 |     46.46s |     89.72s     3 |
| rk_stage hydro - integrals L_h      |      2163 |       114s     0 |     114.8s |     115.7s     2 |
| rk_stage hydro - inv mass + vec upd |      2163 |     44.62s     3 |      88.4s |     127.9s     1 |
+-------------------------------------------------+------------------+------------+------------------+
```
The figure compares the numerical and exact solutions during the last oscillation corresponding to the drying phase. In the same figure, the region close to the
wet-dry interface where the scheme is reverted to r=0 is also highlighted with a gray line. In figure we show the relative mass conservation error, only two iterations of the Newton method are required to ensure mass conservation within numerical round-off errors associated with double precision arithmetic.
