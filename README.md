# What is this repository about?

It is a finite element shallow water solver based on the deal.ii library for coastal applications. It uses high order discontinuous finite elements and adaptive non-conforming quadrilateral meshes.

## Requirements

In order to compile and use the code you need to have the following programs installed:
- CMake (tested version 3.22 or later)
- a c++ compiler (tested gcc versions 11.2 and 11.4 and intel/oneapi-2021--binary)
- mpi (tested openmpi version 5.0 and intelmpi/oneapi-2021)
- the finite element library deal.ii (version 9.5.1) with p4est for handling dynamic meshes across multiple processors. For both see https://www.dealii.org/current/readme.html.

## Quick compilation

Go into the oceano repository and create a build directory:
```
$ cd oceano
$ mkdir build
$ cd build
$ ccmake ../
```
Set the `deal.II_DIR` variable to your deal.II install directory. Then generate the Makefile
and compile with:
```
$ make
```
