/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Luca Arpaia,        2023
 */
#ifndef ICBC_IMPULSIVEWAVE_HPP
#define ICBC_IMPULSIVEWAVE_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // The first test case is an impulsive wave in a closed square basin with dimensions 
  // $[−5m, 5m] \times [−5m, 5m]$ and with constant bathymetry $z_b = 1 m$. 
  // The basin is initially at rest and the free surface is perturbed by 
  // the following Gaussian hump $\zeta(x, t = 0) = A \exp(−r^2 /\tau)$,
  // with $A = 0.5 m$, $\tau = 2.0 m^2 and $r =\sqrt{ x^2 + y^2 }$.
  using namespace dealii;
  
  // We define global parameters that help in the definition of the initial
  // and boundary conditions. They are the initial wave parameters, namely 
  // the amplitude $A$ and the decay rate $\tau$.
  constexpr double A         = 0.5;
  constexpr double tau       = 2.0;

  // lrp: to be removed
  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions.
  template <int dim, int n_vars>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double time)
      : Function<dim>(n_vars, time)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  // For the channel test case, we simply select a density of 1, a velocity of
  // 0.4 in $x$ direction and zero in the other directions, and an energy that
  // corresponds to a speed of sound of 1.3 measured against the background
  // velocity field, computed from the relation $E = \frac{c^2}{\gamma (\gamma
  // -1)} + \frac 12 \rho \|u\|^2$.
  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & x,
                                           const unsigned int component) const
  {
    (void)x;
    if (component == 0)
      return 1.;
    else if (component == 1)
      return 0.;
    else
      return 0.;
  }

  // @sect3{Equation data}
  //
  // The `Ic` class defines the initial condition for the test-case.
  // This is realized here thanks to a derived class of the deal.II `Function` class
  // that define many type of time and space functions. The initial condition class
  // overload the the constructor of the base class providing automatically
  // a zero time. Note that, apart for the template for the dimension which is in common with
  // the base `Function` class, we have added the number of variables to construct the base
  // class with the correct number of dimension and do some sanity checks.
  template <int dim, int n_vars>
  class Ic : public Function<dim>
  {
  public:
    Ic()
      : Function<dim>(n_vars, 0.)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  // We return either the water depth or the momentum
  // depending on which component is requested. Two sanity checks have been added. One is to
  // control that the space dimension is two (you cannot run this test in one dimension) and
  // another one on the number of variables, that for two-dimensional shallow water equation 
  // is three.
  template <int dim, int n_vars>
  double Ic<dim, n_vars>::value(const Point<dim>  &x,
                                const unsigned int component) const
  {
    Assert(dim == 2, ExcNotImplemented());
    Assert(n_vars == 3, ExcNotImplemented());
    
    Point<dim> x0;
    x0[0] = 0.; x0[1] = 0.;
    const double radius_sqr = (x - x0).norm_square();
    const double initial_wave = 1. + A*std::exp(-radius_sqr/tau);

    if (component == 0)
      return initial_wave;
    else if (component == 1)
      return 0.;
    else
      return 0.;
  }



  // The `Bc` class define the boundary conditions for the test-case.
  template <int dim, int n_vars>  
  class BcImpulsiveWave : public BcBase<dim, n_vars>
  {
  public:

    BcImpulsiveWave(){};
    ~BcImpulsiveWave(){};
         
    void set_boundary_conditions() override;

  };

  // A closed bassin means a wall-boundary condition at the
  // four boundaries.
  template <int dim, int n_vars>
  void BcImpulsiveWave<dim, n_vars>::set_boundary_conditions()
  {
    this->set_wall_boundary(0);
  }



  // This should become the `Data` class ... where all the spatial data specific to
  // the test-case are defined. This data are piled up into a vector.
  // Up to now there are two functions. The bathymetry at the first component and the
  // friction coefficient at the second component.
  // Also in this case we use as base class the deal.II `Function` class.
  // We consider a flat bottom bassin with a depth of $1m$. We have put the vertical
  // reference framework attached to the bottom, so the first conserved variable results
  // the sum of the free-surface plus the bathymetry (+1m). We can safely use a zero
  // bathymetry here. This test case is also frictionless.
  template <int dim>
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &/*prm*/)
    : Function<dim>(dim+3)
  {}
  
  

  template <int dim>
  double ProblemData<dim>::value(const Point<dim> & x,
                                 const unsigned int component) const
  {
    (void)x;
    if (component == 0)
      return 0.0;
    else
      return 0.0;
  }
    
} // namespace ICBC

#endif //ICBC_ISENTROPICVORTEX_HPP
