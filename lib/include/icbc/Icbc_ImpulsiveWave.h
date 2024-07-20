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



  // @sect3{Equation data}
  //
  // lrp: to be removed
  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions.
  template <int dim, int n_vars>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double time,
                  IO::ParameterHandler &/*prm*/)
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



  // The `Ic` and `Bc` classes define the initial/boundary condition for the
  // test-case. They are very similar in the templates and the constructor.
  // They both take as argument the parameter class and they stored it
  // internally. This means that we can read the Parameter file from
  // anywhere when we are implementing ic/bc and we can access constants or
  // filenames from which the initial/boundary data depends. In this case
  // we do not need any acces to the configuration file and it has been commented.
  // The initial condition is realized thanks to a derived class of the
  // deal.II `Function` class that define many type of time and space functions.
  // The initial condition class overloads the constructor of the base class
  // providing automatically a zero time. Note that, apart for the template for
  // the dimension which is in common with the base `Function` class, we have
  // added the number of variables to construct the base class with the correct
  // number of dimension and do some sanity checks. We return either the water
  // depth or the momentum depending on which component is requested. Two sanity
  // checks have been added. One is to control that the space dimension is two
  // (you cannot run this test in one dimension) and another one on the number
  // of variables, that for two-dimensional shallow water equation is three.
  // A closed bassin means a wall-boundary condition at the four boundaries.
  template <int dim, int n_vars>
  class Ic : public Function<dim>
  {
  public:
    Ic(IO::ParameterHandler &/*prm*/)
      : Function<dim>(n_vars, 0.)
    {}
    ~Ic(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

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



  template <int dim, int n_vars>  
  class BcImpulsiveWave : public BcBase<dim, n_vars>
  {
  public:

    BcImpulsiveWave(IO::ParameterHandler &/*prm*/){};
    ~BcImpulsiveWave(){};
         
    void set_boundary_conditions() override;

  };

  template <int dim, int n_vars>
  void BcImpulsiveWave<dim, n_vars>::set_boundary_conditions()
  {
    this->set_wall_boundary(0);
  }



  // We need a class to handle the problem data. Problem data are case dependent; for this 
  // reason it appears inside the `ICBC` namespace. The data in general depends on
  // both time and space. Deal.II has a class `Function` which returns function
  // of space and time, thus we simply create a derived class. The size of the data is 
  // fixed to `dim+3=5` scalar quantities. The first component is the bathymetry. 
  // The second is the bottom friction coefficient. The third and fourth components 
  // are the cartesian components of the wind velocity (in order, eastward and northward).
  // The fifth one is the Coriolis parameter. The test-dependent functions `stommelGyre_wind()`
  // and `stommelGyre_coriolis()` contain the definition of analytical functions for the 
  // different data. The call to `value()` returns all the external data necessary to 
  // complete the computation. 
  //
  // Finally the parameter handler class allows to read constants from the prm file.
  // The parameter handler class may seems redundant but it is not! Constants that appears
  // in you data may be easily recovered from the configuration file. More important file 
  // names which contains the may be imported too. 
  //
  // We can safely use a zero
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
