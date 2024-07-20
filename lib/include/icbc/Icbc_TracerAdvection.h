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
 * Author: Martin Kronbichler, 2020
 *         Luca Arpaia,        2023
 */
#ifndef ICBC_TRACERADVECTION_HPP
#define ICBC_TRACERADVECTION_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // This test consists in a flat-bottom channel with constant discharge and height.
  // The code should be able to preserve this initial state.
  // A concentration is realesed at the initial time upstream. The initial concentration
  // profile is advected with the flow without changing its shape. We can test the
  // accuracy of the discontinous Galerkin method.

  using namespace dealii;
  
  // We define global parameters that help in the definition of the initial
  // and boundary conditions. In this case the parameters of the initial concentration,
  // the channel water depth and the free-stream velocity.
  // follows. We choose a channel $[0,1]\times[0,2]$ with a depth of 1.
  // The initial concentration is a cosine bell with an
  // amplitude of 1 and a radius of one fourth of the channel width.
  constexpr double hoo     = 1.0;
  constexpr double uoo     = 1.0;
  // the depth at the center:  
  constexpr double cmax    = 1.0;
  // the radius:
  constexpr double radius0 = 0.25;



  // @sect3{Equation data}

  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions. Apart for the template for the
  // dimension which is in common with the base `Function` class, we have added
  // the number of variables. We return either the water depth, the momentum or
  // the concentration, depending on which component is requested.
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

  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & x,
                                           const unsigned int component) const
  {
    const double t = this->get_time();

    Assert(dim == 2, ExcNotImplemented());

    Point<dim> x0;
    x0[0] = 0.5 + uoo * t;
    x0[1] = 0.5;

    const double radius = (x - x0).norm();
    const double rho_half = 0.5*M_PI*radius/radius0;

    const double conc =
      radius < radius0 ? cmax * std::cos(rho_half) : 0.0;


    if (component == 0)
      return 0.;
    else if (component == 1)
      return hoo * uoo;
    else  if (component == 2)
      return 0.;
    else
      return hoo * conc;
  }



  // The `Ic` class define the initial condition for the test-case.
  // In this case it is recovered from the exact solution at time zero.
  // This is realized here thanks to a derived class of `ExactSolution` that
  // overload the the constructor of the base class providing automatically
  // a zero time
  template <int dim, int n_vars>
  class Ic : public ExactSolution<dim, n_vars>
  {
  public:
    Ic(IO::ParameterHandler &prm)
      : ExactSolution<dim, n_vars>(0., prm)
    {}
    ~Ic(){};
  };



  template <int dim, int n_vars>
  class BcTracerAdvection : public BcBase<dim, n_vars>
  {
  public:

    BcTracerAdvection(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcTracerAdvection(){};

    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  };

  // Dirichlet boundary conditions (inflow) are specified on the left boundary of the domain.
  // The right boundary is for outflow. Top and bottom boundaries are wall. Please note that,
  // for the vortex parameters given above, the flow is supercritical and the choice of
  // boundary conditions seems appropriate.
  template <int dim, int n_vars>
  void BcTracerAdvection<dim, n_vars>::set_boundary_conditions()
  {
    this->set_inflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    this->set_supercritical_outflow_boundary(
      2, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));

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
  // We do not have any source term and no associated data values.
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
  double ProblemData<dim>::value(const Point<dim> & /*x*/,
                                 const unsigned int /*component*/) const
  {
    return hoo;
  }

} // namespace ICBC

#endif //ICBC_TRACERADVECTION_HPP
