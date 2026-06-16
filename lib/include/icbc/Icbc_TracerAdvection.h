/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2026 by CNR-ISMAR
 *
 * This code, as the deal.II library is free software; you can use it,
 * redistribute it, and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later
 * version. The full text of the license can be found in the file
 * LICENSE.md at the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Luca Arpaia, 2023
 *         Giuseppe Orlando, 2026
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
  constexpr double hoo     = 0.5;
  constexpr double uoo     = 1.0;
  // the depth at the center:
  constexpr double cmax    = 1.0;
  // the radius:
  constexpr double radius0 = 0.25;



  // @sect3{Equation data}
  //
  // We do not have any source term and no associated data values.
  template <int dim>
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData() = default;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &/*prm*/)
    : Function<dim>(dim+3)
  {}



  template <int dim>
  double ProblemData<dim>::value(const Point<dim> & /*x*/,
                                 const unsigned int component) const
  {
    if (component == 0)
      return hoo;
    else
      return 0.;
  }



  // The exact solution is simple in this case, since the concentration is simply
  // advected with the same shape.
  template <int dim, int n_vars>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double time,
                  IO::ParameterHandler &/*prm*/)
      : Function<dim>(n_vars, time)
    {}
    ~ExactSolution() = default;

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
      return uoo;
    else if (component == 2)
      return 0.;
    else if (component == 3)
      return conc;
    else
      return conc * 2.0;
  }



  // In this case the initial condition is recovered from the exact solution
  // at time zero. This is realized here thanks to a derived class of `ExactSolution`
  // that overload the the constructor of the base class providing automatically
  // a zero time. If multiple
  // concentrations are advected, different initial conditions are assigned to
  // the each concentration species.
  template <int dim, int n_vars>
  class Ic : public ExactSolution<dim, n_vars>
  {
  public:
    Ic(IO::ParameterHandler &prm)
      : ExactSolution<dim, n_vars>(0., prm)
    {}
    ~Ic(){};
  };



  // Dirichlet boundary conditions (inflow) are specified on the left boundary of the domain.
  // The right boundary is for outflow. Top and bottom boundaries are wall. Please note that,
  // for the vortex parameters given above, the flow is supercritical and the choice of
  // boundary conditions seems appropriate.
  template <int dim, int n_vars>
  class BcTracerAdvection : public BcBase<dim, n_vars>
  {
  public:

    BcTracerAdvection(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcTracerAdvection() = default;

    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  };

  template <int dim, int n_vars>
  void BcTracerAdvection<dim, n_vars>::set_boundary_conditions()
  {
    this->set_supercritical_inflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    this->set_supercritical_outflow_boundary(
      2, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));

    this->set_wall_boundary(0);
  }
} // namespace ICBC

#endif //ICBC_TRACERADVECTION_HPP
