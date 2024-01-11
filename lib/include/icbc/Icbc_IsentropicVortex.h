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
#ifndef ICBC_ISENTROPICVORTEX_HPP
#define ICBC_ISENTROPICVORTEX_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // The first test case is an isentropic vortex case (see e.g. the book by Hesthaven
  // and Warburton, Example 6.1 in Section 6.6 on page 209) which fulfills the
  // Euler equations with zero force term on the right hand side. 

  using namespace dealii;
  
  // We define global parameters that help in the definition of the initial
  // and boundary conditions. In  this case $gamma$ is defined also in the main 
  // program and we could have recoverd it from there. We redefine $gamma$ here for now.
  constexpr double gamma       = 1.4;



  // @sect3{Equation data}

  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions.
  template <int dim>  
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double time)
      : Function<dim>(dim + 2, time)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };  

  // We return either the density, the momentum, or the energy
  // depending on which component is requested. Note that the original
  // definition of the density involves the $\frac{1}{\gamma -1}$-th power of
  // some expression. Since `std::pow()` has pretty slow implementations on
  // some systems, we replace it by logarithm followed by exponentiation (of
  // base 2), which is mathematically equivalent but usually much better
  // optimized. This formula might lose accuracy in the last digits
  // for very small numbers compared to `std::pow()`, but we are happy with
  // it anyway, since small numbers map to data close to 1.
  //
  // For the channel test case, we simply select a density of 1, a velocity of
  // 0.4 in $x$ direction and zero in the other directions, and an energy that
  // corresponds to a speed of sound of 1.3 measured against the background
  // velocity field, computed from the relation $E = \frac{c^2}{\gamma (\gamma
  // -1)} + \frac 12 \rho \|u\|^2$.
  template <int dim>
  double ExactSolution<dim>::value(const Point<dim> & x,
                                   const unsigned int component) const
  {
    const double t = this->get_time();

    Assert(dim == 2, ExcNotImplemented());
    const double beta = 5;

    Point<dim> x0;
    x0[0] = 5.;
    const double radius_sqr =
      (x - x0).norm_square() - 2. * (x[0] - x0[0]) * t + t * t;
    const double factor =
      beta / (numbers::PI * 2) * std::exp(1. - radius_sqr);
    const double density_log = std::log2(
      std::abs(1. - (gamma - 1.) / gamma * 0.25 * factor * factor));
    const double density = std::exp2(density_log * (1. / (gamma - 1.)));
    const double u       = 1. - factor * (x[1] - x0[1]);
    const double v       = factor * (x[0] - t - x0[0]);

    if (component == 0)
      return density;
    else if (component == 1)
      return density * u;
    else if (component == 2)
      return density * v;
    else
      {
        const double pressure =
          std::exp2(density_log * (gamma / (gamma - 1.)));
        return pressure / (gamma - 1.) +
               0.5 * (density * u * u + density * v * v);
      }
  }



  // The `Ic` class define the initial condition for the test-case.
  // In this case it is recovered from the exact solution at time zero.
  // This is realized here thanks to a derived class of `ExactSolution` that 
  // overload the the constructor of the base class providing automatically 
  // a zero time
  template <int dim>  
  class Ic : public ExactSolution<dim>
  {
  public:
    Ic()
      : ExactSolution<dim>(0.)
    {}
  };



  // The `Bc` class define the boundary conditions for the test-case.
  template <int dim, int n_vars>  
  class BcIsentropicVortex : public BcBase<dim, n_vars>
  {
  public:
  
    BcIsentropicVortex(){};
    ~BcIsentropicVortex(){};
         
    void set_boundary_conditions() override;

  }; 

  // Dirichlet boundary conditions (inflow) are specified all around the domain.
  template <int dim, int n_vars>
  void BcIsentropicVortex<dim, n_vars>::set_boundary_conditions()
  {
     this->set_inflow_boundary(
       0, std::make_unique<ExactSolution<dim>>(0));
  }         



  // The `BodyForce` class define the body force for the test-case.
  // In this case it is null.
  template <int dim>  
  class BodyForce : public Function<dim>
  {
  public:
    BodyForce()
      : Function<dim>(dim)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double BodyForce<dim>::value(const Point<dim> & x,
                               const unsigned int component) const
  {
    (void)x;
    if (component == 1)
      return 0.;
    else
      return 0.;
  }
    
} // namespace ICBC

#endif //ICBC_ISENTROPICVORTEX_HPP
