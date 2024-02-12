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
#ifndef ICBC_SHALLOWWATERVORTEX_HPP
#define ICBC_SHALLOWWATERVORTEX_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // The first test case is a shallow water travelling compact vortex 
  // (Ricchiuto and Bollerman, 2008) which fulfills the shallow water equations 
  // with zero force term on the right hand side. 
  // The RB vortex which is $C^4$ in the depth but only $C^1$ 
  // in the velocity. According to the classical interpolation estimate
  // of finite element theory, we can expect only second order of convergence for
  // the variables, included momentum. This vortex is thus suited to test second order schemes.
  // For an extension of the RB-vortex to an arbitrary degree of smoothness, 
  // see Ricchiuto and Torlo, 2021 arXiv:2109.10183v1. The implementation followed here
  // is the the same provided in the last reference for the lowest degree of smoothness (p=1).  
  // The iterative corrections to improve the vortex smoothness and 
  // tests higher then second order schemes can be readily implemented.
  
  using namespace dealii;
  
  // We define global parameters that help in the definition of the initial
  // and boundary conditions. In  this case $g$ is defined also in the main 
  // program and we could have recoverd it from there. We redefine $g$ here for now.
  constexpr double g       = 1.0;
  // The others are the parameters of the vortex, the undisturbed water depth and velocity.
  // Wee choose a shallow channel $[0,1]\times[0,2]$ with depth of 0.1. The vortex have a 
  // small amplitude of 0.01 and radius slightly less than half of the channel width.
  constexpr double h0      = 1.0;
  constexpr double uoo     = 1.0;  
  // the depth at the center:  
  constexpr double hmin    = 0.99;
  // the radius
  constexpr double radius0 = 0.45;



  // @sect3{Equation data}

  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions. Apart for the template for the
  // dimension which is in common with the base `Function` class, we have added
  // the number of variables. 
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

  // We return either the water depth or the momentum
  // depending on which component is requested.
  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & x,
                                           const unsigned int component) const
  {
    const double t = this->get_time();

    Assert(dim == 2, ExcNotImplemented());

    Point<dim> x0;
    x0[0] = 0.5 + uoo *t; 
    x0[1] = 0.5;

    //const unsigned int p = 1; ! we code only p=1 case
    const double twoPowp = 2; //std::pow(2,p);
    
    //const double Gamma = M_PI/(twoPowp*radius0) * std::sqrt( g*(h0-hmin)/(0.034179687*M_PI*M_PI-0.222222222)); !p=2
    const double Gamma = M_PI/(twoPowp*radius0) * std::sqrt( g*(h0-hmin)/(4.*(0.046875*M_PI*M_PI-0.25)));

        
    const double radius = (x - x0).norm();

    const double pi_half = 0.5*M_PI;

    double cosx = std::cos(pi_half);
    double cos2x = std::cos(2.*pi_half);
    double sin2x = std::sin(2.*pi_half);
    double cosx2 = cosx*cosx;
    double cos2x2 = cos2x*cos2x;
    double x2 = pi_half*pi_half;
    
    //double H_pi_half = 0.091145833*cos2x + 0.182291667*pi_half*sin2x 
    //  + std::pow(cosx,6)* (0.015625*cosx2 + 0.024305556) + 0.011393229*cos2x2 
    //  + 0.13671875*x2 + 0.045572917*pi_half*cos2x*sin2x + 0.125*pi_half*std::pow(cosx,5)*sinx*(cosx2 + 1.166666667); !p=2
    double H_pi_half = 0.125*cos2x + 0.25*pi_half*sin2x 
      + 0.015625*cos2x2 + 0.1875*x2 + 0.0625*pi_half*cos2x*sin2x;

    const double rho_half = 0.5*M_PI*radius/radius0; 

    cosx = std::cos(rho_half);
    cos2x = std::cos(2.*rho_half);
    sin2x = std::sin(2.*rho_half);
    cosx2 = cosx*cosx;
    cos2x2 = cos2x*cos2x;
    x2 = rho_half*rho_half;

    //double H_rho_half = 0.091145833*cos2x + 0.182291667*rho_half*sin2x 
    //  + std::pow(cosx,6)* (0.015625*cosx2 + 0.024305556) + 0.011393229*cos2x2 
    //  + 0.13671875*x2 + 0.045572917*rho_half*cos2x*sin2x + 0.125*rho_half*std::pow(cosx,5)*sinx*(cosx2 + 1.166666667); !p=2
    double H_rho_half = 0.125*cos2x + 0.25*rho_half*sin2x 
      + 0.015625*cos2x2 + 0.1875*x2 + 0.0625*rho_half*cos2x*sin2x;

    const double inv_gpi = 1./(g * M_PI * M_PI);
    const double omega = twoPowp * Gamma * cosx2; //* std::pow(cosx2,p); !p=2
    const double num = twoPowp * 2. * Gamma * radius0;
        
    const double depth =
      radius < radius0 ? h0 - inv_gpi * num * num * (H_pi_half - H_rho_half) : h0;
    const double u     = 
      radius < radius0 ? uoo - (x[1]-x0[1]) * omega : uoo;
    const double v     = 
      radius < radius0 ? (x[0]-x0[0]) * omega : 0.;             

    if (component == 0)
      return depth;
    else if (component == 1)
      return depth * u;
    else
      return depth * v;
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
    Ic()
      : ExactSolution<dim, n_vars>(0.)
    {}
  };



  // The `Bc` class define the boundary conditions for the test-case.
  template <int dim, int n_vars>  
  class BcShallowWaterVortex : public BcBase<dim, n_vars>
  {
  public:
  
    BcShallowWaterVortex(){};
    ~BcShallowWaterVortex(){};
         
    void set_boundary_conditions() override;

  }; 

  // Dirichlet boundary conditions (inflow) are specified all around the domain.
  template <int dim, int n_vars>
  void BcShallowWaterVortex<dim, n_vars>::set_boundary_conditions()
  {
    this->set_inflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0));
    this->set_subsonic_outflow_boundary(
      2, std::make_unique<ExactSolution<dim, n_vars>>(0));

    this->set_wall_boundary(0);
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

#endif //ICBC_SHALLOWWATERVORTEX_HPP
