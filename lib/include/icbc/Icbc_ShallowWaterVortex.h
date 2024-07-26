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

  // The first test case is a compact shallow water travelling vortex (shortly RB-vortex)
  // (Ricchiuto and Bollerman, 2008) which fulfills the shallow water equations
  // with zero force term on the right hand side.
  // The RB-vortex is $C^4$ in the depth but only $C^1$
  // in the velocity. According to the classical interpolation estimate
  // of finite element theory, we can expect only second order of convergence for
  // the variables, included momentum. This vortex is thus suited to test second order schemes.
  // We have also coded an extension of the RB-vortex to an arbitrary degree of smoothness,
  // see (Ricchiuto and Torlo, 2021 arXiv:2109.10183v1). The implementation followed here
  // is the same provided in the last reference for the degree of smoothness $p=2$.
  // Other iterative corrections to improve the vortex smoothness and
  // test higher then third order schemes can be readily implemented.
#undef  ICBC_SHALLOWWATERVORTEX_REGULARITYP1
#define ICBC_SHALLOWWATERVORTEX_REGULARITYP2

  using namespace dealii;
  
  // We define constant parameters that help in the definition of the initial
  // and boundary conditions. These are the parameters of the vortex
  // such as the undisturbed water depth and the free-strem velocity.
  // We choose a channel $[0,1]\times[0,2]$ with a depth of 10. The vortex have an
  // amplitude of 1 and a radius less than half of the channel width.
  constexpr double h0      = 10.0;
  constexpr double uoo     = 6.0;
  // the depth at the center:  
  constexpr double hmin    = 9.0;
  // the radius:
  constexpr double radius0 = 0.25;



  // @sect3{Equation data}
  //
  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions. Apart for the template for the
  // dimension which is in common with the base `Function` class, we have added
  // the number of variables.
  template <int dim, int n_vars>  
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double time,
                  IO::ParameterHandler &prm)
      : Function<dim>(n_vars, time)
    {
      prm.enter_subsection("Physical constants");
      g = prm.get_double("g");
      prm.leave_subsection();
    }

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private: 
    double g;
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

#if defined ICBC_SHALLOWWATERVORTEX_REGULARITYP1
    const unsigned int p = 1;
#elif defined ICBC_SHALLOWWATERVORTEX_REGULARITYP2
    const unsigned int p = 2;
#endif
    const double corr = 4.;
    const double twoPowp = std::pow(2,p);

#if defined ICBC_SHALLOWWATERVORTEX_REGULARITYP1
    const double Gamma = M_PI/(twoPowp*radius0) * std::sqrt( 1./corr* g*(h0-hmin)/(0.046875*M_PI*M_PI-0.25));
#elif defined ICBC_SHALLOWWATERVORTEX_REGULARITYP2
    const double Gamma = M_PI/(twoPowp*radius0) * std::sqrt( 1./corr* g*(h0-hmin)/(0.034179687*M_PI*M_PI-0.222222222));
#endif
        
    const double radius = (x - x0).norm();

    const double pi_half = 0.5*M_PI;

    double cosx = std::cos(pi_half);
    double cos2x = std::cos(2.*pi_half);
    double sin2x = std::sin(2.*pi_half);
    double cosx2 = cosx*cosx;
    double cos2x2 = cos2x*cos2x;
    double x2 = pi_half*pi_half;

#if defined ICBC_SHALLOWWATERVORTEX_REGULARITYP1
    double H_pi_half = 0.125*cos2x + 0.25*pi_half*sin2x 
      + 0.015625*cos2x2 + 0.1875*x2 + 0.0625*pi_half*cos2x*sin2x;
#elif defined ICBC_SHALLOWWATERVORTEX_REGULARITYP2
    double sinx = std::sin(pi_half);
    double H_pi_half = 0.091145833*cos2x + 0.182291667*pi_half*sin2x 
      + std::pow(cosx,6)* (0.015625*cosx2 + 0.024305556) + 0.011393229*cos2x2 
      + 0.13671875*x2 + 0.045572917*pi_half*cos2x*sin2x + 0.125*pi_half*std::pow(cosx,5)*sinx*(cosx2 + 1.166666667);
#endif
    H_pi_half *= corr;

    const double rho_half = 0.5*M_PI*radius/radius0; 

    cosx = std::cos(rho_half);
    cos2x = std::cos(2.*rho_half);
    sin2x = std::sin(2.*rho_half);
    cosx2 = cosx*cosx;
    cos2x2 = cos2x*cos2x;
    x2 = rho_half*rho_half;

#if defined ICBC_SHALLOWWATERVORTEX_REGULARITYP1
    double H_rho_half = 0.125*cos2x + 0.25*rho_half*sin2x 
      + 0.015625*cos2x2 + 0.1875*x2 + 0.0625*rho_half*cos2x*sin2x;
#elif defined ICBC_SHALLOWWATERVORTEX_REGULARITYP2
    sinx = std::sin(rho_half);
    double H_rho_half = 0.091145833*cos2x + 0.182291667*rho_half*sin2x 
      + std::pow(cosx,6)* (0.015625*cosx2 + 0.024305556) + 0.011393229*cos2x2 
      + 0.13671875*x2 + 0.045572917*rho_half*cos2x*sin2x + 0.125*rho_half*std::pow(cosx,5)*sinx*(cosx2 + 1.166666667);
#endif
    H_rho_half *= corr;

    const double inv_gpi = 1./(g * M_PI * M_PI);
    const double omega = twoPowp * Gamma * std::pow(cosx2,p);
    const double num = twoPowp * Gamma * radius0;
        
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



  // The `Ic` and `Bc` classes define the initial/boundary condition for the
  // test-case. They are very similar in the templates and the constructor.
  // They both take as argument the parameter class and they stored it
  // internally. This means that we can read the Parameter file from
  // anywhere when we are implementing ic/bc and we can access constants or
  // filenames from which the initial/boundary data depends.
  // In this case the intial condition is recovered from the exact solution 
  // at time zero. This is realized here thanks to a derived class of
  // `ExactSolution` that overload the the constructor of the base class 
  // providing automatically a zero time.
  // Dirichlet boundary conditions (inflow) are specified on the left boundary of the domain.
  // The right boundary is for outflow. Top and bottom boundaries are wall. Please note that,
  // for the vortex parameters given above, the flow is supercritical and the choice of 
  // boundary conditions seems appropriate.
  template <int dim, int n_vars>  
  class Ic : public ExactSolution<dim, n_vars>
  {
  public:
    Ic(IO::ParameterHandler &prm)
      : ExactSolution<dim, n_vars>(0.,prm)
  //    , prm(prm)
    {}
    ~Ic(){};

  //private:
  //  ParameterHandler &prm;
  };



  template <int dim, int n_vars>  
  class BcShallowWaterVortex : public BcBase<dim, n_vars>
  {
  public:
 
    BcShallowWaterVortex(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcShallowWaterVortex(){};
         
    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  }; 

  template <int dim, int n_vars>
  void BcShallowWaterVortex<dim, n_vars>::set_boundary_conditions()
  {
    this->set_supercritical_inflow_boundary(
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
    return 0.0;
  }
    
} // namespace ICBC

#endif //ICBC_SHALLOWWATERVORTEX_HPP
