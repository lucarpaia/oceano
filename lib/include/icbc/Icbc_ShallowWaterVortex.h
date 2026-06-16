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
#ifndef ICBC_SHALLOWWATERVORTEX_H
#define ICBC_SHALLOWWATERVORTEX_H

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
  //
  // Additionally you can add a tracer which is initalized as a linear function
  // of $y$.
#undef  ICBC_SHALLOWWATERVORTEX_REGULARITYP1
#define ICBC_SHALLOWWATERVORTEX_REGULARITYP2

  using namespace dealii;

  // We define constant parameters that help in the definition of the initial
  // and boundary conditions. These are the parameters of the vortex
  // such as the undisturbed water depth and the free-strem velocity.
  // We choose a channel $[0,1]\times[0,2]$ with a depth of 10. The vortex have an
  // amplitude of 1 and a radius less than half of the channel width.
  constexpr double h0      = 1.0;
  constexpr double uoo     = 6.0;
  // the depth at the center:
  constexpr double hmin    = 0.9;
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
                                 const unsigned int /*component*/) const
  {
    return 0.0;
  }



  // The exact solution is:
  // \begin{equation*}
  //  h(R) = h_{0} - \frac{4}{g}\rpth{\frac{4\varGamma r_{0}}{\pi}}^{2}\rpth{H\rpth{\pi/2} - H\rpth{\rho/2}},\quad \rho = \frac{\pi R}{r_{0}},
  // \end{equation*}
  // \begin{equation*}
  //  u = u_{\infty} - \rpth{y - y_{0}}\omega(\rho),
  // \end{equation*}
  // \begin{equation*}
  //  v = \rpth{x - x_{0}}\omega(\rho),
  // \end{equation*}
  // with $R$ denoting the distance from the vortex center $\rpth{x_{0}, y_{0}}$. Moreover,
  // $u_{\infty}$ is the background velocity oriented along the $x-$axis at which the vortex
  // is transported and $\omega(\rho) = 4\varGamma\cos^{4}\rpth{\rho/2},$ where $\varGamma$
  // is a free parameter that controls the vortex strength. It has been selected to have a
  // minimal depth of $h(0) = h_{\text{min}}$ at the center of the vortex. The definition of
  // $H(x)$ is the following
  // \begin{equation*}
  //  H(x) = \frac{35\cos(2x)}{384} + \frac{35x\sin(2x)}{192} + \cos^{6}(x)\rpth{\frac{\cos^{2}(x)}{64} + \frac{7}{288}} + \frac{35\cos^{2}(2x)}{3072} + \frac{35x^{2}}{256}
  //  + \frac{35x\cos(2x)\sin(2x)}{768} + \frac{x\cos^{5}(x)\sin x\rpth{\cos^{2}(x) + \frac{7}{6}}}{8}.
  // \end{equation*}
  // Notice that the initial profiles and the solution are $\mathcal{C}^{4}$ functions, which
  // is enough to test a fourth order accurate RK-DG scheme, according to  classical
  // interpolation estimates.
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
    ~ExactSolution() = default;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    double g;
  };

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
      return u;
    else if (component == 2)
      return v;
    else
      return 0.1 * x[1];
  }



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
    {}
    ~Ic() = default;
  };



  template <int dim, int n_vars>
  class BcShallowWaterVortex : public BcBase<dim, n_vars>
  {
  public:

    BcShallowWaterVortex(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcShallowWaterVortex() = default;

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
} // namespace ICBC
#endif //ICBC_SHALLOWWATERVORTEX_H
