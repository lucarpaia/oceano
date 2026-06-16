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
#ifndef ICBC_THACKEROSCILLATIONS2D_HPP
#define ICBC_THACKEROSCILLATIONS2D_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // Thacker in 1981 found an exact solution of the shallow water equations
  // with wetting and drying. In fact, the shallow water equations admit analytic
  // periodic solutions (there is no damping) of a flood wave, eventually with Coriolis
  // effect that here is neglected. The test implemented is the is the Thacker solution
  // with a curved surface with the parameter proposed in the SWASH test suite (Delestre,2016).
  // With respect to the latter we only invert the sign of the bathymetry which is measured
  // positive downward.

  using namespace dealii;

  // We define constant parameters that help in the definition of the initial
  // and boundary conditions. We use the same notation and parameter value of the
  // SWASH test case.
  constexpr double h0      = 0.1;
  constexpr double a       = 1.0;
  constexpr double radius0 = 0.8;
  constexpr double L       = 3.0;



  // @sect3{Equation data}
  //
  // For this case we need to define the bathyemtry data value with parabolic function:
  //%
  //$$z_{b}(R)= h_{0}\rpth{1-\frac{R^{2}}{a^{2}}},$$
  //%
  //with $R$ denoting the distance from the center of the domain $\rpth{L/2,L/2}$.
  template <int dim>
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData() = default;

    inline double thackerOscillations2d_bathymetry(const Point<dim> & p) const;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &/*prm*/)
    : Function<dim>(dim+3)
  {}


  template <int dim>
  inline double ProblemData<dim>::thackerOscillations2d_bathymetry(
    const Point<dim> & x) const
  {
    Point<dim> x0;
    x0[0] = 0.5 *L;
    x0[1] = 0.5 *L;
    const double radius = (x - x0).norm();
    const double inv_a = 1./a;

    return h0 * (1. - inv_a*inv_a * radius*radius);
  }


  template <int dim>
  double ProblemData<dim>::value(const Point<dim> & x,
                                 const unsigned int component) const
  {
    if (component == 0)
      return thackerOscillations2d_bathymetry(x);
    else
      return 0.0;
  }



  // The class `ExactSolution` defines the Thacker analytical function:
  //
  // \begin{equation*}
  // \zeta(R,t) =
  //   - h_{0}\spth{\frac{\sqrt{1 - A^{2}}}{1-A\cos\rpth{\omega t}} - 1 - \frac{R^{2}}{a^{2}}\rpth{\rpth{\frac{\sqrt{1 - A^{2}}}{1-A\cos\rpth{\omega t}}}^{2} -1}}
  // \end{equation*}
  // \begin{equation*}
  // u(x,t) =
  //   \frac{1}{1 - A\cos\rpth{\omega t}}\rpth{\frac{A\omega}{2}\rpth{x - \frac{L}{2}}\sin\rpth{\omega t}},
  // \end{equation*}
  // \begin{equation*}
  // v(y,t) =
  //   \frac{1}{1 - A\cos\rpth{\omega t}}\rpth{\frac{A\omega}{2}\rpth{y - \frac{L}{2}}\sin\rpth{\omega t}},
  // \end{equation*}
  //
  // where $\omega = \frac{\sqrt{8gh_{0}}}{a}$ and $A = \frac{a^{2} - r_{0}^{2}}{a^{2} + r_{0}^{2}}$.
  //
  // We initialize dry cells with a virtual free-surface under the bathymetry
  // level and it is up to the non-linear algorithm that we use into the continuity
  // equation to find the "good" water level.
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
    x0[0] = 0.5 *L;
    x0[1] = 0.5 *L;
    const double radius = (x - x0).norm();

    const double inv_a = 1./a;
    const double omega = std::sqrt(8.*g*h0) * inv_a;
    const double a2 = a*a;
    const double radius02 = radius0*radius0;
    const double radius2 = radius*radius;
    const double A = (a2 - radius02)/(a2 + radius02);
    const double denom = 1./(1. - A * std::cos(omega*t));
    const double tmp = std::sqrt(1.0 - A*A) * denom;
    const double inv_temp = 1./tmp;

    double zb;
    double h;
    double u;
    double v;
    if (radius > a * std::sqrt(inv_temp))
      {
        const double r2 = a2 * inv_temp;
        zb = h0 * (1. - inv_a*inv_a * r2);
        h = 0.;
        u = 0.;
        v = 0.;
      }
    else
      {
        zb = h0 * (1. - inv_a*inv_a * radius2);
        h = h0 * (tmp - 1. - radius2*inv_a*inv_a *(tmp*tmp - 1.)) + zb;
        u = denom * 0.5 * omega * (x[0]-x0[0]) * A * std::sin(omega*t);
        v = denom * 0.5 * omega * (x[1]-x0[1]) * A * std::sin(omega*t);
      }

    if (component == 0)
      return h - zb;
    else if (component == 1)
      return u;
    else if (component == 2)
      return v;
    else
      return 1.;
  }



  // In this case the initial condition is recovered from the exact solution
  // at time zero. This is realized here thanks to a derived class of
  // `ExactSolution` that overload the the constructor of the base class
  // providing automatically a zero time.
  template <int dim, int n_vars>
  class Ic : public ExactSolution<dim, n_vars>
  {
  public:
    Ic(IO::ParameterHandler &prm)
      : ExactSolution<dim, n_vars>(0.,prm)
    {}
    ~Ic() = default;
  };



  // The wave does not interact with the boundaries so we can safely consider
  // wall boundaries.
  template <int dim, int n_vars>
  class BcThackerOscillations2d : public BcBase<dim, n_vars>
  {
  public:

    BcThackerOscillations2d(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcThackerOscillations2d() = default;

    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  };

  template <int dim, int n_vars>
  void BcThackerOscillations2d<dim, n_vars>::set_boundary_conditions()
  {
    this->set_wall_boundary(0);
  }
} // namespace ICBC

#endif //ICBC_THACKEROSCILLATIONS2D_HPP
