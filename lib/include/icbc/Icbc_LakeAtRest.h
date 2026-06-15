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
#ifndef ICBC_LAKEATREST_HPP
#define ICBC_LAKEATREST_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // This test case is a small perturbation af a two-dimensional lake-at-rest state.
  // The lake is 1m deep with an unven bathymetry. We consider two cases.
  // The first bathymetry is composed of a double-peaked smooth hill. We also
  // consider a non-smooth case to check if the scheme is capable of preserving the lake
  // at rest with such a discontinuous data. Having a scheme that can treat discontinous
  // bathymetry may be important for coastal applications where dams, dykes or
  // coastal barriers are present.
  // We add a 1cm small perturbation confined in a narrow band. Two small amplitude waves
  // generate: the left-propagating one goes out from the left boundary thanks to a
  // subsonic outflow boundary condition. The right-propagating wave travels over the
  // uneven bathymetry and transforms. This test allows to check the ability of the
  // a numerical scheme to catch smooth wave patterns and the lake at rest state in
  // the unperturbed regions. To use the discontinuous bathymetry use the following
  // cpp key:
#undef  ICBC_LAKEATREST_BATHYMETRYDISCONTINUOUS
  // You can also check the well-balanced property of the scheme with respect to the
  // "water-at-rest" state, without the perturbation. Define one of the following cpp key,
  // depending if you want to test the wet or the dry lake at rest test.
#undef  ICBC_LAKEATREST_WATERATRESTWET
#undef  ICBC_LAKEATREST_WATERATRESTDRY

  using namespace dealii;

  // We define global parameters that help in the definition of the initial
  // and boundary conditions. The parameters that are needed for this test are
  // the bassin depth far from the hill:
  constexpr double h0      = 1.0;
  // the amplitude of the perturbation:
#if defined ICBC_LAKEATREST_WATERATRESTWET || defined ICBC_LAKEATREST_WATERATRESTDRY
  constexpr double a0      = 0.0;
#else
  constexpr double a0      = 0.01;
#endif
  // and the height of the hill:
#if defined ICBC_LAKEATREST_WATERATRESTDRY
  constexpr double b0      = 1.3;
#else
#if defined ICBC_LAKEATREST_BATHYMETRYDISCONTINUOUS
  constexpr double b0      = 0.65;
#else
  constexpr double b0      = 0.80;
#endif
#endif
  // We specify a non trivial initial state
  // (a water height level different from zero). This is realized thanks to the
  // following offset:
  constexpr double z0      = 1.0;



  // @sect3{Equation data}
  //
  // For this case we need to define the bathyemtry data values.
  template <int dim>
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData(){};

    inline double lakeAtRest_bathymetry(const Point<dim> & p) const;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &/*prm*/)
    : Function<dim>(dim+3)
  {}



  template <int dim>
  inline double ProblemData<dim>::lakeAtRest_bathymetry(
    const Point<dim> & x) const
  {
    const double x0 = x[0] - 0.9;
    const double x1 = x[1] - 0.5;
#if defined ICBC_LAKEATREST_BATHYMETRYDISCONTINUOUS
    if ( x[0] >= 0.9 && x[0] <= 1.1 && x[1] >= 0.3 && x[1] <= 0.7 )
      {
        const double pot = std::sqrt( x0 * x0 + x1 * x1 );
        return -z0 +h0 -b0 * std::exp(pot);
      }
    else
#endif
      {
        const double pot = -5. * x0 * x0 - 50. * x1 * x1;
        return -z0 +h0 -b0 * std::exp(pot);
      }
  }

  template <int dim>
  double ProblemData<dim>::value(const Point<dim>  &x,
                                 const unsigned int component) const
  {
    if (component == 0)
      return lakeAtRest_bathymetry(x);
    else
      return 0.0;
  }



  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions. Apart for the template for the
  // dimension which is in common with the base `Function` class, we have added
  // the number of variables.
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

  // We code the exact solution as the lake at rest state. In the Oceano
  // variables (free-surface and momentum) it is the null vector. If you
  // set the amplitude to zero you can use the exact solution to check
  // that the method preserve the initial condition. We check that the test runs
  // in two-dimensions (you cannot run this test in one dimension).
  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & /*x*/,
                                           const unsigned int component) const
  {
    Assert(dim == 2, ExcNotImplemented());
    if (component == 0)
      return z0;
    else if (component == 1)
      return 0.;
    else
      return 0.;
  }



  // Two sanity checks have been added. One is to control that the
  // space dimension is two (you cannot run this test in one dimension) and
  // another one on the number of variables, that for two-dimensional shallow
  // water equation is three.
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

    if (component == 0)
      if ((0.05 <= x[0]) && (x[0] <= 0.15))
        return z0 + a0;
      else
        return z0;
    else
      return 0.;
  }



  // An absorbing outflow boundary condition is specified on the left and
  // right boundary of the domain. In this way we let the wave smoothly go out from the
  // the domain. Top and bottom boundaries are walls. For the water-at-rest test
  // we use a closed basin with four walls. This avoids spurious effects from the
  // boundaries.
  template <int dim, int n_vars>
  class BcLakeAtRest : public BcBase<dim, n_vars>
  {
  public:

    BcLakeAtRest(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcLakeAtRest(){};

    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  };

  template <int dim, int n_vars>
  void BcLakeAtRest<dim, n_vars>::set_boundary_conditions()
  {
    this->set_wall_boundary(0);
#if defined ICBC_LAKEATREST_WATERATRESTWET || defined ICBC_LAKEATREST_WATERATRESTDRY
    this->set_wall_boundary(1);
#else
    this->set_absorbing_outflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
#endif
  }
} // namespace ICBC

#endif //ICBC_LAKEATREST_HPP
