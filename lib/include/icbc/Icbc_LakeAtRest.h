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
#undef  ICBC_LAKEATREST_WATERATREST

  using namespace dealii;
  
  // We define global parameters that help in the definition of the initial
  // and boundary conditions. The parameters that are needed for this test are
  // the bassin depth far from the hill:
  constexpr double h0      = 1.0;
  // the amplitude of the perturbation:
  constexpr double a0      = 0.01;

  // @sect3{Equation data}

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
      return 0.;
    else if (component == 1)
      return 0.;
    else
      return 0.;
  }



  // @sect3{Equation data}
  //
  // The `Ic` and `Bc` classes define the initial/boundary condition for the
  // test-case. They are very similar in the templates and the constructor.
  // They both take as argument the parameter class and they stored it
  // internally. This means that we can read the Parameter file from
  // anywhere when we are implementing ic/bc and we can access constants or
  // filenames from which the initial/boundary data depends.
  // The initial condition is realized thanks to a derived class of the
  // deal.II `Function` class that define many type of time and space functions.
  // The initial condition class overloads the constructor of the base class
  // providing automatically a zero time. Note that, apart for the template for
  // the dimension which is in common with the base `Function` class, we have
  // added the number of variables to construct the base class with the correct
  // number of dimension and do some sanity checks. 
  // We return either the water depth or the momentum depending on which component
  // is requested. Two sanity checks have been added. One is to control that the
  // space dimension is two (you cannot run this test in one dimension) and
  // another one on the number of variables, that for two-dimensional shallow
  // water equation is three.
  //
  // An absorbing outflow boundary condition is specified on the left and
  // right boundary of the domain. In this way we let the wave smoothly go out from the
  // the domain. Top and bottom boundaries are walls.
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
#ifndef ICBC_LAKEATREST_WATERATREST
        return a0;
#else
        return 0.;
#endif
      else
        return 0.;
    else
      return 0.;
  }



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
    this->set_absorbing_outflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
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
        return h0 - 0.6 * std::exp(pot);
      }
    else
#endif
      {
        const double pot = -5. * x0 * x0 - 50. * x1 * x1;
        return h0 - 0.8 * std::exp(pot);
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
    
} // namespace ICBC

#endif //ICBC_LAKEATREST_HPP
