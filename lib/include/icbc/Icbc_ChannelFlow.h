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
#ifndef ICBC_CHANNELFLOW_HPP
#define ICBC_CHANNELFLOW_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
#include <io/TxtDataReader.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // This is a steady state solution of the one dimensional shallow water equations
  // with constant discharge $hu=q_0$, varying topography and friction. Given a
  // water depth profile $h(x)$ we get the corresponding bathymetry from the
  // integration of the following ODE:
  // \[
  // \partial_x z_b = \left( 1 -\frac{q_0^2}{gh^3(x)} \right)\partial_x h + \frac{F}{gh}
  // \]
  // If $F \neq 0$ (we have friction at the bottom), the following solutions can prove
  // if the friction terms are coded in order to satisfy the steady states.
  // We consider a 1000 m long channel, with a constant discharge $q_0 = 2.5 m^2 /s$.
  // The flow is supercritical both at inflow and at outflow, which simplifies the
  // boundary conditions. For a general topography $z(x)$ we need to solve the ODE
  // numerically on a very fine grid, much finer than the resolution of the deal.ii model.
  
  using namespace dealii;
  
  // We define global parameters that help in the definition of the initial
  // and boundary conditions. For this test we do not need a lot of parameters because
  // we read the external data from an external file. In fact we just need the gravity:
  constexpr double g       = 9.81;
  // and the discharge:
  constexpr double q0       = 2.5;


  // @sect3{Equation data}

  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions. Apart for the template for the
  // dimension which is in common with the base `Function` class, we have added
  // the number of variables. As seen in the introduction the free-surface is
  // available in a semi-analytical form and must be read from a file. We have
  // thus modified the constructor with two classes: a data reader class and a
  // the data class itself. Thanks to them we can read and compute the exact
  // solution with a bilinear interpolation. We do not talk more about these two
  // classes because they are discussed in detail when for the bathymetry.
  template <int dim, int n_vars>  
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double time)
      : Function<dim>(n_vars, time)
      , data_reader("exact_free_surface.txt.gz") //filename(prm)) //for now prm is not available
      , data(
          data_reader.endpoints,
          data_reader.n_intervals,
          Table<dim, double>(data_reader.n_intervals.front()+1,
                             data_reader.n_intervals.back()+1,
                             data_reader.get_data(data_reader.filename).begin()))
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    std::string filename(IO::ParameterHandler &prm) const;
    IO::TxtDataReader<dim> data_reader;
    const Functions::InterpolatedUniformGridData<dim> data;
  };  

  // We code the exact solution. In the Oceano variables the exact solution
  // must be given in the free-surface and discharge variables. We
  // check that the test runs in two-dimensions which is consistent with the
  // dimension of the file (otherwise it would raise an error difficult to
  // detect). The free-surface is computed by a bilinear interpolation of the
  // data read from file.
  template <int dim, int n_vars>
  std::string ExactSolution<dim, n_vars>::filename(IO::ParameterHandler &prm) const
  {
    prm.enter_subsection("Input data files");
    std::string filename = prm.get("Exact_solution_filename");
    prm.leave_subsection();

    return filename;
  }

  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & x,
                                           const unsigned int component) const
  {
    Assert(dim == 2, ExcNotImplemented());
    if (component == 0)
      return data.value(x);
    else if (component == 1)
      return q0;
    else
      return 0.;
  }



  // The `Ic` class define the initial condition for the test-case.
  // The initial conditions are a $\zeta(0,x) = 0\,m$ and 
  // $hu(0,x) = q_0\, m^2 /s$. 
  template <int dim, int n_vars>
  class Ic : public Function<dim>
  {
  public:
    Ic()
      : Function<dim>(n_vars, 0.)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  // We return either the water depth or the momentum
  // depending on which component is requested. Two sanity checks have been added. One is to
  // control that the space dimension is two (you cannot run this test in one dimension) and
  // another one on the number of variables, that for two-dimensional shallow water equation
  // is three or more (if you have tracers). If tracers are presents they are simply set to zero.
  template <int dim, int n_vars>
  double Ic<dim, n_vars>::value(const Point<dim>  &/*x*/,
                                const unsigned int component) const
  {
    Assert(dim == 2, ExcNotImplemented());
    Assert(n_vars < 4, ExcNotImplemented());

    if (component == 0)
      return 0.;
    else if (component == 1)
      return q0;
    else
      return 0.;
  }



  // The `Bc` class define the boundary conditions for the test-case.
  template <int dim, int n_vars>  
  class BcChannelFlow : public BcBase<dim, n_vars>
  {
  public:
  
    BcChannelFlow(){};
    ~BcChannelFlow(){};
         
    void set_boundary_conditions() override;

  };
  
  // A supercritical inflow/outflow boundary condition is specified on the left and
  // right boundary of the domain. Top and bottom boundaries are walls.
  template <int dim, int n_vars>
  void BcChannelFlow<dim, n_vars>::set_boundary_conditions()
  {
    this->set_inflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0));
    this->set_subcritical_outflow_boundary(
      2, std::make_unique<ExactSolution<dim, n_vars>>(0));
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
  // The parameter handler class allows to read constants from the prm file.
  // The parameter handler class may seems redundant but it is not! Constants that appears
  // in you data may be easily recovered from the configuration file. More important file
  // names which contains the may be imported too.
  //
  // For this case we need the bathymetry is not available in an analytical expression but
  // need to be interpolated from a reference field given on a fine mesh. For structured
  // quadrilateral meshes deal.ii has a special `Function` class that instead of the standard
  // analytical one it computes the values by (bi-, tri-)linear interpolation from a set of
  // point data that are arranged on a uniformly spaced tensor product mesh. It is called
  // `InterpolatedUniformGridData`. Considering the two-dimensional case, let there be points
  // $x_0,\,...,\,x_{K−1}$ that result from a uniform subdivision of the interval $\[x_0,x_{K−1}\]$
  // into $K−1$ sub-intervals of size $\Delta x=\frac{x_{K−1}−x_0}/{K−1}$, and similarly
  // $y_0,\,...,\,y_{L−1}$. Also consider bathymetry data $z_{kl}$ defined at point
  // $\left(x_k,y_l\right)^T$, then evaluating the function at a point $x=(x,y,z)$ will find
  // the box so that $x_k\le x \le x_{k+1},\, y_l \le y \le y_{l+1}$ and do a bilinear
  // interpolation of the data on this cell. Let us talk about the constructor of this class.
  // It takes as argument the interval_endpoints, the number of subintervals in each coordinate
  // direction and a `dim`-dimensional table of data at each of the mesh points defined by the
  // coordinate arrays above.
  //
  // We have also used an auxiliary class that reads the file. Thanks to this class we open,
  // read and store the header lines, then read the bathymetry values and close the file.
  template <int dim>  
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    std::string bathymetry_filename(IO::ParameterHandler &prm) const;
    IO::TxtDataReader<dim> bathymetry_data_reader;
    const Functions::InterpolatedUniformGridData<dim> bathymetry_data;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &prm)
    : Function<dim>(dim+3)
    , bathymetry_data_reader(bathymetry_filename(prm))
    , bathymetry_data(
        bathymetry_data_reader.endpoints,
        bathymetry_data_reader.n_intervals,
        Table<dim, double>(bathymetry_data_reader.n_intervals.front()+1,
                           bathymetry_data_reader.n_intervals.back()+1,
                           bathymetry_data_reader.get_data(bathymetry_data_reader.filename).begin()))
  {}



  template <int dim>
  std::string ProblemData<dim>::bathymetry_filename(IO::ParameterHandler &prm) const
  {
    prm.enter_subsection("Input data files");
    std::string filename = prm.get("Bathymetry_filename");
    prm.leave_subsection();

    return filename;
  }

  template <int dim>
  double ProblemData<dim>::value(const Point<dim>  &x,
                                 const unsigned int component) const
  {
    if (component == 0)
      return bathymetry_data.value(x);
    else
      return 0.0;
  }
} // namespace ICBC

#endif //ICBC_CHANNELFLOW_HPP
