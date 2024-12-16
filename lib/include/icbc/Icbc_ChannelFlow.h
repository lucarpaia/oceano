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
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // This is a steady state solution of the one dimensional shallow water equations
  // with constant discharge $hu=q_0$, varying topography and friction. Given a
  // bathymetry profile $z_b(x)$ we get the corresponding water depth from the
  // integration of the following ODE:
  // \[
  // \partial_x h =
  //   \left( \frac{q_0^2}{gh^3(x)} -1 \right)^{-1}
  //   \left( -\partial_x z_b + \frac{F}{gh} \right)
  // \]
  // If $F \neq 0$ (we have friction at the bottom), the following solutions can
  // prove if the friction terms are coded in order to satisfy the steady states.
  // We consider a 100 m long channel, with a constant discharge as in
  // (Rosatti&Bonaventura,2011). For a subcritical flow we can
  // impose only one characteristic variable or a phyisical one. Based on physical
  // arguments we impose the discharge at the inflow and the water height at the
  // outflow.

  using namespace dealii;

  // We define global parameters that help in the definition of the initial
  // and boundary conditions. For this test we just need the discharge and
  // and the Manning coefficient:
  constexpr double b0      = 2.;
  constexpr double c0      = 10.;
  constexpr double d0      = 5.;
  constexpr double q0      = 5.;
  constexpr double n0      = 0.04;

  // @sect3{Equation data}

  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions. Apart for the template for the
  // dimension which is in common with the base `Function` class, we have added
  // the number of variables. As seen in the introduction the free-surface is
  // available in a semi-analytical form and must be read from a file. We have
  // thus modified the constructor with two classes: a data reader class and a
  // the data class itself. Thanks to them we can read and compute the exact
  // solution with a bilinear interpolation. We do not talk more about these two
  // classes and about the file format because they are discussed in detail in
  // `Icbc_Realistic` where the same classes are used to read the bathymetry.
  //
  // In the Oceano variables the exact solution must be given in the free-surface
  // and discharge variables. We check that the test runs in two-dimensions
  // which is consistent with the dimension of the file (otherwise it would raise
  // an error difficult to detect). The free-surface is computed by a bilinear
  // interpolation of the data read from file.
  template <int dim, int n_vars>  
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double          time,
                  IO::ParameterHandler &prm)
      : Function<dim>(n_vars, time)
      , data_reader(filename(prm))
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



  // The `Ic` and `Bc` classes define the initial/boundary condition for the
  // test-case. They are very similar in the templates and the constructor.
  // They both take as argument the parameter class and they stored it
  // internally. This means that we can read the Parameter file from
  // anywhere when we are implementing ic/bc and we can access constants or
  // filenames from which the initial/boundary data depends.
  // We start with a wet channel in equilibrium with a sloping topography
  // without any bump or hill, that is with $\partial_x h=0$ in the above equation.
  // The initial conditions are $hu(0,x) = q_0$ and a free-surface that in
  // pressure-gradient friction balance.
  // We return either the water depth or the momentum depending on which
  // component is requested. Two sanity checks have been added. One is to
  // control that the space dimension is two (you cannot run this test in
  // one dimension) and another one on the number of variables, that for
  // two-dimensional shallow water equation is three or more (if you have
  // tracers).
  //
  // If tracers are presents they are simply set to zero.
  // A subcritical inflow/outflow boundary condition is specified on the left and
  // right boundary of the domain. Top and bottom boundaries are walls.
  template <int dim, int n_vars>
  class Ic : public Function<dim>
  {
  public:
    Ic(IO::ParameterHandler &prm)
      : Function<dim>(n_vars, 0.)
    {
      prm.enter_subsection("Physical constants");
      g = prm.get_double("g");
      prm.leave_subsection();
    }
    ~Ic(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  private:
    double g;
  };

  template <int dim, int n_vars>
  double Ic<dim, n_vars>::value(const Point<dim>  &x,
                                const unsigned int component) const
  {
    Assert(dim == 2, ExcNotImplemented());
    Assert(n_vars < 4, ExcNotImplemented());

    if (component == 0)
      {
        const double inv_depth = std::exp( std::log(d0) * 10./3. );
        return - n0*n0 * q0*q0/inv_depth * x[0];
      }
    else if (component == 1)
        return q0;
    else
        return 0.;
  }



  template <int dim, int n_vars>  
  class BcChannelFlow : public BcBase<dim, n_vars>
  {
  public:
  
    BcChannelFlow(IO::ParameterHandler &prm);
    ~BcChannelFlow(){};

    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  };
  
  template <int dim, int n_vars>
  BcChannelFlow<dim, n_vars>::BcChannelFlow(IO::ParameterHandler &prm)
    : prm(prm)
  {}

  template <int dim, int n_vars>
  void BcChannelFlow<dim, n_vars>::set_boundary_conditions()
  {
    this->set_discharge_inflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    this->set_height_inflow_boundary(
      2, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
    this->set_wall_boundary(0);
  }         



  // We need a class to handle the problem data. Problem data are case dependent;
  // for this reason it appears inside the `ICBC` namespace. The data in general
  // depends on both time and space. Deal.II has a class `Function` which returns
  // function of space and time, thus we simply create a derived class. The size
  // of the data is fixed to `dim+3=5` scalar quantities. The first component is
  // the bathymetry. The second is the bottom friction coefficient. The third and
  // fourth components are the cartesian components of the wind velocity (in order,
  // eastward and northward). The fifth one is the Coriolis parameter. The test-dependent
  // `channelFlow_bathymetry()` contain the definition of the analytical bathyemtry
  // function. The call to `value()` returns all the external data necessary to
  // complete the computation.
  //
  // Finally the parameter handler class allows to read constants from the prm file.
  // The parameter handler class may seems redundant but it is not! Constants that appears
  // in you data may be easily recovered from the configuration file. More important file
  // names which contains the may be imported too.
  //
  // For this case we need to define the bathyemtry data values and the manning friction.
  template <int dim>  
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData(){};

    inline double channelFlow_bathymetry(const Point<dim> & p) const;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &/*prm*/)
    : Function<dim>(dim+3)
  {}



  template <int dim>
  inline double ProblemData<dim>::channelFlow_bathymetry(
    const Point<dim> & x) const
  {
    const double L = 100.;
    const double xc = x[0] - 0.5*L;

    double zb = d0 + 0.001*x[0];
    zb =  fabs(xc) < c0 ? zb - b0 * std::cos(M_PI*xc/(2.0*c0))*std::cos(M_PI*xc/(2.0*c0)) : zb;
    return zb;
  }

  template <int dim>
  double ProblemData<dim>::value(const Point<dim>  &x,
                                 const unsigned int component) const
  {
    if (component == 0)
      return channelFlow_bathymetry(x);
    else if (component == 1)
      return n0;
    else
      return 0.0;
  }
} // namespace ICBC

#endif //ICBC_CHANNELFLOW_HPP
