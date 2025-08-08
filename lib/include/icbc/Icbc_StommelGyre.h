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
#ifndef ICBC_STOMMELGYRE_HPP
#define ICBC_STOMMELGYRE_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // A large scale test is the Stommel gyre, see 
  // "The westward intensification of wind-driven ocean currents, H. Stommel 1948"
  // that mimick the dynamics of wind-driven ocean circulation. This test is made of a wind-driven circulation 
  // in a homogeneous flat and rectangular ocean under the influence of surface wind stress, linear bottom friction,
  // and variable Coriolis force with the so $\beta-$plane approximation. A sinusoidal wind stress typical of the 
  // northern ocean induces a clockwise circulation, while the linear dissipation term balances the wind stress forcing.
  // An intense current parallel to the western boundary is observed, that is observed also in the real ocean. 
  // Stommel has computed an exact solution for the linearized shallow water equations to which we can compare, 
  // even if our model solves the non-linear version.
  
  using namespace dealii;
  
  // We define global parameters that help in the definition of the initial
  // and boundary conditions. The Coriolis parameter 
  // $f=f_0+\beta y$ is defined using mid-latitude values for the northern 
  // emisphere. For the parameters name we try to follow the original reference:
  constexpr double f0     = 1e-4;
  constexpr double beta   = 2e-11;
  // The parameters of the forcing are the amplitude of the sinusoidal wind $F$
  // and the bottom drag $R$: 
  constexpr double F      = 1e-7 * 1000.;
  constexpr double R      = 5e-7;
  // The bassin geometrical parameters follows. The width $b$, the length $\lambda$ and
  // the bassin depth $D$:  
  constexpr double b      = 1e+6;  
  constexpr double lambda = 1e+6; 
  constexpr double D      = 1000.; 
    
  // @sect3{Equation data}
  //
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
  // For the stommel gyre we define the zonal wind field, a friction coefficient and
  // the coriolis parameter. Note also how we define the $f$ and $\beta$ parameters. They are
  // members of the `ProblemData` class and they are initialized in the constructor
  // thanks to the parameter handler class.
  template <int dim>
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData(){};

    double f0;
    double beta;

    inline double stommelGyre_wind(const Point<dim> & p,
                                   const unsigned int component = 0) const;

    inline double stommelGyre_coriolis(const Point<dim> & p) const;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &prm)
    : Function<dim>(dim+3)
  {
    prm.enter_subsection("Physical constants");
    f0 = prm.get_double("coriolis_f0");
    beta = prm.get_double("coriolis_beta");
    prm.leave_subsection();
  }



  template <int dim>
  inline double ProblemData<dim>::stommelGyre_wind(
    const Point<dim> & x,
    const unsigned int component) const
  {
    if (component == 0)
      return F * std::sin( M_PI * ( x[1] - 0.5*b ) / b );
    else
      return 0.;
  }

  template <int dim>
  inline double ProblemData<dim>::stommelGyre_coriolis(
    const Point<dim> & x) const
  {
    return f0 + beta * x[1];
  }

  template <int dim>
  double ProblemData<dim>::value(const Point<dim> & x,
                                 const unsigned int component) const
  {
    if (component == 1)
      return R;
    else if (component == 2)
      return stommelGyre_wind(x, 0);
    else if (component == 3)
      return stommelGyre_wind(x, 1);
    else if (component == 4)
      return stommelGyre_coriolis(x);
    else
      return 0.;
  }



  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions. For the gyre test it only defines
  // the analytical solution seen above. Apart for the template for the
  // dimension which is in common with the base `Function` class, we have added
  // the number of variables. We return either the water depth or the momentum
  // depending on which component is requested.
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

  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & x,
                                           const unsigned int component) const
  {
    Assert(dim == 2, ExcNotImplemented());

    double pi_invb = M_PI/b;
    double b_invpi = 1./pi_invb;
    double b_invpi2 = b_invpi * b_invpi;
    double gamma = F * pi_invb / R;
    double gamma_invg = gamma/g;	
    double F_invgD = F / (g*D);
    double cosyOverb = std::cos((x[1]-0.5*b) * pi_invb);
    double sinyOverb = std::sin((x[1]-0.5*b) * pi_invb);
    double alpha = D/R*beta;
    double alpha_half = 0.5 * alpha;
    double det = std::sqrt( 0.25*alpha*alpha + pi_invb*pi_invb );
    double A = - alpha_half - det;
    double B = - alpha_half + det;
    double P = (1. - std::exp(B*lambda))/(std::exp(A*lambda)-std::exp(B*lambda));
    double Q = 1. - P;
    double PexpAx = P * std::exp(A * x[0]);
    double QexpBx = Q * std::exp(B * x[0]);
    double expu = PexpAx + QexpBx - 1.;
    double expv =  A * PexpAx + B * QexpBx;
                
    const double depth = D
      - F_invgD * (1./A * PexpAx + 1./B * QexpBx)
      - b_invpi2 * F_invgD * expv * (cosyOverb - 1.)
      - ( f0 * gamma_invg * b_invpi2 * sinyOverb 
      - beta * gamma_invg * b_invpi2*b_invpi * (cosyOverb - 1.) ) 
      * expv;
    const double u = 
      gamma * b_invpi * cosyOverb * expu;
    const double v = 
      - gamma * b_invpi2 * sinyOverb * expv;
      
    if (component == 0)
      return depth;
    else if (component == 1)
      return depth*u;
    else
      return depth*v;
  }



  // The `Ic` and `Bc` classes define the initial/boundary condition for the
  // test-case. They are very similar in the templates and the constructor.
  // They both take as argument the parameter class and they stored it
  // internally. This means that we can read the Parameter file from
  // anywhere when we are implementing ic/bc and we can access constants or
  // filenames from which the initial/boundary data depends.
  // This is realized here thanks to a derived class of the deal.II `Function` class
  // that define many type of time and space functions. The initial condition class
  // overload the the constructor of the base class providing automatically
  // a zero time. Note that, apart for the template for the dimension which is in common with
  // the base `Function` class, we have added the number of variables to construct the base
  // class with the correct number of dimension and do some sanity checks.
  //
  // We return either the water depth or the momentum
  // depending on which component is requested. Two sanity checks have been added. One is to
  // control that the space dimension is two (you cannot run this test in one dimension) and
  // another one on the number of variables, that for two-dimensional shallow water equation 
  // is three. The initial solution for the gyre test is an ocean at rest.
  // Wall boundary conditions are assumed on the four sides of the rectangular ocean.
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
    Assert(n_vars <= 4, ExcNotImplemented());

    if (component == 0)
      return D;
    else if (component == 1 || component == 2)
      return 0.;
    else
      return 30. - 0.000025 * x[1];
  }



  template <int dim, int n_vars>  
  class BcStommelGyre : public BcBase<dim, n_vars>
  {
  public:
  
    BcStommelGyre(IO::ParameterHandler &/*prm*/){};
    ~BcStommelGyre(){};
         
    void set_boundary_conditions() override;

  };

  template <int dim, int n_vars>
  void BcStommelGyre<dim, n_vars>::set_boundary_conditions()
  {
    this->set_wall_boundary(0);
  }
} // namespace ICBC

#endif //ICBC_STOMMELGYRE_HPP
