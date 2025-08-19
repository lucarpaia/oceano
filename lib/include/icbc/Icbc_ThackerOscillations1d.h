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
#ifndef ICBC_THACKEROSCILLATIONS1D_HPP
#define ICBC_THACKEROSCILLATIONS1D_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // Thacker in 1981 found an exact solutions of the Shallow Water equations
  // with moving boundaries, that is with wetting and drying. He found an analytic
  // periodic solutions (there is no damping) with Coriolis effect that here is
  // neglected. The test implemented is the one-dimensional simplification of the
  // Thacker solution with a planar surface, proposed in the SWASH test-suite of
  // Delestre,2016. The bathymetry is a parabola.

  using namespace dealii;
  
  // We define constant parameters that help in the definition of the initial
  // and boundary conditions. We use the same parameter of the SWASH test case:
  constexpr double h0      = 0.5;
  constexpr double a       = 1.0;
  constexpr double L       = 4.0;



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
  // We do not have any source term and no associated data values.
  template <int dim>  
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData(){};

   inline double thackerOscillations1d_bathymetry(const Point<dim> & p) const;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &/*prm*/)
    : Function<dim>(dim+3)
  {}



  template <int dim>
  inline double ProblemData<dim>::thackerOscillations1d_bathymetry(
    const Point<dim> & x) const
  {
    const double x_half = x[0]-0.5*L;
    const double inv_a = 1./a;
    return -h0 * (inv_a*inv_a * x_half*x_half - 1.);
  }

  template <int dim>
  double ProblemData<dim>::value(const Point<dim> & x,
                                 const unsigned int component) const
  {
    if (component == 0)
      return thackerOscillations1d_bathymetry(x);
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

    const double c = std::sqrt(2.*g*h0);
    const double inv_a = 1./a; 
    const double B = 0.5*c*inv_a;

    const double x1 = -0.5 * cos(c*inv_a*t) - a + 0.5*L;
    const double x2 = -0.5 * cos(c*inv_a*t) + a + 0.5*L;

    double zb;
    double h;
    double u;
    if (x[0] < x1)
      {
        const double x_half = x1-0.5*L;
        zb = h0 * (inv_a*inv_a * x_half*x_half - 1.);
        h = 0.;
        u = 0.;
      }
    else if (x[0] > x2)
      {
        const double x_half = x2-0.5*L;
        zb = h0 * (inv_a*inv_a * x_half*x_half - 1.);        
        h = 0.;
        u = 0.;
      }
    else
      {
        const double x_half = x[0]-0.5*L;
        zb = h0 * (inv_a*inv_a * x_half*x_half - 1.);
        const double tmp = inv_a*x_half + B/c*std::cos(c*inv_a*t);
        h = -h0 * (tmp*tmp - 1.);
        u = B * std::sin(c*inv_a*t);
      }    

    if (component == 0)
      return h + zb;
    else if (component == 1)
      return std::max(h, 0.) * u;
    else
      return 0.;
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
  // The wave does not interact with the boundaries so we can safely consider
  // wall boundaries.
  template <int dim, int n_vars>
  class Ic : public ExactSolution<dim, n_vars>
  {
  public:
    Ic(IO::ParameterHandler &prm)
      : ExactSolution<dim, n_vars>(0.,prm)
    {}
    ~Ic(){};
  };



  template <int dim, int n_vars>  
  class BcThackerOscillations1d : public BcBase<dim, n_vars>
  {
  public:
 
    BcThackerOscillations1d(IO::ParameterHandler &prm)
      : prm(prm)
    {}
    ~BcThackerOscillations1d(){};
         
    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  }; 

  template <int dim, int n_vars>
  void BcThackerOscillations1d<dim, n_vars>::set_boundary_conditions()
  {
    this->set_wall_boundary(0);
  }    
} // namespace ICBC

#endif //ICBC_THACKEROSCILLATIONS1D_HPP
