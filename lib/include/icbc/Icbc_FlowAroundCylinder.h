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
#ifndef ICBC_FLOWAROUNDCYLIDER_HPP
#define ICBC_FLOWAROUNDCYLIDER_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

 // The second test case start with a cylinder immersed in a
 // channel. Here, we impose a subsonic initial field at Mach number 
 // of $\mathrm{Ma}=0.307$ with a constant velocity in $x$ direction. 
 // At the top and bottom walls as well as at
 // the cylinder, we impose a no-penetration (i.e., tangential flow)
 // condition. This setup forces the flow to re-orient as compared to the initial
 // condition, which results in a big sound wave propagating away from the
 // cylinder. In upstream direction, the wave travels more slowly (as it
 // has to move against the oncoming gas), including a
 // discontinuity in density and pressure. In downstream direction, the transport
 // is faster as sound propagation and fluid flow go in the same direction, which smears
 // out the discontinuity somewhat. Once the sound wave hits the upper and lower
 // walls, the sound is reflected back.

  using namespace dealii;

  // The class `ExactSolution` defines analytical functions that can be useful
  // to define initial and boundary conditions.
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

  // For the channel test case, we simply select a density of 1, a velocity of
  // 0.4 in $x$ direction and zero in the other directions, and an energy that
  // corresponds to a speed of sound of 1.3 measured against the background
  // velocity field, computed from the relation $E = \frac{c^2}{\gamma (\gamma
  // -1)} + \frac 12 \rho \|u\|^2$.
  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & x,
                                           const unsigned int component) const
  {
    (void)x;
    if (component == 0)
      return 1.;
    else if (component == 1)
      return 0.4;
    else if (component == dim + 1)
      return 3.097857142857143;
    else
      return 0.;
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
  class BcFlowAroundCylinder : public BcBase<dim, n_vars>
  {
  public:
        
    BcFlowAroundCylinder(IO::ParameterHandler &/*prm*/){};
    ~BcFlowAroundCylinder(){};
         
    void set_boundary_conditions() override;
    
  }; 

  // Here, we have a larger variety of boundaries. The inflow
  // part at the left of the channel is given the inflow type, for which we
  // choose a constant inflow profile, whereas we set a subsonic outflow at
  // the right. For the boundary around the cylinder (boundary id equal to 2)
  // as well as the channel walls (boundary id equal to 3) we use the wall
  // boundary type, which is no-normal flow.
  template <int dim, int n_vars>
  void BcFlowAroundCylinder<dim, n_vars>::set_boundary_conditions()
  {
    this->set_inflow_boundary(
      0, std::make_unique<ExactSolution<dim, n_vars>>(0));
    this->set_supercritical_outflow_boundary(
      1, std::make_unique<ExactSolution<dim, n_vars>>(0));

    this->set_wall_boundary(2);
    this->set_wall_boundary(3);
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
  // In this case it is a constant gravity, of course in the vertical
  // direction.
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
                                 const unsigned int component) const
  {
    if (component == 1)
      return -0.2;
    else
      return 0.;
  }

} // namespace ICBC

#endif //ICBC_FLOWAROUNDCYLIDER_HPP
