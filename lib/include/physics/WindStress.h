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
 * Author: Luca Arpaia,        2023
 */
#ifndef WINDSTRESS_HPP
#define WINDSTRESS_HPP

/**
 * Namespace containing the so-called phyisics of the governing equations.
 * They are the different parametrizations.
 */

namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the bottom friction}

  // In the following classes, we implement the different formulations
  // of the wind stress term. As for other classes of the same namespace, the base
  // class is used to store the physical constants appearing into the different 
  // formulations. The quadratic drag coefficient needs a wind drag coefficient 
  // named as `cd` as well as the air and water density.  
  class WindStressBase
  {
  public:
    WindStressBase(IO::ParameterHandler &prm);
    ~WindStressBase(){};

    double cd;
    double air_density;
    double water_density;

    // The next function is the one that actually computes the bottom friction.
    // It is overloaded by the same function defined in the derived classes.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Number* wind_parameter) const;
  };
  
  // Not surprisingly the constructor of the base class takes as arguments 
  // only the parameters handler class in order to read the physical constants
  // from the prm file.
  WindStressBase::WindStressBase(
    IO::ParameterHandler &prm)
  {
    prm.enter_subsection("Physical constants");
    air_density = prm.get_double("air_density");
    water_density = prm.get_double("water_density");
    cd = prm.get_double("wind_drag");    
    prm.leave_subsection();
  }
  
  
#if defined PHYSICS_WINDSTRESSGENERAL
  // The first formulation is rather the general one. One should 
  // specify directly the wind stresses. Then this is simply assigned to the source:
  // \[
  // \boldsymbol{F} = \boldsymbol{\tau}
  // \]
  // where $\boldsymbol{\tau}$ is space and time-varying. The wind stress 
  // components must be given in the test case class. They are then passed
  // as constant pointer to the `source` function. This means
  // that we can easily move across the memory accessing to the next wind stress 
  // component and, at the same time, that we cannot modify the memory address.
  class WindStressGeneral : public WindStressBase
  {
  public:
    WindStressGeneral(IO::ParameterHandler &prm);
    ~WindStressGeneral(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Number* wind_stress) const;
  };

  WindStressGeneral::WindStressGeneral(
    IO::ParameterHandler &param)
    : WindStressBase(param)
  {}
  
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    WindStressGeneral::source(const Number* wind_stress) const
  {
    Tensor<1, dim, Number> source;
    for (unsigned int d = 0; d < dim; ++d)
      source[d] = wind_stress[d];

    return source;
  }



#elif defined PHYSICS_WINDSTRESSQUADRATIC
  // The second formulation is the classical quadratic wind stress formula. 
  // It takes the form:
  // \[
  // \boldsymbol{F} = \frac{\rho_{air}}{\rho_{0}}C_D||\boldsymbol{u}_{10}||\boldsymbol{u}_{10}
  // \]
  // with $C_D$ a constant wind drag coefficient, $\boldsymbol{u}_{10}$ the wind velocity 
  // computed ad $10\,m$ height, $\rho_{air}$ and \rho_{0} the air and water density.
  class WindStressQuadratic : public WindStressBase
  {
  public:
    WindStressQuadratic(IO::ParameterHandler &prm);
    ~WindStressQuadratic(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Number* wind_velocity) const;
  };

  WindStressQuadratic::WindStressQuadratic(
    IO::ParameterHandler &param)
    : WindStressBase(param)
  {}  
  
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    WindStressQuadratic::source(const Number* wind_velocity) const
  {
    Tensor<1, dim, Number> source;
    Number wind_norm = std::sqrt( wind_velocity[0]*wind_velocity[0] 
      + wind_velocity[1]*wind_velocity[1] );
    for (unsigned int d = 0; d < dim; ++d)
      source[d] = air_density/water_density * cd * wind_norm
        * wind_velocity[d];

    return source;
  }
#endif

} // namespace Physics
#endif //WINDSTRESS_HPP
