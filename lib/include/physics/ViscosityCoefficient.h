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
 * Author: Luca Arpaia,        2025
 */
#ifndef VISCOSITYCOEFFICIENT_HPP
#define VISCOSITYCOEFFICIENT_HPP

/**
 * Namespace containing the so-called phyisics of the governing equations.
 * They are the different parametrizations.
 */

namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the horizontal viscosity coefficient}

  // In the following classes, we implement the horizontal viscosity.
  // The horizontal viscosity can be the molecular one but, at the scale
  // at which coastal models operates, it will be more often an eddy viscosity.
  // In coastal flows, strong shear currents maybe under-resolved. This fact,
  // toghether with the low diffusion offered by high order finite elements may lead
  // to spurious modes at the grid scale. These modes can be cured either with
  // standard TVD schemes or with an horizontal "eddy" viscosity. At state of 
  // the art, a rigorous theory to model eddy viscosity in coastal flow is missing.
  // Coastal models use either constant viscosity or simple turbulence model based
  // on the mixing length.
  // The next class and its derived classes, basically compute the eddy viscosity. 
  class ViscosityCoefficientBase
  {
  public:
    ViscosityCoefficientBase(IO::ParameterHandler &prm);
    ~ViscosityCoefficientBase(){};

    double nu;

    // The next function is the one that actually computes the viscosity coefficient.
    // It is overloaded by the same function defined in the derived classes.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value() const;
  };
  
  // Not surprisingly the constructor of the base class takes as arguments 
  // only the parameters handler class in order to read the physical constants
  // from the prm file.
  ViscosityCoefficientBase::ViscosityCoefficientBase(
    IO::ParameterHandler &prm)
  {
    prm.enter_subsection("Physical constants");
    nu = prm.get_double("horizontal_viscosity");
    prm.leave_subsection();
  }



  // The first formulation of the viscosity coefficient is a constant
  // one, the value being read from the configuration file.
  class ViscosityCoefficientConstant : public ViscosityCoefficientBase
  {
  public:
    ViscosityCoefficientConstant(IO::ParameterHandler &prm);
    ~ViscosityCoefficientConstant(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value() const;
  };

  ViscosityCoefficientConstant::ViscosityCoefficientConstant(
    IO::ParameterHandler &param)
    : ViscosityCoefficientBase(param)
  {}
  
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ViscosityCoefficientConstant::value() const
  {
    return nu;
  }
} // namespace Physics
#endif //VISCOSITYCOEFFICIENT_HPP
