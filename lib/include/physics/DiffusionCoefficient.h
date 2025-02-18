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
#ifndef DIFFUSIONCOEFFICIENT_HPP
#define DIFFUSIONCOEFFICIENT_HPP

/**
 * Namespace containing the so-called phyisics of the governing equations.
 * They are the different parametrizations.
 */

namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the horizontal diffusion coefficient}

  // In the following classes, we implement the horizontal diffusivity.
  // The horizontal diffusivity can be the molecular one but, at the scale
  // at which coastal models operates, it will be more often an eddy diffusivity.
  // In coastal flows, transport brings closer water masses with different
  // properties, continuously building up strong gradients. These can be
  // accurately simulated with TVD schemes or with an horizontal "eddy" diffusion.
  // The next class and its derived classes, basically compute the eddy diffusivity. 
  class DiffusionCoefficientBase
  {
  public:
    DiffusionCoefficientBase(IO::ParameterHandler &prm);
    ~DiffusionCoefficientBase(){};

    double nu;

    // The next function is the one that actually computes the diffusion coefficient.
    // It is overloaded by the same function defined in the derived classes.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value() const;
  };
  
  // Not surprisingly the constructor of the base class takes as arguments 
  // only the parameters handler class in order to read the physical constants
  // from the prm file.
  DiffusionCoefficientBase::DiffusionCoefficientBase(
    IO::ParameterHandler &prm)
  {
    prm.enter_subsection("Physical constants");
    nu = prm.get_double("horizontal_diffusivity");
    prm.leave_subsection();
  }



  // The first formulation of the diffusion coefficient is a constant
  // one, the value being read from the configuration file.
  class DiffusionCoefficientConstant : public DiffusionCoefficientBase
  {
  public:
    DiffusionCoefficientConstant(IO::ParameterHandler &prm);
    ~DiffusionCoefficientConstant(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      value() const;
  };

  DiffusionCoefficientConstant::DiffusionCoefficientConstant(
    IO::ParameterHandler &param)
    : DiffusionCoefficientBase(param)
  {}
  
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    DiffusionCoefficientConstant::value() const
  {
    return nu;
  }
} // namespace Physics
#endif //DIFFUSIONCOEFFICIENT_HPP
