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
#ifndef CORIOLIS_HPP
#define CORIOLIS_HPP

/**
 * Namespace containing the so-called phyisics of the governing equations.
 * They are the different parametrizations.
 */

namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the Coriolis force}

  // In the following classes, we implement the force
  // related to the Earth's rotation. We use the $\beta-$plane approximation.
  class CoriolisBase
  {
  public:
    CoriolisBase();
    ~CoriolisBase(){};



    // The next function is the one that actually computes the bottom friction.
    // It is overloaded by the same function defined in the derived classes.
    template <int dim, int n_vars, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Tensor<1, n_vars, Number> &conserved_variables,
             const Number                     coriolis_parameter) const;
  };
  
  // Not surprisingly the constructor of the base class takes as arguments 
  // only the parameters handler class in order to read the physical constants
  // from the prm file.
  CoriolisBase::CoriolisBase()
  {}
  
  
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
  class CoriolisBeta : public CoriolisBase
  {
  public:
    CoriolisBeta();
    ~CoriolisBeta(){};

    template <int dim, int n_vars, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Tensor<1, n_vars, Number> &conserved_variables,
             const Number                     coriolis_parameter) const;
  };

  CoriolisBeta::CoriolisBeta()
    : CoriolisBase()
  {}
  
  template <int dim, int n_vars, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    CoriolisBeta::source(const Tensor<1, n_vars, Number> &conserved_variables,
                         const Number                     coriolis_parameter) const
  {
    Tensor<1, dim, Number> source;
    source[0] = coriolis_parameter * conserved_variables[2];
    source[1] = -coriolis_parameter * conserved_variables[1];

    return source;
  }

} // namespace Physics
#endif //CORIOLIS_HPP
