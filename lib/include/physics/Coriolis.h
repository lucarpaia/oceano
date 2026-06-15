/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2026 by CNR-ISMAR
 *
 * This code, as the deal.II library is free software; you can use it,
 * redistribute it, and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later
 * version. The full text of the license can be found in the file
 * LICENSE.md at the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Luca Arpaia, 2023
 *         Giuseppe Orlando, 2026
 */
#ifndef CORIOLIS_H
#define CORIOLIS_H

/**
 * Namespace containing the so-called physics of the governing equations.
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



    // The next function is the one that actually computes the coriolis parameter.
    // It is overloaded by the same function defined in the derived classes.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Tensor<1, dim, Number> &discharge,
             const Number                  coriolis_parameter) const;
  };

  // Not surprisingly the constructor of the base class takes as arguments 
  // only the parameters handler class in order to read the physical constants
  // from the prm file.
  CoriolisBase::CoriolisBase()
  {}
  
  
  // The Coriolis term reads:
  // \begin{equation*}
  //   F_x = + f hv
  // \end{equation*}
  // \begin{equation*}
  //   F_y = -f hu
  // \end{equation*}
  // where $f$ is the Coriolis parameter.
  class CoriolisBeta : public CoriolisBase
  {
  public:
    CoriolisBeta();
    ~CoriolisBeta(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Tensor<1, dim, Number> &discharge,
             const Number                  coriolis_parameter) const;
  };

  CoriolisBeta::CoriolisBeta()
    : CoriolisBase()
  {}

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    CoriolisBeta::source(const Tensor<1, dim, Number> &discharge,
                         const Number                  coriolis_parameter) const
  {
    Tensor<1, dim, Number> source;
    source[0] = coriolis_parameter * discharge[1];
    source[1] = -coriolis_parameter * discharge[0];

    return source;
  }

} // namespace Physics
#endif //CORIOLIS_H
