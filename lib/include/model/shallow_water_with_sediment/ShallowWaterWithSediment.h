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
#ifndef SHALLOWWATERWITHSEDIMENT_H
#define SHALLOWWATERWITHSEDIMENT_H

// The following files include the oceano libraries
#include <model/shallow_water/ShallowWater.h>
#include <model/shallow_water_with_tracer/ShallowWaterWithTracer.h>

/**
 * Namespace containing the model equations.
 */

namespace Model
{

  using namespace dealii;



  // @sect3{Implementation of the sediment equations}

  // In the following functions, we implement the various problem-specific
  // operators pertaining to the suspended sediment equations.
  //
  // We follow the coding style of the base class, with the inlining
  // of each pointwise operation.
  class ShallowWaterWithSediment : public ShallowWaterWithTracer
  {
  public:
    ShallowWaterWithSediment(IO::ParameterHandler &prm);
    ~ShallowWaterWithSediment() = default;

    template <int dim, int n_tra>
    inline DEAL_II_ALWAYS_INLINE //
      void
      set_vars_name();
  };


  // The class constructor simply constructs the base classes: the
  // tracer class and the shallow water class with the related
  // class members (advective and diffusive fluxes, etc ...)
  // The specificity of this class consists in the source term
  // that models the suspended sediment erosion and deposition.
  ShallowWaterWithSediment::ShallowWaterWithSediment(
    IO::ParameterHandler &prm)
    : ShallowWaterWithTracer(prm)
  {}

  template <int dim, int n_tra>
  inline DEAL_II_ALWAYS_INLINE //
    void
    ShallowWaterWithSediment::set_vars_name()
  {
    ShallowWater::set_vars_name <dim,n_tra>();
    for (unsigned int t = 0; t < n_tra; ++t)
      vars_name.push_back("sediment_"+std::to_string(t+1));
  }
} // namespace Model
#endif //SHALLOWWATERWITHSEDIMENT_H
