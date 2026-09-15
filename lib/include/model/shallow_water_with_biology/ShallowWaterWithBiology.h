/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2026 by CNR-ISMAR
 *
 * This code, as the deal.II library is free software; you can use it,
 * redistribute it, and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later
 * version.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Luca Arpaia, 2026
 */

#ifndef SHALLOWWATERWITHBIOLOGY_H
#define SHALLOWWATERWITHBIOLOGY_H

// The following files include the oceano libraries
#include <model/shallow_water_with_tracer/ShallowWaterWithTracer.h>
#include <model/shallow_water_with_biology/physics/GrowthDeathRateRosenzweigMacArthur.h>

/**
 * Namespace containing the model equations.
 */
namespace Model
{

  using namespace dealii;



  // @sect3{Implementation of a simple biological model}

  // In the following functions, we implement the various problem-specific
  // operators pertaining to the biological tracer equations.
  // We follow the coding style of the base class, with the inlining
  // of each pointwise operation.
  // The specificity of this class consists in the source term
  // that overloads the base class sources to model the prey-predator
  // interaction through the Rosenzweig-MacArthur model.
  class ShallowWaterWithBiology : public ShallowWaterWithTracer
  {
  public:
    ShallowWaterWithBiology(IO::ParameterHandler &prm);
    ~ShallowWaterWithBiology() = default;

    Physics::GrowthDeathRateRosenzweigMacArthur growth_death_rate;

    template <int dim, int n_tra>
    inline DEAL_II_ALWAYS_INLINE //
      void
      set_vars_name();

    template <int dim, int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Number>
      tracer_source(
        const Number                    height,
        const Tensor<1, dim, Number>   &discharge,
        const Tensor<1, n_tra, Number> &concentration,
        const Number                    bathymetry,
        const Number                    drag_coefficient) const;
  };



  // The class constructor constructs the base tracer class and the
  // growth-death rate parametrization.
  ShallowWaterWithBiology::ShallowWaterWithBiology(
    IO::ParameterHandler &prm)
    : ShallowWaterWithTracer(prm)
    , growth_death_rate(prm)
  {}



  template <int dim, int n_tra>
  inline DEAL_II_ALWAYS_INLINE //
    void
    ShallowWaterWithBiology::set_vars_name()
  {
    ShallowWater::set_vars_name <dim,n_tra>();
    vars_name.push_back("prey_concentration");
    vars_name.push_back("predator_concentration");
  }



  // The biological reaction term is evaluated in the dedicated
  // physics class.
  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Number>
    ShallowWaterWithBiology::tracer_source(
      const Number                    height,
      const Tensor<1, dim, Number>   &/*discharge*/,
      const Tensor<1, n_tra, Number> &concentration,
      const Number                    bathymetry,
      const Number                    /*drag_coefficient*/) const
  {
    const Number h = depth(height, bathymetry);

    return h * growth_death_rate.value<n_tra, Number>(concentration);
  }
} // namespace Model
#endif // SHALLOWWATERWITHBIOLOGY_H
