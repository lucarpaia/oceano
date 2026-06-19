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
#include <model/shallow_water_with_sediment/physics/SettlingVelocity.h>
#include <model/shallow_water_with_sediment/physics/NearBedConcentration.h>

/**
 * Namespace containing the model equations.
 */

namespace Model
{

  using namespace dealii;



  // @sect3{Implementation of the suspended sediment equations}

  // In the following functions, we implement the various problem-specific
  // operators pertaining to the suspended sediment equations.
  // We follow the coding style of the base class, with the inlining
  // of each pointwise operation.
  // The specificity of this class consists in the source term
  // that overload the base class sources to model the erosion and
  // deposition processes.
  // For these source terms, new physical classes are used: one for
  // the definition of the settling velocity, one for the near-bed
  // concentration.
  class ShallowWaterWithSediment : public ShallowWaterWithTracer
  {
  public:
    ShallowWaterWithSediment(IO::ParameterHandler &prm);
    ~ShallowWaterWithSediment() = default;

    double g;
    double nu;
    double d50;
    double rho_water;
    double rho_sediment;
    double w_s;
    double shields_denominator;

    Physics::SettlingVelocityStokes settling_velocity;
    Physics::NearBedConcentrationZysermanFredsoe near_bed_concentration;

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

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      tracer_source(
        const Number                  height,
        const Tensor<1, dim, Number> &discharge,
        const Number                  concentration,
        const Number                  bathymetry,
        const Number                  drag_coefficient) const;
  };



  // The class constructor 1/ constructs the base classes: the
  // tracer class and the shallow water class with the related
  // class members (advective and diffusive fluxes, etc ...)
  // 2/ reads physical constant from the parameter file 3/ compute
  // some derived constants, notably the settling velocity and the
  // denominator for the Shields parameter. These factors are used
  // at each quadrature point and they involve (expensive) divisions.
  // We optimze a bit computing these divisions
  // once at constructor stage.
  ShallowWaterWithSediment::ShallowWaterWithSediment(
    IO::ParameterHandler &prm)
    : ShallowWaterWithTracer(prm)
    , settling_velocity()
    , near_bed_concentration()
  {
    prm.enter_subsection("Physical constants");
    g = prm.get_double("g");
    nu = prm.get_double("water_kinematic_viscosity");
    d50 = prm.get_double("sediment_diameter");
    rho_water = prm.get_double("water_density");
    rho_sediment = prm.get_double("sediment_density");
    prm.leave_subsection();
    w_s = settling_velocity.value(g, nu, d50, rho_water, rho_sediment);
    shields_denominator = 1./((rho_sediment/rho_water - 1.) * g * d50);
  }

  template <int dim, int n_tra>
  inline DEAL_II_ALWAYS_INLINE //
    void
    ShallowWaterWithSediment::set_vars_name()
  {
    ShallowWater::set_vars_name <dim,n_tra>();
    for (unsigned int t = 0; t < n_tra; ++t)
      vars_name.push_back("sediment_concentration_"+std::to_string(t+1));
  }

  // This implementation provides an erosion-only source for now:
  //
  // $$E = w_s max(c_b - c, 0)$$
  //
  // where $c_b$ is the near-bed concentration, c is the suspended
  // sediment concentration and $w_s$ is the settling velocity.
  // The near-bed concentration, a function of the Shield
  // parameter $\theta$, and the settling velocity are
  // computed in dedicated physics classes.
  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Number>
    ShallowWaterWithSediment::tracer_source(
      const Number                    height,
      const Tensor<1, dim, Number>   &discharge,
      const Tensor<1, n_tra, Number> &concentration,
      const Number                    bathymetry,
      const Number                    drag_coefficient) const
  {
    const Tensor<1, dim, Number> v =
      velocity<dim>(height, discharge, bathymetry);
    const Number h = depth(height, bathymetry);
    const Number bottom_fric =
      bottom_friction.source<dim, Number>(v, drag_coefficient, h).norm();

    const Number theta = bottom_fric / shields_denominator;
    const Number cb = near_bed_concentration.value<Number>(theta);

    Tensor<1, n_tra, Number> source;
    for (unsigned int t = 0; t < n_tra; ++t)
      source[t] =
        w_s * std::max(cb - concentration[t], Number(0.0));

    return source;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWaterWithSediment::tracer_source(
      const Number                    height,
      const Tensor<1, dim, Number>   &discharge,
      const Number                    concentration,
      const Number                    bathymetry,
      const Number                    drag_coefficient) const
  {
    const Tensor<1, dim, Number> v =
      velocity<dim>(height, discharge, bathymetry);
    const Number h = depth(height, bathymetry);
    const Number bottom_fric =
      bottom_friction.source<dim, Number>(v, drag_coefficient, h).norm();

    const Number theta = bottom_fric / shields_denominator;
    const Number cb = near_bed_concentration.value<Number>(theta);

    return w_s * std::max(cb - concentration, Number(0.0));
  }
} // namespace Model
#endif //SHALLOWWATERWITHSEDIMENT_H
