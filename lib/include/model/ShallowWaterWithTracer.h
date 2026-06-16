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
#ifndef SHALLOWWATERWITHTRACER_H
#define SHALLOWWATERWITHTRACER_H

// The following files include the oceano libraries
#include <model/ShallowWater.h>
#include <physics/DiffusionCoefficient.h>

/**
 * Namespace containing the model equations.
 */

namespace Model
{

  using namespace dealii;



  // @sect3{Implementation of the tracer equations}

  // In the following functions, we implement the various problem-specific
  // operators pertaining to the tracers equations. Each function acts on the
  // vector of prognostic variables $[\zeta, h\mathbf{u}, c]$ that we hold in
  // three different solution vectors.
  // From the solution we computes various derived quantities, i.e. advective
  // and diffusive fluxes.
  //
  // We follow the coding style of the base class, with the inlining
  // of each pointwise operation.
  class ShallowWaterWithTracer : public ShallowWater
  {
  public:
    ShallowWaterWithTracer(IO::ParameterHandler &prm);
    ~ShallowWaterWithTracer() = default;

#if defined PHYSICS_DIFFUSIONCOEFFICIENTCONSTANT
    Physics::DiffusionCoefficientConstant diffusion_coefficient;
#elif defined PHYSICS_DIFFUSIONCOEFFICIENTSMAGORINSKY
    Physics::DiffusionCoefficientSmagorinsky diffusion_coefficient;
#else
    Assert(false, ExcNotImplemented());
    return 0.;
#endif

    template <int dim, int n_tra>
    inline DEAL_II_ALWAYS_INLINE //
      void
      set_vars_name();

    // We define an advective flux which is used in the numerical flux
    // and an advective-diffusive flux used in the volume term.
    // For consistency with the continuity equation, we have paid
    // attention to use the same formulation of these fluxes as in
    // the continuity equation.
    template <int dim, int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Tensor<1, dim, Number>>
      tracer_advective_flux(
        const Tensor<1, dim, Number>   &discharge,
        const Tensor<1, n_tra, Number> &tracer) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      tracer_advective_flux(
        const Tensor<1, dim, Number>   &discharge,
        const Number                    tracer) const;

    template <int dim, int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Tensor<1, dim, Number>>
      tracer_advective_diffusive_flux(
        const Number                                    height,
        const Tensor<1, dim, Number>                   &discharge,
        const Tensor<1, n_tra, Number>                 &tracer,
        const Tensor<dim, dim, Number>                 &gradient_velocity,
        const Tensor<1, n_tra, Tensor<1, dim, Number>> &gradient_tracer,
        const Number                                    bathymetry,
        const Number                                    area) const;

    template <int dim, int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Tensor<1, dim, Number>>
      tracer_advective_diffusive_flux(
        const Number                    height,
        const Tensor<1, dim, Number>   &discharge,
        const Tensor<1, n_tra, Number> &tracer,
        const Tensor<dim, dim, Number> &gradient_velocity,
        const Tensor<2, dim, Number>   &gradient_tracer,
        const Number                    bathymetry,
        const Number                    area) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      tracer_advective_diffusive_flux(
        const Number                    height,
        const Tensor<1, dim, Number>   &discharge,
        const Number                    tracer,
        const Tensor<dim, dim, Number> &gradient_velocity,
        const Tensor<1, dim, Number>   &gradient_tracer,
        const Number                    bathymetry,
        const Number                    area) const;
  };



  // Similarly to the base class we have the class constructor
  // and similar class members that implement prognostic variables,
  // advective and diffusive fluxes, etc ...
  // The specificity here is that we duplicate/triplicate the flux
  // functions. The duplication is resolved with an overloading depending
  // the return data type and allows to handle the case of one single
  // tracer (with data type Number) and multiple tracers (with data type
  // Tensor<1, dim, Number>) without an `if` statement.
  ShallowWaterWithTracer::ShallowWaterWithTracer(
    IO::ParameterHandler &prm)
    : ShallowWater(prm)
    , diffusion_coefficient(prm)
  {}

  template <int dim, int n_tra>
  inline DEAL_II_ALWAYS_INLINE //
    void
    ShallowWaterWithTracer::set_vars_name()
  {
    ShallowWater::set_vars_name <dim,n_tra>();
    for (unsigned int t = 0; t < n_tra; ++t)
      vars_name.push_back("t_"+std::to_string(t+1));
  }

  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Tensor<1, dim, Number>>
    ShallowWaterWithTracer::tracer_advective_flux(
      const Tensor<1, dim, Number>   &discharge,
      const Tensor<1, n_tra, Number> &tracer) const
  {
    Tensor<1, n_tra, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < n_tra; ++e)
        flux[e][d] = tracer[e] * discharge[d];

    return flux;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWaterWithTracer::tracer_advective_flux(
      const Tensor<1, dim, Number>  &discharge,
      const Number                   tracer) const
  {
    return tracer * discharge;
  }

  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Tensor<1, dim, Number>>
    ShallowWaterWithTracer::tracer_advective_diffusive_flux(
      const Number                                    height,
      const Tensor<1, dim, Number>                   &discharge,
      const Tensor<1, n_tra, Number>                 &tracer,
      const Tensor<dim, dim, Number>                 &gradient_velocity,
      const Tensor<1, n_tra, Tensor<1, dim, Number>> &gradient_tracer,
      const Number                                    bathymetry,
      const Number                                    area) const
  {
    const Number h = depth(height, bathymetry);
    const Number nu = diffusion_coefficient.value<dim, Number>(gradient_velocity, area);
    const Tensor<1, n_tra, Tensor<1, dim, Number>> nuhdt = nu * h * gradient_tracer;

    Tensor<1, n_tra, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < n_tra; ++e)
        flux[e][d] = tracer[e] * discharge[d]
          - nuhdt[e][d];

    return flux;
  }

  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Tensor<1, dim, Number>>
    ShallowWaterWithTracer::tracer_advective_diffusive_flux(
      const Number                    height,
      const Tensor<1, dim, Number>   &discharge,
      const Tensor<1, n_tra, Number> &tracer,
      const Tensor<dim, dim, Number> &gradient_velocity,
      const Tensor<2, dim, Number>   &gradient_tracer,
      const Number                    bathymetry,
      const Number                    area) const
  {
    const Number h = depth(height, bathymetry);
    const Number nu = diffusion_coefficient.value<dim, Number>(gradient_velocity, area);
    const Tensor<2, dim, Number> nuhdt = nu * h * gradient_tracer;

    Tensor<1, n_tra, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < n_tra; ++e)
        flux[e][d] = tracer[e] * discharge[d]
          -  nuhdt[e][d];

    return flux;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWaterWithTracer::tracer_advective_diffusive_flux(
      const Number                    height,
      const Tensor<1, dim, Number>   &discharge,
      const Number                    tracer,
      const Tensor<dim, dim, Number> &gradient_velocity,
      const Tensor<1, dim, Number>   &gradient_tracer,
      const Number                    bathymetry,
      const Number                    area) const
  {
    const Number h = depth(height, bathymetry);

    return tracer * discharge
      - diffusion_coefficient.value<dim, Number>(gradient_velocity, area)
        * h * gradient_tracer;
  }
} // namespace Model
#endif //SHALLOWWATERWITHTRACER_H
