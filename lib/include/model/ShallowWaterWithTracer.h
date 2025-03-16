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
#ifndef SHALLOWWATERWITHTRACER_HPP
#define SHALLOWWATERWITHTRACER_HPP

// The following files include the oceano libraries
#include <model/ShallowWater.h>
#include <physics/DiffusionCoefficient.h>

/**
 * Namespace containing the model equations.
 */

namespace Model
{

  using namespace dealii;



  // @sect3{Implementation of point-wise operations of the tracer equations}

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
    ~ShallowWaterWithTracer(){};

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

    template <int dim, int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Tensor<1, dim, Number>>
      tracer_adv_flux(const Tensor<1, dim, Number>   &discharge,
                      const Tensor<1, n_tra, Number> &tracer) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      tracer_adv_flux(const Tensor<1, dim, Number>   &discharge,
                      const Number                    tracer) const;

    template <int dim, int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Tensor<1, dim, Number>>
      tracer_adv_diff_flux(
        const Tensor<1, dim, Number>                   &discharge,
        const Tensor<1, n_tra, Number>                 &tracer,
        const Tensor<dim, dim, Number>                 &gradient_velocity,
        const Tensor<1, n_tra, Tensor<1, dim, Number>> &gradient_tracer,
        const Number                                    area) const;

    template <int dim, int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Tensor<1, dim, Number>>
      tracer_adv_diff_flux(
        const Tensor<1, dim, Number>   &discharge,
        const Tensor<1, n_tra, Number> &tracer,
        const Tensor<dim, dim, Number> &gradient_velocity,
        const Tensor<2, dim, Number>   &gradient_tracer,
        const Number                    area) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      tracer_adv_diff_flux(
        const Tensor<1, dim, Number>   &discharge,
        const Number                    tracer,
        const Tensor<dim, dim, Number> &gradient_velocity,
        const Tensor<1, dim, Number>   &gradient_tracer,
        const Number                    area) const;
  };



  // For the model class we do not use an implementation file. This
  // is because of the fact the all the function called are templated
  // or inlined. Both templated and inlined functions are hard to be separated
  // between declaration and implementation. We keep them in the header file. 
  
  // The constructor of the model class takes as arguments the parameters handler
  // class in order to read the test-case/user dependent parameters. These
  // parameters are stored as class members. In this way they are defined/read 
  // from file in one place and then used whenever needed  with `model.param`, 
  // instead of being read/defined multiple times.
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
    vars_name.push_back("free_surface");
    for (unsigned int d = 0; d < dim; ++d)
      vars_name.push_back("hu");
    for (unsigned int t = 0; t < n_tra; ++t)
        vars_name.push_back("t_"+std::to_string(t+1));

    for (unsigned int d = 0; d < dim; ++d)
      postproc_vars_name.push_back("velocity");
    postproc_vars_name.push_back("depth");
  }

  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Tensor<1, dim, Number>>
    ShallowWaterWithTracer::tracer_adv_flux(
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
    ShallowWaterWithTracer::tracer_adv_flux(
      const Tensor<1, dim, Number>  &discharge,
      const Number                   tracer) const
  {
    return tracer * discharge;
  }

  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Tensor<1, dim, Number>>
    ShallowWaterWithTracer::tracer_adv_diff_flux(
      const Tensor<1, dim, Number>                   &discharge,
      const Tensor<1, n_tra, Number>                 &tracer,
      const Tensor<dim, dim, Number>                 &gradient_velocity,
      const Tensor<1, n_tra, Tensor<1, dim, Number>> &gradient_tracer,
      const Number                                    area) const
  {
    Number nu = diffusion_coefficient.value<dim, Number>(gradient_velocity, area);

    Tensor<1, n_tra, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < n_tra; ++e)
        flux[e][d] = tracer[e] * discharge[d]
          - nu * gradient_tracer[e][d];

    return flux;
  }

  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Tensor<1, dim, Number>>
    ShallowWaterWithTracer::tracer_adv_diff_flux(
      const Tensor<1, dim, Number>   &discharge,
      const Tensor<1, n_tra, Number> &tracer,
      const Tensor<dim, dim, Number> &gradient_velocity,
      const Tensor<2, dim, Number>   &gradient_tracer,
      const Number                    area) const
  {
    Number nu = diffusion_coefficient.value<dim, Number>(gradient_velocity, area);

    Tensor<1, n_tra, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < n_tra; ++e)
        flux[e][d] = tracer[e] * discharge[d]
          - nu * gradient_tracer[e][d];

    return flux;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWaterWithTracer::tracer_adv_diff_flux(
      const Tensor<1, dim, Number>   &discharge,
      const Number                    tracer,
      const Tensor<dim, dim, Number> &gradient_velocity,
      const Tensor<1, dim, Number>   &gradient_tracer,
      const Number                    area) const
  {
    return tracer * discharge
      - diffusion_coefficient.value<dim, Number>(gradient_velocity, area)
        * gradient_tracer;
  }
} // namespace Model
#endif //SHALLOWWATERWITHTRACER_HPP
