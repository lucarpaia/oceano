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
 * Author: Martin Kronbichler, 2020
 *         Luca Arpaia,        2023
 */
#ifndef LAXFRIEDRICHSMODIFIED_HPP
#define LAXFRIEDRICHSMODIFIED_HPP

// The following files include the oceano libraries 
#include <numerical_flux/NumericalFluxBase.h>

/**
 * Namespace containing the numerical flux.
 */

namespace NumericalFlux
{

  using namespace dealii;
  
  
  
  // For the model class we do not use an implementation file. This
  // is because of the fact the all the function called are templated
  // or inlined. Both templated and inlined functions are hard to be separated
  // between declaration and implementation. We keep them in the header file.
  //
  // I would have liked to template the numerical flux class with
  // <int dim, typename Number> which would have been cleaner. But I was not able
  // to compile the call to the function `numerical_flux_weak()` which take
  // as argument `Tensor<1, dim, Number>` while is receiving
  // `Tensor<1, dim, VectorizedArray<Number>>`. I don't know why, without
  // a template class, everything works. I leave this for future work.
  //
  // In this and the following functions, we use `z` for the mass variable,
  // (for us the water height or the free-surface position) and `q`
  // stands for the momentum (for us the discharge).
  // We use variable suffixes `_m` and
  // `_p` to indicate quantities derived from $\mathbf{w}^-$ and $\mathbf{w}^+$,
  // i.e., values "here" and "there" relative to the current cell when looking
  // at a neighbor cell. We also make use of the standard overloads
  // provided by deal.II's tensor classes for implementing the dot product.
  class LaxFriedrichsModified : public NumericalFluxBase
  {
  public:
    LaxFriedrichsModified(IO::ParameterHandler &param);
    ~LaxFriedrichsModified(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      numerical_massflux_weak(const Number                  z_m,
                              const Number                  z_p,
                              const Tensor<1, dim, Number> &q_m,
                              const Tensor<1, dim, Number> &q_p,
                              const Tensor<1, dim, Number> &normal,
                              const Number                  data_m,
                              const Number                  data_p) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      numerical_advflux_weak(const Number                  z_m,
                             const Number                  z_p,
                             const Tensor<1, dim, Number> &q_m,
                             const Tensor<1, dim, Number> &q_p,
                             const Tensor<1, dim, Number> &normal,
                             const Number                  data_m,
                             const Number                  data_p) const;

    template <int dim, int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Number>
      numerical_tracflux_weak(const Number                    z_m,
                              const Number                    z_p,
                              const Tensor<1, dim, Number>   &q_m,
                              const Tensor<1, dim, Number>   &q_p,
                              const Tensor<1, n_tra, Number> &t_m,
                              const Tensor<1, n_tra, Number> &t_p,
                              const Tensor<1, dim, Number>   &normal,
                              const Number                    data_m,
                              const Number                    data_p) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      numerical_tracflux_weak(const Number                  z_m,
                              const Number                  z_p,
                              const Tensor<1, dim, Number> &q_m,
                              const Tensor<1, dim, Number> &q_p,
                              const Number                  t_m,
                              const Number                  t_p,
                              const Tensor<1, dim, Number> &normal,
                              const Number                  data_m,
                              const Number                  data_p) const;
  };



  // The constructor of the numerical flux class takes as arguments the
  // numerical parameters which may be test-case/user dependent. These
  // parameters are stored as class members.
  // In this way they are defined/read from file in one place and then used
  // whenever needed with `numerical_flux.param`, instead of being read/defined
  // multiple times. I hope this does not add much overhead.
  LaxFriedrichsModified::LaxFriedrichsModified(
    IO::ParameterHandler &param)
    : NumericalFluxBase(param)
  {}

  // For the local Lax--Friedrichs flux, the definition is $\hat{\mathbf{F}}
  // =\frac{\mathbf{F}(\mathbf{w}^-)+\mathbf{F}(\mathbf{w}^+)}{2} +
  // \frac{\lambda}{2}\left[\mathbf{w}^--\mathbf{w}^+\right]\otimes
  // \mathbf{n^-}$, where the factor $\lambda =
  // \max\left(\|\mathbf{u}^-\|+c^-, \|\mathbf{u}^+\|+c^+\right)$ gives the
  // maximal wave speed and $c = \sqrt{g h}$ is the speed of the gravity
  // waves.
  //
  // Since the numerical flux is multiplied by the normal vector in the weak
  // form, we multiply by the result by the normal vector for all terms in the
  // equation. In these multiplications, the `operator*` defined above enables
  // a compact notation similar to the mathematical definition.
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    LaxFriedrichsModified::numerical_massflux_weak(
      const Number                  z_m,
      const Number                  z_p,
      const Tensor<1, dim, Number> &q_m,
      const Tensor<1, dim, Number> &q_p,
      const Tensor<1, dim, Number> &normal,
      const Number                  data_m,
      const Number                  data_p) const
  {
    const auto v_m = model.velocity<dim>(z_m, q_m, data_m);
    const auto v_p = model.velocity<dim>(z_p, q_p, data_p);

    const auto lambda_m = v_m.norm() + std::sqrt(model.square_wavespeed(z_m, data_m));
    const auto lambda_p = v_p.norm() + std::sqrt(model.square_wavespeed(z_p, data_p));

    const auto lambda =
      0.5 * std::max(lambda_p, lambda_m);

    return 0.5 * (q_m * normal + q_p * normal) +
           0.5 * lambda * (z_m - z_p);
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    LaxFriedrichsModified::numerical_advflux_weak(
      const Number                  z_m,
      const Number                  z_p,
      const Tensor<1, dim, Number> &q_m,
      const Tensor<1, dim, Number> &q_p,
      const Tensor<1, dim, Number> &normal,
      const Number                  data_m,
      const Number                  data_p) const
  {
    const auto v_m = model.velocity<dim>(z_m, q_m, data_m);
    const auto v_p = model.velocity<dim>(z_p, q_p, data_p);

    const auto lambda_m = v_m.norm() + std::sqrt(model.square_wavespeed(z_m, data_m));
    const auto lambda_p = v_p.norm() + std::sqrt(model.square_wavespeed(z_p, data_p));

    const auto lambda =
      0.5 * std::max(lambda_p, lambda_m);

    const auto flux_m = model.momentum_adv_flux<dim>(z_m, q_m, data_m);
    const auto flux_p = model.momentum_adv_flux<dim>(z_p, q_p, data_p);

    return 0.5 * (flux_m * normal + flux_p * normal) +
           0.5 * lambda * (q_m - q_p);
  }

#ifdef OCEANO_WITH_TRACERS
  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Number>
    LaxFriedrichsModified::numerical_tracflux_weak(
      const Number                     z_m,
      const Number                     z_p,
      const Tensor<1, dim, Number>    &q_m,
      const Tensor<1, dim, Number>    &q_p,
      const Tensor<1, n_tra, Number>  &t_m,
      const Tensor<1, n_tra, Number>  &t_p,
      const Tensor<1, dim, Number>    &normal,
      const Number                     data_m,
      const Number                     data_p) const
  {
    const auto v_m = model.velocity<dim>(z_m, q_m, data_m);
    const auto v_p = model.velocity<dim>(z_p, q_p, data_p);

    const auto lambda_m = v_m.norm() + std::sqrt(model.square_wavespeed(z_m, data_m));
    const auto lambda_p = v_p.norm() + std::sqrt(model.square_wavespeed(z_p, data_p));

    const auto lambda =
      0.5 * std::max(lambda_p, lambda_m);

    const auto flux_m = model.tracer_adv_flux<dim, n_tra>(q_m, t_m);
    const auto flux_p = model.tracer_adv_flux<dim, n_tra>(q_p, t_p);

    Tensor<1, n_tra, Number> numflux;
    for (unsigned int t = 0; t < n_tra; ++t)
      numflux[t] = 0.5 * (flux_m[t] * normal + flux_p[t] * normal) +
                   0.5 * lambda * (z_m * t_m[t] - z_p * t_p[t]);

     return numflux;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    LaxFriedrichsModified::numerical_tracflux_weak(
      const Number                  z_m,
      const Number                  z_p,
      const Tensor<1, dim, Number> &q_m,
      const Tensor<1, dim, Number> &q_p,
      const Number                  t_m,
      const Number                  t_p,
      const Tensor<1, dim, Number> &normal,
      const Number                  data_m,
      const Number                  data_p) const
  {
    const auto v_m = model.velocity<dim>(z_m, q_m, data_m);
    const auto v_p = model.velocity<dim>(z_p, q_p, data_p);

    const auto lambda_m = v_m.norm() + std::sqrt(model.square_wavespeed(z_m, data_m));
    const auto lambda_p = v_p.norm() + std::sqrt(model.square_wavespeed(z_p, data_p));

    const auto lambda =
      0.5 * std::max(lambda_p, lambda_m);

    const auto flux_m = model.tracer_adv_flux(q_m, t_m);
    const auto flux_p = model.tracer_adv_flux(q_p, t_p);

    return 0.5 * (flux_m * normal + flux_p * normal) +
           0.5 * lambda * (z_m * t_m - z_p * t_p);
  }
#endif
} // namespace NumericalFlux

#endif //LAXFRIEDRICHSMODIFIED_HPP
