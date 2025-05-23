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
#ifndef HARTENVANLEER_HPP
#define HARTENVANLEER_HPP

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
  // as argument `Tensor<1, n_vars, Number>` while is receiving
  // `Tensor<1, n_vars, VectorizedArray<Number>>`. I don't know why, without
  // a template class, everything works. I leave this for future work.
  //
  // In this and the following functions, we use variable suffixes `_m` and
  // `_p` to indicate quantities derived from $\mathbf{w}^-$ and $\mathbf{w}^+$,
  // i.e., values "here" and "there" relative to the current cell when looking
  // at a neighbor cell.
  class HartenVanLeer : public NumericalFluxBase
  {
  public:
    HartenVanLeer(IO::ParameterHandler &param);
    ~HartenVanLeer(){};

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
  };

 
 
  // For the model class we do not use an implementation file. This
  // is because of the fact the all the function called are templated
  // or inlined. Both templated and inlined functions are hard to be separated
  // between declaration and implementation. We keep them in the header file.
  //
  // The constructor of the numerical flux class takes as arguments the 
  // numerical parameters which may be test-case/user dependent. These
  // parameters are stored as class members.
  // In this way they are defined/read from file in one place and then used 
  // whenever needed with `numerical_flux.param`, instead of being read/defined 
  // multiple times. I hope this does not add much overhead. The physical parameter
  // `gamma` is also passed to construct the model class.
  HartenVanLeer::HartenVanLeer(
    IO::ParameterHandler &param)
    : NumericalFluxBase(param)
  {}

  // For the HLL flux, we follow the formula from literature, introducing an
  // additional weighting of the two states from Lax--Friedrichs by a
  // parameter $s$. It is derived from the physical transport directions of
  // the governing equations in terms of the current direction of velocity and
  // sound speed. For the velocity, we here choose a simple arithmetic average
  // which is sufficient for DG scenarios and moderate jumps in material
  // parameters.
  //
  // Since the numerical flux is multiplied by the normal vector in the weak
  // form, we multiply by the result by the normal vector for all terms in the
  // equation. In these multiplications, the `operator*` defined above enables
  // a compact notation similar to the mathematical definition.
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    HartenVanLeer::numerical_massflux_weak(
      const Number                      z_m,
      const Number                      z_p,
      const Tensor<1, dim, Number>     &q_m,
      const Tensor<1, dim, Number>     &q_p,
      const Tensor<1, dim, Number>     &normal,
      const Number                      data_m,
      const Number                      data_p) const
  {
    const auto velocity_m = model.velocity<dim>(z_m, q_m, data_m);
    const auto velocity_p = model.velocity<dim>(z_p, q_p, data_p);

    const auto csquare_m = model.square_wavespeed(z_m, data_m);
    const auto csquare_p = model.square_wavespeed(z_p, data_m);

    const auto flux_m = model.mass_flux<dim>(q_m);
    const auto flux_p = model.mass_flux<dim>(q_p);

    const auto avg_velocity_normal =
      0.5 * ((velocity_m + velocity_p) * normal);
    const auto   avg_c = std::sqrt(std::abs(
      0.5 * (csquare_p + csquare_m)));
    const Number s_pos =
      std::max(Number(), avg_velocity_normal + avg_c);
    const Number s_neg =
      std::min(Number(), avg_velocity_normal - avg_c);
    const Number inverse_s = Number(1.) / (s_pos - s_neg);

    return inverse_s *
           ((s_pos * (flux_m * normal) - s_neg * (flux_p * normal)) -
           s_pos * s_neg * (z_m - z_p));
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    HartenVanLeer::numerical_advflux_weak(
      const Number                      z_m,
      const Number                      z_p,
      const Tensor<1, dim, Number>     &q_m,
      const Tensor<1, dim, Number>     &q_p,
      const Tensor<1, dim, Number>     &normal,
      const Number                      data_m,
      const Number                      data_p) const
  {
    const auto velocity_m = model.velocity<dim>(z_m, q_m, data_m);
    const auto velocity_p = model.velocity<dim>(z_p, q_p, data_p);

    const auto csquare_m = model.square_wavespeed(z_m, data_m);
    const auto csquare_p = model.square_wavespeed(z_p, data_m);

    const auto flux_m = model.momentum_adv_flux<dim>(z_m, q_m, data_m);
    const auto flux_p = model.momentum_adv_flux<dim>(z_p, q_p, data_p);

    const auto avg_velocity_normal =
      0.5 * ((velocity_m + velocity_p) * normal);
    const auto   avg_c = std::sqrt(std::abs(
      0.5 * (csquare_p + csquare_m)));
    const Number s_pos =
      std::max(Number(), avg_velocity_normal + avg_c);
    const Number s_neg =
      std::min(Number(), avg_velocity_normal - avg_c);
    const Number inverse_s = Number(1.) / (s_pos - s_neg);

    return inverse_s *
           ((s_pos * (flux_m * normal) - s_neg * (flux_p * normal)) -
           s_pos * s_neg * (q_m - q_p));
  }
} // namespace NumericalFlux

#endif //HARTENVANLEER_HPP
