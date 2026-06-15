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
 * Author: Martin Kronbichler (copied from), 2020
 *         Luca Arpaia, 2023
 *         Giuseppe Orlando, 2026
 */
#ifndef HARTENVANLEER_H
#define HARTENVANLEER_H

// The following files include the oceano libraries
#include <numerical_flux/NumericalFluxBase.h>

/**
 * Namespace containing the numerical flux.
 */

namespace NumericalFlux
{

  using namespace dealii;



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



  // The constructor of the numerical flux class takes as arguments the
  // numerical parameters which may be test-case/user dependent. These
  // parameters are stored as class members.
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

    const auto flux_m = model.advective_flux<dim>(z_m, q_m, data_m);
    const auto flux_p = model.advective_flux<dim>(z_p, q_p, data_p);

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
#endif //HARTENVANLEER_H
