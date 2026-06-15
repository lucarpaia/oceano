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
#ifndef LAXFRIEDRICHSMODIFIED_H
#define LAXFRIEDRICHSMODIFIED_H

// The following files include the oceano libraries
#include <numerical_flux/NumericalFluxBase.h>

/**
 * Namespace containing the numerical flux.
 */

namespace NumericalFlux
{

  using namespace dealii;



  // In this and the following functions, we use `z` for the mass variable,
  // (for us the water height or the free-surface position) and `q`
  // stands for the discharge. We use variable suffixes `_m` and
  // `_p` to indicate quantities derived from $\mathbf{w}^-$ and $\mathbf{w}^+$,
  // i.e., values "here" and "there" relative to the current cell when looking
  // at a neighbor cell. We also make use of the standard overloads
  // provided by deal.II's tensor classes for implementing the dot product.
  //
  // We do not use an implementation file because of the fact the all the
  // class function called are templated or inlined. Both templated and inlined
  // functions are hard to be separated between declaration and implementation.
  // We keep them in the header file.
  //
  // I would have liked to template the numerical flux class with
  // <int dim, typename Number> which would have been cleaner. But I was not able
  // to compile the call to the function `numerical_flux_weak()` which take
  // as argument `Tensor<1, dim, Number>` while is receiving
  // `Tensor<1, dim, VectorizedArray<Number>>`. I don't know why, without
  // a template class, everything works. I leave this for future work.
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
                              const Number                  zb_m,
                              const Number                  zb_p) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      numerical_advflux_weak(const Number                  z_m,
                             const Number                  z_p,
                             const Tensor<1, dim, Number> &q_m,
                             const Tensor<1, dim, Number> &q_p,
                             const Tensor<1, dim, Number> &normal,
                             const Number                  zb_m,
                             const Number                  zb_p) const;

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
                              const Number                    zb_m,
                              const Number                    zb_p) const;

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
                              const Number                  zb_m,
                              const Number                  zb_p) const;
  };



  // The constructor of the numerical flux class takes as arguments the
  // numerical parameters which may be test-case/user dependent. These
  // parameters are stored as class members:
  LaxFriedrichsModified::LaxFriedrichsModified(
    IO::ParameterHandler &param)
    : NumericalFluxBase(param)
  {}

  // For the local Lax--Friedrichs flux in the continuity equation, the definition is:
  //
  // \begin{equation*}
  //   \rpth{\ave{\disc^{*,(m)}} + \frac{\lambda^{(m)}}{2}\jump{\zeta^{*,(m)}}}
  // \end{equation*}
  //
  // In the computation of the integral at the interface between two elements
  // a Rusanov flux \cite{rusanov:1962} is used, so that
  //
  // \begin{equation*}
  //   \lambda^{(m)} = \max\rpth{\left|\vel^{+,(m)} \cdot \bm{n}^{+}\right| + \sqrt{gh^{+,(m)}},
  //      \left|\vel^{-,(m)} \cdot \bm{n}^{-}\right| + \sqrt{gh^{-,(m)}}}.
  // \end{equation*}
  //
  // We employ the so-called hydrostatic reconstruction (Audusse,2004, Gross,2002),
  // which consists in selecting a single-valued bathymetry at the interface when
  // the bathymetry is discontinuous, i.e.
  //
  // \begin{equation*}
  //    z^{*}_{b} = \min\rpth{z_{b}^{-}, z_{b}^{+}},
  // \end{equation*}
  //
  // and modifying the water depth, the discharge and the free-surface, accordingly:
  //
  // \begin{eqnarray*}
  //    h^{*,+} = \max\rpth{\zeta^{+} + z^{*}_{b}, 0}, \qquad h^{*,-} = \max\rpth{\zeta^{-} + z^{*}_{b}, 0}
  // \end{eqnarray*}
  // \begin{eqnarray*}
  //   \disc^{*,+} = h^{*,+} \vel^{+},\qquad\qquad\qquad\,\, \disc^{*,-} = h^{*,-} \vel^{-},
  // \end{eqnarray*}
  // \begin{eqnarray*}
  //   \zeta^{*,+} = \max\rpth{\zeta^{+}, -z^{*}_{b}}, \quad \textrm{and}
  //     \quad \zeta^{*,-} = \max\rpth{\zeta^{-}, -z^{*}_{b}}.
  // \end{eqnarray*}
  // This is necessary, in presence of discontinuous bathymetry, to have positivity
  // but it also avoids spurious fluxes.
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
      const Number                  zb_m,
      const Number                  zb_p) const
  {
    const auto v_m = model.velocity<dim>(z_m, q_m, zb_m);
    const auto v_p = model.velocity<dim>(z_p, q_p, zb_p);

    const auto lambda_m = std::abs(v_m * normal)
      + std::sqrt(model.square_wavespeed(z_m, zb_m));
    const auto lambda_p = std::abs(v_p * normal)
      + std::sqrt(model.square_wavespeed(z_p, zb_p));

    const auto lambda = std::max(lambda_p, lambda_m);
    const auto zb_star = std::min(zb_m, zb_p);

    const auto flux_m = model.depth(z_m, zb_star) * v_m;
    const auto flux_p = model.depth(z_p, zb_star) * v_p;

    const auto zcorr_m = std::max(z_m, -zb_star);
    const auto zcorr_p = std::max(z_p, -zb_star);

    return 0.5 * (flux_m * normal + flux_p * normal) +
           0.5 * lambda * (zcorr_m - zcorr_p);
  }

  // For the local Lax--Friedrichs flux in the momentum equation, the definition is:
  //
  // \begin{eqnarray*}
  //   \rpth{\ave{\disc^{*,(m)} \otimes \vel^{(m)}} + \frac{\lambda^{(m)}}{2}\tjump{\disc^{*,(m)}}}
  // \end{equation*}
  //
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    LaxFriedrichsModified::numerical_advflux_weak(
      const Number                  z_m,
      const Number                  z_p,
      const Tensor<1, dim, Number> &q_m,
      const Tensor<1, dim, Number> &q_p,
      const Tensor<1, dim, Number> &normal,
      const Number                  zb_m,
      const Number                  zb_p) const
  {
    const auto v_m = model.velocity<dim>(z_m, q_m, zb_m);
    const auto v_p = model.velocity<dim>(z_p, q_p, zb_p);

    const auto lambda_m = std::abs(v_m * normal)
      + std::sqrt(model.square_wavespeed(z_m, zb_m));
    const auto lambda_p = std::abs(v_p * normal)
      + std::sqrt(model.square_wavespeed(z_p, zb_p));

    const auto lambda = std::max(lambda_p, lambda_m);
    const auto zb_star = std::min(zb_m, zb_p);

    const auto hu_m = model.depth(z_m, zb_star) * v_m;
    const auto hu_p = model.depth(z_p, zb_star) * v_p;
    const auto flux_m = model.advective_flux<dim>(z_m, hu_m, zb_star);
    const auto flux_p = model.advective_flux<dim>(z_p, hu_p, zb_star);

    return 0.5 * (flux_m * normal + flux_p * normal) +
           0.5 * lambda * (hu_m - hu_p);
  }

#ifdef OCEANO_WITH_TRACERS
  // The tracer numerical flux follows similarly to the numerical mass flux as we
  // have to preserve the consistency with the continuity equation.
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
      const Number                     zb_m,
      const Number                     zb_p) const
  {
    const auto v_m = model.velocity<dim>(z_m, q_m, zb_m);
    const auto v_p = model.velocity<dim>(z_p, q_p, zb_p);

    const auto lambda_m = std::abs(v_m * normal)
      + std::sqrt(model.square_wavespeed(z_m, zb_m));
    const auto lambda_p = std::abs(v_p * normal)
      + std::sqrt(model.square_wavespeed(z_p, zb_p));

    const auto lambda = std::max(lambda_p, lambda_m);
    const auto zb_star = std::min(zb_m, zb_p);

    const auto hu_m = model.depth(z_m, zb_star) * v_m;
    const auto hu_p = model.depth(z_p, zb_star) * v_p;
    const auto flux_m = model.tracer_advective_flux<dim, n_tra>(hu_m, t_m);
    const auto flux_p = model.tracer_advective_flux<dim, n_tra>(hu_p, t_p);

    const auto h_m = model.depth(z_m, zb_star);
    const auto h_p = model.depth(z_p, zb_star);

    Tensor<1, n_tra, Number> numflux;
    for (unsigned int t = 0; t < n_tra; ++t)
      numflux[t] = 0.5 * (flux_m[t] * normal + flux_p[t] * normal) +
                   0.5 * lambda * (h_m * t_m[t] - h_p * t_p[t]);

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
      const Number                  zb_m,
      const Number                  zb_p) const
  {
    const auto v_m = model.velocity<dim>(z_m, q_m, zb_m);
    const auto v_p = model.velocity<dim>(z_p, q_p, zb_p);

    const auto lambda_m = std::abs(v_m * normal)
      + std::sqrt(model.square_wavespeed(z_m, zb_m));
    const auto lambda_p = std::abs(v_p * normal)
      + std::sqrt(model.square_wavespeed(z_p, zb_p));

    const auto lambda = std::max(lambda_p, lambda_m);
    const auto zb_star = std::min(zb_m, zb_p);

    const auto hu_m = model.depth(z_m, zb_star) * v_m;
    const auto hu_p = model.depth(z_p, zb_star) * v_p;
    const auto flux_m = model.tracer_advective_flux(hu_m, t_m);
    const auto flux_p = model.tracer_advective_flux(hu_p, t_p);

    const auto h_m = model.depth(z_m, zb_star);
    const auto h_p = model.depth(z_p, zb_star);

    return 0.5 * (flux_m * normal + flux_p * normal) +
           0.5 * lambda * (h_m * t_m - h_p * t_p);
  }
#endif
} // namespace NumericalFlux
#endif //LAXFRIEDRICHSMODIFIED_H
