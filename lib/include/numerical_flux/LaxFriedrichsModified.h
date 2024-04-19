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
  // as argument `Tensor<1, n_vars, Number>` while is receiving
  // `Tensor<1, n_vars, VectorizedArray<Number>>`. I don't know why, without
  // a template class, everything works. I leave this for future work.
  //
  // In this and the following functions, we use variable suffixes `_m` and
  // `_p` to indicate quantities derived from $\mathbf{w}^-$ and $\mathbf{w}^+$,
  // i.e., values "here" and "there" relative to the current cell when looking
  // at a neighbor cell.
  class LaxFriedrichsModified : public NumericalFluxBase
  {
  public:
    LaxFriedrichsModified(IO::ParameterHandler &param);
    ~LaxFriedrichsModified(){};

    template <int dim, int n_vars, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_vars, Number>
      numerical_flux_weak(const Tensor<1, n_vars, Number> &u_m,
                          const Tensor<1, n_vars, Number> &u_p,
                          const Tensor<1, dim, Number>    &normal,
                          const Number                     data_m,
                          const Number                     data_p) const;
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
  // maximal wave speed and $c = \sqrt{\gamma p / \rho}$ is the speed of
  // sound. Here, we choose two modifications of that expression for reasons
  // of computational efficiency, given the small impact of the flux on the
  // solution. For the above definition of the factor $\lambda$, we would need
  // to take four square roots, two for the two velocity norms and two for the
  // speed of sound on either side. The first modification is hence to rather
  // use $\sqrt{\|\mathbf{u}\|^2+c^2}$ as an estimate of the maximal speed
  // (which is at most a factor of 2 away from the actual maximum, as shown in
  // the introduction). This allows us to pull the square root out of the
  // maximum and get away with a single square root computation. The second
  // modification is to further relax on the parameter $\lambda$---the smaller
  // it is, the smaller the dissipation factor (which is multiplied by the
  // jump in $\mathbf{w}$, which might result in a smaller or bigger
  // dissipation in the end). This allows us to fit the spectrum into the
  // stability region of the explicit Runge--Kutta integrator with bigger time
  // steps. However, we cannot make dissipation too small because otherwise
  // imaginary eigenvalues grow larger. Finally, the current conservative
  // formulation is not energy-stable in the limit of $\lambda\to 0$ as it is
  // not skew-symmetric, and would need additional measures such as split-form
  // DG schemes in that case.
  //
  // Since the numerical flux is multiplied by the normal vector in the weak
  // form, we multiply by the result by the normal vector for all terms in the
  // equation. In these multiplications, the `operator*` defined above enables
  // a compact notation similar to the mathematical definition.
  template <int dim, int n_vars, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_vars, Number>
    LaxFriedrichsModified::numerical_flux_weak(
      const Tensor<1, n_vars, Number>  &u_m,
      const Tensor<1, n_vars, Number>  &u_p,
      const Tensor<1, dim, Number>     &normal,
      const Number                      data_m,
      const Number                      data_p) const
  {
    const auto lambda_m = model.square_speed_estimate<dim, n_vars>(u_m, data_m);
    const auto lambda_p = model.square_speed_estimate<dim, n_vars>(u_p, data_p);

    const auto flux_m = model.flux<dim, n_vars>(u_m, data_m);
    const auto flux_p = model.flux<dim, n_vars>(u_p, data_p);

    const auto lambda =
      0.5 * std::sqrt(std::max(lambda_p, lambda_m));

    return 0.5 * (flux_m * normal + flux_p * normal) +
           0.5 * lambda * (u_m - u_p);
  }
} // namespace NumericalFlux

#endif //LAXFRIEDRICHSMODIFIED_HPP
