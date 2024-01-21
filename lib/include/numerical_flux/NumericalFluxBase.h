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
#ifndef NUMERICALFLUXBASE_HPP
#define NUMERICALFLUXBASE_HPP

// The following files include the oceano libraries 
#if defined MODEL_EULER
#include <model/Euler.h>
#elif defined MODEL_SHALLOWWATER
#include <model/ShallowWater.h>
#endif

/**
 * Namespace containing the numerical flux.
 */

namespace NumericalFlux
{

  using namespace dealii;

 
  
  // This next function is a helper to simplify the implementation of the
  // numerical flux, implementing the action of a tensor of tensors (with
  // non-standard outer dimension of size `n_vars`, so the standard overloads
  // provided by deal.II's tensor classes do not apply here) with another
  // tensor of the same inner dimension, i.e., a matrix-vector product.
  template <int n_components, int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_components, Number>
    operator*(const Tensor<1, n_components, Tensor<1, dim, Number>> &matrix,
              const Tensor<1, dim, Number> &                         vector)
  {
    Tensor<1, n_components, Number> result;
    for (unsigned int d = 0; d < n_components; ++d)
      result[d] = matrix[d] * vector;
    return result;
  }
    
  // This function implements the numerical flux (Riemann solver). It gets the
  // state from the two sides of an interface and the normal vector, oriented
  // from the side of the solution $\mathbf{w}^-$ towards the solution
  // $\mathbf{w}^+$. In finite volume methods which rely on piece-wise
  // constant data, the numerical flux is the central ingredient as it is the
  // only place where the physical information is entered. In DG methods, the
  // numerical flux is less central due to the polynomials within the elements
  // and the physical flux used there. As a result of higher-degree
  // interpolation with consistent values from both sides in the limit of a
  // continuous solution, the numerical flux can be seen as a control of the
  // jump of the solution from both sides to weakly impose continuity. It is
  // important to realize that a numerical flux alone cannot stabilize a
  // high-order DG method in the presence of shocks, and thus any DG method
  // must be combined with further shock-capturing techniques to handle those
  // cases. In this tutorial, we focus on wave-like solutions of the Euler
  // equations in the subsonic regime without strong discontinuities where our
  // basic scheme is sufficient.
  //
  // Nonetheless, the numerical flux is decisive in terms of the numerical
  // dissipation of the overall scheme and influences the admissible time step
  // size with explicit Runge--Kutta methods. We consider two choices, a
  // modified Lax--Friedrichs scheme and the widely used Harten--Lax--van Leer
  // (HLL) flux. For both variants, we first need to get the velocities and
  // pressures from both sides of the interface and evaluate the physical
  // Euler flux.
  //
  // I would have liked to template the numerical flux class with 
  // <int dim, typename Number> which would have been cleaner. But I was not able 
  // to compile the call to the function `euler_numerical_flux()` which take
  // as argument `Tensor<1, n_vars, Number>` while is receiving 
  // `Tensor<1, n_vars, VectorizedArray<Number>>`. I don't know why, without
  // a template class, everything works. I leave this for future work.  
  class NumericalFluxBase
  {
  public:
    NumericalFluxBase(IO::ParameterHandler &param);
    ~NumericalFluxBase(){};
                                    
  public:
#if defined MODEL_EULER
    Model::Euler model;
#elif defined MODEL_SHALLOWWATER
    Model::ShallowWater model;
#endif
  };

  NumericalFluxBase::NumericalFluxBase(
    IO::ParameterHandler &param)
    : model(param)
  {} 
   
} // namespace NumericalFlux

#endif //NUMERICALFLUXBASE_HPP
