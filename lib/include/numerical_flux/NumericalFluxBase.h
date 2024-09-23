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

 
  
  // This class implements the numerical flux (Riemann solver). It gets the
  // state from the two sides of an interface and the normal vector, oriented
  // from the side of the solution $\mathbf{w}^-$ towards the solution
  // $\mathbf{w}^+$. Also the data can be passed in a discontinous form, so we
  // have the data from both sides of the interface.
  // In finite volume methods which rely on piece-wise
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
  // cases. In this tutorial, we focus on wave-like solutions in the
  // subsonic regime without strong discontinuities where our
  // basic scheme is sufficient.
  //
  // Nonetheless, the numerical flux is decisive in terms of the numerical
  // dissipation of the overall scheme and influences the admissible time step
  // size with explicit Runge--Kutta methods. We consider two choices, a
  // modified Lax--Friedrichs scheme and the widely used Harten--Lax--van Leer
  // (HLL) flux. For both variants, we first need to get the velocities and
  // pressures from both sides of the interface and evaluate the physical flux.
  //
  // I would have liked to template the numerical flux class with 
  // <int dim, typename Number> which would have been cleaner. But I was not able 
  // to compile the call to the function `numerical_flux_weak()` which take
  // as argument `Tensor<1, Number>` while is receiving
  // `Tensor<1, VectorizedArray<Number>>`. I don't know why, without
  // a template class, everything works. I leave this for future work.  
  class NumericalFluxBase
  {
  public:
    NumericalFluxBase(IO::ParameterHandler &param);
    ~NumericalFluxBase(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      numerical_presflux_strong(const Number                     z_m,
                                const Number                     z_p,
                                const Tensor<1, dim, Number>    &normal,
                                const Number                     data_m,
                                const Number                     data_p) const;

#if defined MODEL_EULER
    Model::Euler model;
#elif defined MODEL_SHALLOWWATER
    Model::ShallowWater model;
#elif defined MODEL_SHALLOWWATERWITHTRACER
    Model::ShallowWaterWithTracer model;
#endif
  };

  NumericalFluxBase::NumericalFluxBase(
    IO::ParameterHandler &param)
    : model(param)
  {} 

#if defined MODEL_SHALLOWWATER
  // We implement the contribution to the numerical flux coming from the terms that
  // have been approximated with the strong formulation of discontinuos Galerkin.
  // This is the case for the pressure term in the shallow water equations.
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    NumericalFluxBase::numerical_presflux_strong(
      const Number                      z_m,
      const Number                      z_p,
      const Tensor<1, dim, Number>     &normal,
      const Number                      data_m,
      const Number                      data_p) const
  {
    const auto h_p = z_p + data_p;
#define NUMERICALFLUXBASE_TUMMOLO
#undef  NUMERICALFLUXBASE_ORLANDO
#if defined NUMERICALFLUXBASE_TUMMOLO
    const auto h_m = z_m + data_m;
    return model.g * 0.25 * (h_p + h_m) * (z_p - z_m) * normal;
#elif defined NUMERICALFLUXBASE_ORLANDO
    return model.g * 0.5 * h_p * (z_p - z_m) * normal;
#endif
  }
#endif
   
} // namespace NumericalFlux

#endif //NUMERICALFLUXBASE_HPP
