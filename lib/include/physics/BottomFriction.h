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
#ifndef BOTTOMFRICTION_HPP
#define BOTTOMFRICTION_HPP

/**
 * Namespace containing the so-called phyisics of the governing equations.
 * They are the different parametrizations.
 */

namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the bottom friction}

  // In the following classes, we implement the different formulations
  // of the bottom friction. We use a base class that is used to store the
  // physical constants appearing into the different formulations. 
  class BottomFrictionBase
  {
  public:
    BottomFrictionBase(IO::ParameterHandler &prm);
    ~BottomFrictionBase(){};

    double g;

    // The next function is the one that actually computes the bottom friction.
    // It is overloaded by the same function defined in the derived classes.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Tensor<1, dim, Number> &velocity,
             const Number                  drag_parameter,
             const Number                  depth) const;
  };
    
  // Not surprisingly, the constructor of the base class takes as arguments 
  // only the parameters handler class in order to read the test-case/user 
  // dependent parameters.
  BottomFrictionBase::BottomFrictionBase(
    IO::ParameterHandler &prm)
  {
    prm.enter_subsection("Physical constants");
    g = prm.get_double("g");
    prm.leave_subsection();
  }
  
  
#if defined PHYSICS_BOTTOMFRICTIONLINEAR
  // The first bottom friction formulation is the simpler one. A linear 
  // friction model of the form:
  // \[
  // \boldsymbol{F} = C_D \boldsymbol{u}
  // \]
  // with $C_D$ the drag coefficient that can vary in space. This is a highly 
  // simplified model given the turbulent nature of the ocean bottom 
  // boundary layers.
  class BottomFrictionLinear : public BottomFrictionBase
  {
  public:
    BottomFrictionLinear(IO::ParameterHandler &prm);
    ~BottomFrictionLinear(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Tensor<1, dim, Number> &velocity,
             const Number                  drag_coefficient,
             const Number                  depth) const;
  };

  BottomFrictionLinear::BottomFrictionLinear(
    IO::ParameterHandler &param)
    : BottomFrictionBase(param)
  {}
  
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    BottomFrictionLinear::source(
      const Tensor<1, dim, Number> &velocity,
      const Number                  drag_coefficient,
      const Number                  /*depth*/) const
  {
    Tensor<1, dim, Number> source;
    for (unsigned int d = 0; d < dim; ++d)
      source[d] = drag_coefficient * velocity[d];

    return source;
  }



#elif defined PHYSICS_BOTTOMFRICTIONMANNING
  // The third bottom friction formulation is the Manning friction. 
  // The Manning friction takes the form:
  // \[
  // \boldsymbol{F} = g n^2 \frac{||\boldsymbol{u}||}{h^{1/3}} \boldsymbol{u}
  // \]
  // with $n$ the Manning coefficient that can vary in space. Manning
  // friction is recommended in shallow water simulations. Note that the original
  // formulation involves the $\frac{1}{3}$-th power of the water depth.
  // Since `std::pow()` has pretty slow implementations on some systems, we replace 
  // it by logarithm followed by exponentiation, which is mathematically equivalent 
  // but usually much better optimized. This formula might lose accuracy in the last digits
  // for very small numbers compared to `std::pow()`, but we are happy with
  // it anyway, since small numbers map to data close to 1. The inverse of is also 
  // a quite expensive operation and we compute it, as well as the velocity norm just once 
  // before looping on the dimensions.
  class BottomFrictionManning : public BottomFrictionBase
  {
  public:
    BottomFrictionManning(IO::ParameterHandler &prm);
    ~BottomFrictionManning(){};

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Tensor<1, dim, Number> &velocity,
             const Number                  manning,
             const Number                  depth) const;
  };

  BottomFrictionManning::BottomFrictionManning(
    IO::ParameterHandler &param)
    : BottomFrictionBase(param)
  {}  
  
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    BottomFrictionManning::source(
      const Tensor<1, dim, Number> &velocity,
      const Number                  manning,
      const Number                  depth) const
  {
    Tensor<1, dim, Number> source;
    Number inv_depth = 1. / ( std::exp( std::log(depth) * 0.33333333 ) );
    Number velocity_norm = velocity.norm();
    for (unsigned int d = 0; d < dim; ++d)
      source[d] = g * manning * manning * inv_depth * velocity_norm
        * velocity[d];

    return source;
  }
#endif

} // namespace Physics
#endif //BOTTOMFRICTION_HPP
