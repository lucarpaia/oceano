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
#ifndef SHALLOWWATER_HPP
#define SHALLOWWATER_HPP

// The following files include the oceano libraries
#include <io/ParameterReader.h>
#include <physics/BottomFriction.h>
#include <physics/WindStress.h>
#include <physics/Coriolis.h>

/**
 * Namespace containing the model equations.
 */

namespace Model
{

  using namespace dealii;

  // @sect3{Implementation of point-wise operations of the Shallow Water equations}

  // In the following functions, we implement the various problem-specific
  // operators pertaining to the Shallow Water equations. Each function acts on the
  // vector of prognostic variables $[\zeta, h\mathbf{u}]$ that we hold in
  // the two solution vectors for the water height and the discharge.
  // From the solution we computes various derived quantities.
  //
  // First out is the computation of the velocity, that we derive from the
  // discharge variable $h \mathbf{u}$ by division by $h$. One thing to
  // note here is that we decorate all those functions with the keyword
  // `DEAL_II_ALWAYS_INLINE`. This is a special macro that maps to a
  // compiler-specific keyword that tells the compiler to never create a
  // function call for any of those functions, and instead move the
  // implementation <a
  // href="https://en.wikipedia.org/wiki/Inline_function">inline</a> to where
  // they are called. This is critical for performance because we call into some
  // of those functions millions or billions of times: For example, we both use
  // the velocity for the computation of the flux further down, but also for the
  // computation of the pressure, and both of these places are evaluated at
  // every quadrature point of every cell. Making sure these functions are
  // inlined ensures not only that the processor does not have to execute a jump
  // instruction into the function (and the corresponding return jump), but also
  // that the compiler can re-use intermediate information from one function's
  // context in code that comes after the place where the function was called.
  // (We note that compilers are generally quite good at figuring out which
  // functions to inline by themselves. Here is a place where compilers may or
  // may not have figured it out by themselves but where we know for sure that
  // inlining is a win.)
  //
  // Another trick we apply is a separate variable for the inverse depth
  // $\frac{1}{h}$. This enables the compiler to only perform a single
  // division for the flux, despite the division being used at several
  // places. As divisions are around ten to twenty times as expensive as
  // multiplications or additions, avoiding redundant divisions is crucial for
  // performance. We note that taking the inverse first and later multiplying
  // with it is not equivalent to a division in floating point arithmetic due
  // to roundoff effects, so the compiler is not allowed to exchange one way by
  // the other with standard optimization flags. However, it is also not
  // particularly difficult to write the code in the right way.
  //
  // To summarize, the chosen strategy of always inlining and careful
  // definition of expensive arithmetic operations allows us to write compact
  // code without passing all intermediate results around, despite making sure
  // that the code maps to excellent machine code.
  //
  // I would have liked to template the model class with <int dim, typename Number>
  // which would have been cleaner. But I was not able to compile a templated 
  // numerical flux class. For now I have left both classes without template  
  class ShallowWater
  {
  public:
    ShallowWater(IO::ParameterHandler &prm);
    ~ShallowWater(){};
 
    double g;

    std::vector<std::string> vars_name;
    std::vector<std::string> postproc_vars_name;

#if defined PHYSICS_BOTTOMFRICTIONLINEAR
    Physics::BottomFrictionLinear bottom_friction;
#elif defined PHYSICS_BOTTOMFRICTIONMANNING
    Physics::BottomFrictionManning bottom_friction;
#else
    Assert(false, ExcNotImplemented());
    return 0.;
#endif

#if defined PHYSICS_WINDSTRESSGENERAL
    Physics::WindStressGeneral wind_stress;
#elif defined PHYSICS_WINDSTRESSQUADRATIC
    Physics::WindStressQuadratic wind_stress;
#else
    Assert(false, ExcNotImplemented());
    return 0.;
#endif

    Physics::CoriolisBeta coriolis_force;

    // The next function computes the velocity from the vector of conserved
    // variables
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      velocity(const Number                  height,
               const Tensor<1, dim, Number> &discharge,
               const Number                  bathymetry) const;

    // The next function computes the pressure from the vector of conserved
    // variables, using the formula $p = g \frac{h^2}{2}$. As explained above, we use the
    // velocity from the `velocity()` function. Note that we need to
    // specify the first template argument `dim` here because the compiler is
    // not able to deduce it from the arguments of the tensor, whereas the
    // third argument (number type) can be automatically deduced.
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      pressure(const Number height,
               const Number bathymetry) const;

    // Here is the definition of the flux functions, i.e., the definition
    // of the actual equation. We have the mass and the advective flux. Mass flux
    // implementation is straight-forward, being simply the discharge;
    // for the advective flux we need to compute the velocity.
    // The hydrostatic pressure is not included in the flux and it is treated with a
    // a double integration by parts.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      massflux(const Tensor<1, dim, Number> &discharge) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Tensor<1, dim, Number>>
      advectiveflux(const Number                  height,
                    const Tensor<1, dim, Number> &discharge,
                    const Number                  bathymetry) const;

    // Here is the definition of the Shallow Water source function. Thanks to a double
    // integration by parts the pressure appears as a force in the source term and
    // sum up with the bathymetry force giving the term related to the free-surface
    // gradient. Other forces includes the bottom and wind stress and the coriolis force.
    // The computation of the source term involves the conserved variables and $dim+3$
    // non-constant functions (e.g. the bottom friction coefficient).
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source(const Number                    height,
             const Tensor<1, dim, Number>   &discharge,
             const Tensor<1, dim, Number>   &gradient_height,
             const Tensor<1, dim+3, Number> &parameters) const;

    // For ImEx time integration strategy, we separate the source term in a stiff and
    // a non-stiff part. For now the stiff part of the source term is only the bottom friction.
    // If a realistic Manning model is used, the bottom friction term may become very large in
    // shallow area and an extremely small time step would be necessary to integrate it.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source_nonstiff(const Number                    height,
                      const Tensor<1, dim, Number>   &discharge,
                      const Tensor<1, dim, Number>   &gradient_height,
                      const Tensor<1, dim+3, Number> &parameters) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source_stiff(const Number                  height,
                   const Tensor<1, dim, Number> &discharge,
                   const Tensor<1, 2, Number>   &parameters) const;

    // The next function computes an estimate of the square of the graivty wave speed, from
    // the vector of conserved variables
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      square_wavespeed(
        const Number depth,
        const Number bathymetry) const;

    // The next function computes the outgoing Riemann invariant:
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      riemann_invariant_p(
        const Number                  height,
        const Tensor<1, dim, Number> &discharge,
        const Tensor<1, dim, Number> &normal,
        const Number                  bathymetry) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      riemann_invariant_m(
        const Number                  height,
        const Tensor<1, dim, Number> &discharge,
        const Tensor<1, dim, Number> &normal,
        const Number                  bathymetry) const;
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
  ShallowWater::ShallowWater(
    IO::ParameterHandler &prm)
    : bottom_friction(prm)
    , wind_stress(prm)
    , coriolis_force()
  {
    prm.enter_subsection("Physical constants");
    g = prm.get_double("g");
    prm.leave_subsection();

    vars_name = {"free_surface", "hu", "hu"};
    postproc_vars_name = {"velocity", "velocity", "depth"};
  }
  
  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::velocity(const Number                  height,
                           const Tensor<1, dim, Number> &discharge,
                           const Number                  bathymetry) const
  {
    const Number inverse_depth
      = Number(1.) / (height + bathymetry);

    return discharge * inverse_depth;
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::pressure(const Number height,
                           const Number bathymetry) const
  {
    const Number depth = height + bathymetry;
    return 0.5 * g * depth*depth;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::massflux(const Tensor<1, dim, Number> &discharge) const
  {
    return discharge;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Tensor<1, dim, Number>>
    ShallowWater::advectiveflux(const Number                  height,
                                const Tensor<1, dim, Number> &discharge,
                                const Number                  bathymetry) const
  {
    const Tensor<1, dim, Number> v =
      velocity<dim>(height, discharge, bathymetry);
    
    Tensor<1, dim, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        flux[e][d] = discharge[e] * v[d];

    return flux;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::source(const Number                    height,
                         const Tensor<1, dim, Number>   &discharge,
                         const Tensor<1, dim, Number>   &gradient_height,
                         const Tensor<1, dim+3, Number> &parameters) const
  {
    const Tensor<1, dim, Number> v =
      velocity<dim>(height, discharge, parameters[0]);
    const Number depth = height + parameters[0];

    const Tensor<1, dim, Number> bottomfric =
      bottom_friction.source<dim, Number>(v, parameters[1], depth);
    const Tensor<1, dim, Number> windstress =
      wind_stress.source<dim, Number>(&parameters[2]);
    const Tensor<1, dim, Number> coriolis =
      coriolis_force.source<dim, Number>(discharge, parameters[4]);

    Tensor<1, dim, Number> source =
        - g * depth * gradient_height
	- bottomfric
        + windstress
        + coriolis;

    return source;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::source_nonstiff(
      const Number                    height,
      const Tensor<1, dim, Number>   &discharge,
      const Tensor<1, dim, Number>   &gradient_height,
      const Tensor<1, dim+3, Number> &parameters) const
  {
    const Number depth = height + parameters[0];

    const Tensor<1, dim, Number> windstress =
      wind_stress.source<dim, Number>(&parameters[2]);
    const Tensor<1, dim, Number> coriolis =
      coriolis_force.source<dim, Number>(discharge, parameters[4]);

    Tensor<1, dim, Number> source =
        - g * depth * gradient_height
        + windstress
        + coriolis;

    return source;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::source_stiff(
      const Number                  height,
      const Tensor<1, dim, Number> &discharge,
      const Tensor<1, 2, Number>   &parameters) const
  {
    const Tensor<1, dim, Number> v =
      velocity<dim>(height, discharge, parameters[0]);
    const Number depth = height + parameters[0];

    return -bottom_friction.source<dim, Number>(v, parameters[1], depth);
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::square_wavespeed(
      const Number                  height,
      const Number                  bathymetry) const
  {
    return g * (height + bathymetry);
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::riemann_invariant_p(
      const Number                  height,
      const Tensor<1, dim, Number> &discharge,
      const Tensor<1, dim, Number> &normal,
      const Number                  bathymetry) const
  {
    const auto v = velocity<dim>(height, discharge, bathymetry);
    const auto c = std::sqrt(
      square_wavespeed(height, bathymetry));
    Number u = v * normal;

    return u + 2. * c;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::riemann_invariant_m(
      const Number                  height,
      const Tensor<1, dim, Number> &discharge,
      const Tensor<1, dim, Number> &normal,
      const Number                  bathymetry) const
  {
    const auto v = velocity<dim>(height, discharge, bathymetry);
    const auto c = std::sqrt(
      square_wavespeed(height, bathymetry));
    Number u = v * normal;

    return u - 2. * c;
  }
} // namespace Model
#endif //SHALLOWWATER_HPP
