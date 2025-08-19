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
#include <physics/ViscosityCoefficient.h>
#include <physics/Coriolis.h>

/**
 * Namespace containing the model equations.
 */

namespace Model
{

  using namespace dealii;

  // @sect3{Implementation of point-wise operations of the Shallow Water equations}

  // In the following functions, we implement the various problem-specific
  // operators pertaining to the Shallow Water equations. The Shallow Water
  // equations may be implemented in different ways. In this class we
  // implement the Shallow Water equation in conservative form but using as
  // prognostic variables the water height and the velocity. As a result each
  // function acts on the vector of prognostic variables $[\zeta, \mathbf{u}]$,
  // on their gradients and on the bathymetry function.
  // From the solution we computes various derived quantities.
  //
  // One thing to note here is that we decorate all those functions with the keyword
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
  // To summarize, the chosen strategy of always inlining and careful
  // definition of expensive arithmetic operations allows us to write compact
  // code without passing all intermediate results around, despite making sure
  // that the code maps to excellent machine code.
  //
  // I would have liked to template the model class with <int dim, typename Number>
  // which would have been cleaner. But I was not able to compile the call
  // to the class functions which take `Number` while receive
  // `VectorizedArray<Number>`. I don't know why, without a template class,
  // everything works. For now I have left both classes without template.
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

#if defined PHYSICS_VISCOSITYCOEFFICIENTCONSTANT
    Physics::ViscosityCoefficientConstant viscosity_coefficient;
#elif defined PHYSICS_VISCOSITYCOEFFICIENTSMAGORINSKY
    Physics::ViscosityCoefficientSmagorinsky viscosity_coefficient;
#else
    Assert(false, ExcNotImplemented());
    return 0.;
#endif

    Physics::CoriolisBeta coriolis_force;

    // The next function computes the model variable names. It is not a
    // constant function beacause it modifies the class members.
    template <int dim, int n_tra>
    inline DEAL_II_ALWAYS_INLINE //
      void
      set_vars_name();

    // The next function computes the depth from the vector of conserved
    // variables
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      depth(const Number height,
            const Number bathymetry) const;

    // The next function computes the velocity from the vector of conserved
    // variables
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      velocity(const Number                  height,
               const Tensor<1, dim, Number> &velocity,
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
    // a double integration by parts. We also include a flux function with both
    // advective and diffusive fluxes.
    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      mass_flux(const Number                  height,
                const Tensor<1, dim, Number> &velocity,
                const Number                  bathymetry) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Tensor<1, dim, Number>>
      momentum_adv_flux(const Number                  height,
                        const Tensor<1, dim, Number> &velocity,
                        const Number                  bathymetry) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Tensor<1, dim, Number>>
      momentum_adv_diff_flux(const Number                    height,
                             const Tensor<1, dim, Number>   &velocity,
                             const Tensor<dim, dim, Number> &gradient_velocity,
                             const Number                    bathymetry,
                             const Number                    area) const;

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
             const Tensor<1, dim, Number>   &velocity,
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
                      const Tensor<1, dim, Number>   &velocity,
                      const Tensor<1, dim, Number>   &gradient_height,
                      const Tensor<1, dim+3, Number> &parameters) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      source_stiff(const Number                  height,
                   const Tensor<1, dim, Number> &velocity,
                   const Number                  bathymetry,
                   const Number                  drag_coefficient) const;

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
        const Tensor<1, dim, Number> &velocity,
        const Tensor<1, dim, Number> &normal,
        const Number                  bathymetry) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      riemann_invariant_m(
        const Number                  height,
        const Tensor<1, dim, Number> &velocity,
        const Tensor<1, dim, Number> &normal,
        const Number                  bathymetry) const;

    // The next function computes the weights that multiply the prognostic variable in
    // the momentum equation, to obtain a transformation between conservative and
    // non-conservative variables (discharge/velocity). Since we use velocity as
    // prognostic variable, the weight that bring to the conservative discharge is the
    // water depth.
    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      factor_to_discharge(const Number height,
                          const Number bathymetry) const;

    template <typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      factor_from_velocity(const Number height,
                           const Number bathymetry) const;
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
    , viscosity_coefficient(prm)
    , coriolis_force()
  {
    prm.enter_subsection("Physical constants");
    g = prm.get_double("g");
    prm.leave_subsection();
  }

  template <int dim, int n_tra>
  inline DEAL_II_ALWAYS_INLINE //
    void
    ShallowWater::set_vars_name()
  {
    vars_name.push_back("free_surface");
    for (unsigned int d = 0; d < dim; ++d)
      vars_name.push_back("u");
    for (unsigned int d = 0; d < dim; ++d)
      postproc_vars_name.push_back("velocity");
    postproc_vars_name.push_back("depth");
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::depth(const Number height,
                        const Number bathymetry) const
  {
    return std::max(height + bathymetry, Number(5.e-3));
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::velocity(const Number                  /*height*/,
                           const Tensor<1, dim, Number> &velocity,
                           const Number                  /*bathymetry*/) const
  {
    return velocity;
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::pressure(const Number height,
                           const Number bathymetry) const
  {
    const Number h = depth(height, bathymetry);
    return 0.5 * g * h*h;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::mass_flux(const Number                  height,
                            const Tensor<1, dim, Number> &velocity,
                            const Number                  bathymetry) const
  {
    return depth(height, bathymetry) * velocity;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Tensor<1, dim, Number>>
    ShallowWater::momentum_adv_flux(const Number                  height,
                                    const Tensor<1, dim, Number> &velocity,
                                    const Number                  bathymetry) const
  {
    const Number h = depth(height, bathymetry);

    Tensor<1, dim, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        flux[e][d] = h * velocity[e] * velocity[d];

    return flux;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Tensor<1, dim, Number>>
    ShallowWater::momentum_adv_diff_flux(
      const Number                    height,
      const Tensor<1, dim, Number>   &velocity,
      const Tensor<dim, dim, Number> &gradient_velocity,
      const Number                    bathymetry,
      const Number                    area) const
  {
    const Number h = depth(height, bathymetry);

    const Number nu = viscosity_coefficient.value<dim, Number>(gradient_velocity, area);
    const Number div = gradient_velocity[0][0] + gradient_velocity[1][1];

    Tensor<1, dim, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        flux[e][d] = h * velocity[e] * velocity[d]
          - nu * (gradient_velocity[e][d] + gradient_velocity[d][e]);
    for (unsigned int e = 0; e < dim; ++e)
      flux[e][e] += nu * div;

    return flux;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::source(const Number                    height,
                         const Tensor<1, dim, Number>   &velocity,
                         const Tensor<1, dim, Number>   &gradient_height,
                         const Tensor<1, dim+3, Number> &parameters) const
  {
    const Number h = depth(height, parameters[0]);

    const Tensor<1, dim, Number> bottomfric =
      bottom_friction.source<dim, Number>(velocity, parameters[1], h);
    const Tensor<1, dim, Number> windstress =
      wind_stress.source<dim, Number>(&parameters[2]);
    const Tensor<1, dim, Number> coriolis =
      coriolis_force.source<dim, Number>(h*velocity, parameters[4]);

    Tensor<1, dim, Number> source =
        - g * h * gradient_height
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
      const Tensor<1, dim, Number>   &velocity,
      const Tensor<1, dim, Number>   &gradient_height,
      const Tensor<1, dim+3, Number> &parameters) const
  {
    const Number h = depth(height, parameters[0]);

    const Tensor<1, dim, Number> windstress =
      wind_stress.source<dim, Number>(&parameters[2]);
    const Tensor<1, dim, Number> coriolis =
      coriolis_force.source<dim, Number>(h*velocity, parameters[4]);

    Tensor<1, dim, Number> source =
        - g * h * gradient_height
        + windstress
        + coriolis;

    return source;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::source_stiff(
      const Number                  height,
      const Tensor<1, dim, Number> &velocity,
      const Number                  bathymetry,
      const Number                  drag_coefficient) const
  {
    const Number h = depth(height, bathymetry);

    return -bottom_friction.source<dim, Number>(velocity, drag_coefficient, h);
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::square_wavespeed(
      const Number                  height,
      const Number                  bathymetry) const
  {
    return g * depth(height, bathymetry);
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::riemann_invariant_p(
      const Number                  height,
      const Tensor<1, dim, Number> &velocity,
      const Tensor<1, dim, Number> &normal,
      const Number                  bathymetry) const
  {
    const auto c = std::sqrt(
      square_wavespeed(height, bathymetry));
    Number u = velocity * normal;

    return u + 2. * c;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::riemann_invariant_m(
      const Number                  height,
      const Tensor<1, dim, Number> &velocity,
      const Tensor<1, dim, Number> &normal,
      const Number                  bathymetry) const
  {
    const auto c = std::sqrt(
      square_wavespeed(height, bathymetry));
    Number u = velocity * normal;

    return u - 2. * c;
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::factor_to_discharge(
      const Number height,
      const Number bathymetry) const
  {
    return depth(height, bathymetry);
  }

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::factor_from_velocity(
      const Number /*height*/,
      const Number /*bathymetry*/) const
  {
    return Number(1.);
  }
} // namespace Model
#endif //SHALLOWWATER_HPP
