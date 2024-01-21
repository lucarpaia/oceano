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

/**
 * Namespace containing the model equations.
 */

namespace Model
{

  using namespace dealii;

  // @sect3{Implementation of point-wise operations of the Shallow Water equations}

  // In the following functions, we implement the various problem-specific
  // operators pertaining to the Shallow Water equations. Each function acts on the
  // vector of conserved variables $[\zeta, h\mathbf{u}]$ that we hold in
  // the solution vectors, and computes various derived quantities.
  //
  // First out is the computation of the velocity, that we derive from the
  // momentum variable $h \mathbf{u}$ by division by $h$. One thing to
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

    // The next function computes the velocity from the vector of conserved
    // variables
    template <int dim, int n_vars, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      velocity(const Tensor<1, n_vars, Number> &conserved_variables) const;

    // The next function computes the pressure from the vector of conserved
    // variables, using the formula $p = g \frac{h^2}{2}$. As explained above, we use the
    // velocity from the `velocity()` function. Note that we need to
    // specify the first template argument `dim` here because the compiler is
    // not able to deduce it from the arguments of the tensor, whereas the
    // third argument (number type) can be automatically deduced.
    template <int dim, int n_vars, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      pressure(const Tensor<1, n_vars, Number> &conserved_variables) const;

    // Here is the definition of the Shallow Water flux function, i.e., the definition
    // of the actual equation. We use only advective flux. Given the velocity
    // (that the compiler optimization will make sure are done only once),
    // this is straight-forward given the equation stated in the introduction.
    // The hydrostatic pressure in the flux is treated in a non-conservative fashion
    // and added into the source term. For smooth problems in the low Froude regime,
    // at the scales typical of the coastal ocean, the two formulations are equivalent.
    // The non conservative treatment of the pressure avoids all togheter the well-balanced
    // issue which typically involves less complicated numerical fluxes.
    template <int dim, int n_vars, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_vars, Tensor<1, dim, Number>>
      flux(const Tensor<1, n_vars, Number> &conserved_variables) const;

    // Here is the definition of the Shallow Water source function. For now we have coded
    // only the pressure force and the bathymetry force. The computation of the source
    // term involves the conserved variables and non-constant functions (e.g. the bathymetry), that
    // means that the source term must be recomputed at each time step and can add a significant overhead.
    // Note that the pressure and bathyemtry terms are sum into a single term where only the gradient
    // of the free-surface must be computed.
    template <int dim, int n_vars, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_vars, Number>
      source(const Tensor<1, n_vars, Number> &conserved_variables,
             const Tensor<1, dim, Number>    &body_force) const;

    // The next function computes an estimate of the square of the speed from the vector of conserved
    // variables, using the formula $\lambda^2 =  \|\mathbf{u}\|^2+c^2$. The estimate
    // instead of the the true formula is justyfied by efficiency arguments (one evaluation of the square root
    // instead of four). Moroever for low Mach applications, the error committed is very small.
    template <int dim, int n_vars, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      square_speed_estimate(
        const Tensor<1, n_vars, Number> &conserved_variables) const;

    // The next function computes an the square of the gravity wave speed speed:
    template <int dim, int n_vars, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Number
      square_wavespeed(
        const Tensor<1, n_vars, Number> &conserved_variables) const;

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
  {
    prm.enter_subsection("Physical constants");
    g = prm.get_double("g");
    prm.leave_subsection();

    vars_name = {"depth", "momentum", "momentum"};
    postproc_vars_name = {"velocity", "velocity", "pressure"};
  }
  
  template <int dim, int n_vars, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWater::velocity(const Tensor<1, n_vars, Number> &conserved_variables) const
  {
    const Number inverse_depth = Number(1.) / conserved_variables[0];

    Tensor<1, dim, Number> velocity;
    for (unsigned int d = 0; d < dim; ++d)
      velocity[d] = conserved_variables[1 + d] * inverse_depth;

    return velocity;
  }

  template <int dim, int n_vars, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::pressure(const Tensor<1, n_vars, Number> &conserved_variables) const
  {
    return 0.5 * g * conserved_variables[0]*conserved_variables[0];
  }

  template <int dim, int n_vars, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_vars, Tensor<1, dim, Number>>
    ShallowWater::flux(const Tensor<1, n_vars, Number> &conserved_variables) const
  {
    const Tensor<1, dim, Number> v =
      velocity<dim, n_vars>(conserved_variables);

    Tensor<1, n_vars, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      {
        flux[0][d] = conserved_variables[1 + d];
        for (unsigned int e = 0; e < dim; ++e)
          flux[e + 1][d] = conserved_variables[e + 1] * v[d];
      }

    return flux;
  }

  template <int dim, int n_vars, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_vars, Number>
    ShallowWater::source(const Tensor<1, n_vars, Number> &conserved_variables,
                         const Tensor<1, dim, Number>    &body_force) const
  {
    Tensor<1, n_vars, Number> source;
    source[0] = 0.;
    for (unsigned int d = 0; d < dim; ++d)
        source[d + 1] = - g * conserved_variables[0] * body_force[d];

    return source;
  }

  template <int dim, int n_vars, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::square_speed_estimate(
      const Tensor<1, n_vars, Number> &conserved_variables) const
  {
    const auto v = velocity<dim, n_vars>(conserved_variables);

    return v.norm_square() + g * conserved_variables[0];
  }

  template <int dim, int n_vars, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Number
    ShallowWater::square_wavespeed(const Tensor<1, n_vars, Number> &conserved_variables) const
  {
    return g * conserved_variables[0];
  }

} // namespace Model
#endif //SHALLOWWATER_HPP
