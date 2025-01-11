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
#ifndef SHALLOWWATERWITHTRACER_HPP
#define SHALLOWWATERWITHTRACER_HPP

// The following files include the oceano libraries
#include <model/ShallowWater.h>

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
  class ShallowWaterWithTracer : public ShallowWater
  {
  public:
    ShallowWaterWithTracer(IO::ParameterHandler &prm);
    ~ShallowWaterWithTracer(){};

    template <int dim, int n_tra>
    inline DEAL_II_ALWAYS_INLINE //
      void
      set_vars_name();

    template <int dim, int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Tensor<1, dim, Number>>
      tracerflux(const Tensor<1, dim, Number>   &discharge,
                 const Tensor<1, n_tra, Number> &tracer) const;

    template <int dim, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, dim, Number>
      tracerflux(const Tensor<1, dim, Number> &discharge,
                 const Number                  tracer) const;
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
  ShallowWaterWithTracer::ShallowWaterWithTracer(
    IO::ParameterHandler &prm)
    : ShallowWater(prm)
  {}

  template <int dim, int n_tra>
  inline DEAL_II_ALWAYS_INLINE //
    void
    ShallowWaterWithTracer::set_vars_name()
  {
    vars_name.push_back("free_surface");
    for (unsigned int d = 0; d < dim; ++d)
      vars_name.push_back("hu");
    for (unsigned int t = 0; t < n_tra; ++t)
        vars_name.push_back("t_"+std::to_string(t+1));

    for (unsigned int d = 0; d < dim; ++d)
      postproc_vars_name.push_back("velocity");
    postproc_vars_name.push_back("depth");
  }
 
  template <int dim, int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Tensor<1, dim, Number>>
    ShallowWaterWithTracer::tracerflux(
      const Tensor<1, dim, Number>   &discharge,
      const Tensor<1, n_tra, Number> &tracer) const
  {
    Tensor<1, n_tra, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < n_tra; ++e)
        flux[e][d] = tracer[e] * discharge[d];

    return flux;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, dim, Number>
    ShallowWaterWithTracer::tracerflux(
      const Tensor<1, dim, Number> &discharge,
      const Number                  tracer) const
  {
    return tracer * discharge;
  }
} // namespace Model
#endif //SHALLOWWATERWITHTRACER_HPP
