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
#ifndef AMR_HPP
#define AMR_HPP

#include <deal.II/numerics/derivative_approximation.h>
/**
 * Namespace containing the AMR options.
 */

namespace Amr
{

  using namespace dealii;

  // @sect3{Adaptive Mesh Refinement}

  // In the following class, we implement a fine tuning of
  // the Adaptive Mesh Refinement algorithm already available
  // in Deal.ii. The tuning consist of:
  // \begin{itemize}
  // \item mainly we have the function  for the error estimate
  // that is case-dependent. You would like to refine around the free-surface
  // gradients to detect impulsive waves or to strong vorticity patterns.
  // \item the tuning parameters that are read from the parameter file
  // and then are stored into as classe members.
  // \end{itemize}
  // The implementation of this class, is not done as
  // in the other namespaces, with base/derived classes or with different
  // classes under the same namespace. Since just one estimate function
  // is changing, we simply switches the different estimate with the
  // preprocessors.
  class AmrTuner
  {
  public:
    AmrTuner(IO::ParameterHandler &prm);
    ~AmrTuner(){};

    double remesh_tick;
    float threshold_refinement;
    float threshold_coarsening;
    unsigned int max_level_refinement;
    std::string output_filename;

    // The next function is the one that computes the error estimate.
    // It is overloaded by the same function defined in the derived classes.
    template <int dim, typename Number>
    void estimate_error(
      const Mapping<dim>                               &mapping,
      const DoFHandler<dim>                            &dof,
      const LinearAlgebra::distributed::Vector<Number> &solution,
      Vector<float>                                    &error_estimate) const;
  };
    
  // The constructor of the class takes as arguments 
  // only the parameters handler class in order to read the AMR
  // parameters that are strongly case dependent and needs some
  // tuning to obtain the desired meshes.
  AmrTuner::AmrTuner(
    IO::ParameterHandler &prm)
  {
    prm.enter_subsection("Mesh & geometry parameters");
    remesh_tick = prm.get_double("Remesh_tick");
    threshold_refinement = prm.get_double("Threshold_for_refinement");
    threshold_coarsening = prm.get_double("Threshold_for_coarsening");
    max_level_refinement = prm.get_integer("Max_level_of_refinement");
    output_filename = prm.get("Error_filename");
    prm.leave_subsection();
  }
  
  
#if defined AMR_HEIGHTGRADIENT
  // The first error estimator is the simpler one. The gradient of the
  // free-surface. This is recommended only for surface waves
  // that are freely propagating. It is well suited for tsunamis waves,
  // for example.
  template <int dim, typename Number>
    void AmrTuner::estimate_error(
      const Mapping<dim>                               &mapping,
      const DoFHandler<dim>                            &dof,
      const LinearAlgebra::distributed::Vector<Number> &solution,
      Vector<float>                                    &error_estimate) const
  {
    DerivativeApproximation::approximate_gradient(mapping,
                                                  dof,
                                                  solution,
                                                  error_estimate);
  }
#endif

} // namespace Amr
#endif //AMR_HPP
