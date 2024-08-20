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
  //
  // Derivative of the solutions are computed with the class `DerivativeApproximation`.
  // Ghost indices made available in the vector are a tight set of only those
  // indices that are requested by the the function that compute the approximate
  // derivatives. Consequently the solution vector as-is is not suitable for the
  // `DerivativeApproximation` class. The trick is to change the ghost part of the
  // partition, for example using a temporary vector and
  // `LinearAlgebra::distributed::Vector::copy_locally_owned_data_from()` as
  // shown below.
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
      const parallel::distributed::Triangulation<dim>               &triangulation,
      const Mapping<dim>                                            &mapping,
      const std::vector<DoFHandler<dim> *>                          &dof,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &solution,
      Vector<float>                                                 &error_estimate) const;
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
  // free-surface (its norm). This is recommended only for surface waves
  // that are freely propagating. It is well suited for tsunamis waves,
  // for example.
  template <int dim, typename Number>
    void AmrTuner::estimate_error(
      const parallel::distributed::Triangulation<dim>               &triangulation,
      const Mapping<dim>                                            &mapping,
      const std::vector<DoFHandler<dim> *>                          &dof,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &solution,
      Vector<float>                                                 &error_estimate) const
  {
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(*dof[0]);
    LinearAlgebra::distributed::Vector<Number> copy_vec(solution[0]);
    copy_vec.reinit(dof[0]->locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    copy_vec.copy_locally_owned_data_from(solution[0]);
    copy_vec.update_ghost_values();

    DerivativeApproximation::approximate_gradient(mapping,
                                                  *dof[0],
                                                  copy_vec,
                                                  error_estimate);
  }
#elif defined AMR_VORTICITY
  // The second error estimator uses the vorticity. This can detect
  // vortices, eddies and fronts.
  template <int dim, typename Number>
    void AmrTuner::estimate_error(
      const parallel::distributed::Triangulation<dim>               &triangulation,
      const Mapping<dim>                                            &mapping,
      const std::vector<DoFHandler<dim> *>                          &dof,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &solution,
      Vector<float>                                                 &error_estimate) const
  {
    Tensor<1,dim> du;
    Tensor<1,dim> dv;

    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(*dof[1]);
    LinearAlgebra::distributed::Vector<Number> copy_vec(solution[1]);
    copy_vec.reinit(dof[1]->locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    copy_vec.copy_locally_owned_data_from(solution[1]);
    copy_vec.update_ghost_values();

    for (const auto &cell : dof[1]->active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            DerivativeApproximation::approximate_derivative_tensor(mapping,
                                                                   *dof[1],
                                                                   copy_vec,
                                                                   cell,
                                                                   du,
                                                                   0);
            DerivativeApproximation::approximate_derivative_tensor(mapping,
                                                                   *dof[1],
                                                                   copy_vec,
                                                                   cell,
                                                                   dv,
                                                                   1);
          }
        error_estimate(cell->active_cell_index()) = std::abs(dv[0]-du[1]);
      }
  }
#endif

} // namespace Amr
#endif //AMR_HPP
