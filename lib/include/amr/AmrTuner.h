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

#include <deal.II/meshworker/mesh_loop.h>
#include <io/TxtDataReader.h>
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
  //
  // The implementation of this class, is not done as
  // in the other namespaces, with base/derived classes or with different
  // classes under the same namespace. Since just one function is changing
  // (the one that estimate the error), we simply switches the different
  // estimate with the preprocessors. The interface of `estimate_error`
  // is very general to deal with very different computation: from
  // evalution of finite element solutions to general data evaluation
  //
  // The error estimates are computed with the `MeshWorker` interface.
  // This is a collection of functions and classes for the mesh loops.
  // The workhorse of this namespace is the `mesh_loop()` function, which
  // implements a completely generic loop over all mesh cells.
  // The loop() function depends on certain objects handed to it as arguments.
  // These objects are of two types, info objects like `ScratchData` and
  // function objects ("workers") that perform the local operations on cells.
  // ScratchData is a helper struct that create and store the finite element
  // class and a quadrature class with it. When the MeshWorker loops over the
  // cells we will have an easy interface to access information for
  // computing local quantities associated to the finite element solution.
  // The computation of the error estimate is done within a lambda function
  // and it is the only part that really changes between the different
  // indicators.
  class AmrTuner
  {
  public:
    AmrTuner(IO::ParameterHandler &prm);
    ~AmrTuner(){};

    double remesh_tick;
    float threshold_refinement;
    float threshold_coarsening;
    float min_mesh_size;
    unsigned int max_level_refinement;
    unsigned int min_level_refinement;
    std::string refinement_filename;

    template <int dim, typename Number>
    void estimate_error(
      const std::vector<FESystem<dim>*>                             &fe,
      const std::vector<DoFHandler<dim>*>                           &dof,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &solution,
      const Function<dim>                                           &data,
      Vector<float>                                                 &error_estimate) const;

  private:
    template<int dim>
    struct ScratchData
    {
      ScratchData(
          const FESystem<dim>      &fe,
          const unsigned int        quadrature_degree,
          const UpdateFlags         update_flags)
        : fe_values(fe, QGauss<dim>(quadrature_degree), update_flags)
      {}

      ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags()) {}

      FEValues<dim> fe_values;
    };

    struct CopyData
    {
      CopyData()
        : cell_index(numbers::invalid_unsigned_int)
        , value(0.0)
      {}

      CopyData(const CopyData &) = default;

      unsigned int cell_index;
      float       value;
    };
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
    min_level_refinement = prm.get_integer("Number_of_refinements");
    min_mesh_size = prm.get_double("Min_mesh_size");
    refinement_filename = prm.get("Static_refinement_indicator_filename");
    prm.leave_subsection();
  }


#if defined AMR_HEIGHTGRADIENT
  // The first error estimator is the simpler one. The gradient of the
  // free-surface (its norm). This is recommended only for surface waves
  // that are freely propagating. It is well suited for tsunamis waves,
  // for example. The cell_worker computes the finite element gradient
  // of the first water height for all quadrature points and assign to
  // each cell the maximum value.
  // The number of quadrature points is equal to the finite element
  // degree; for linear polynomials the gradient is constant and one
  // point it is enough to evaluate it, etc ...
  template <int dim, typename Number>
    void AmrTuner::estimate_error(
      const std::vector<FESystem<dim>*>                             &fe,
      const std::vector<DoFHandler<dim>*>                           &dof,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &solution,
      const Function<dim>                                           &/*data*/,
      Vector<float>                                                 &error_estimate) const
  {
    const unsigned int n_q_points_amr_1d        = fe[0]->degree;
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    auto cell_worker = [&](const Iterator   &cell,
                           ScratchData<dim> &scratch_data,
                           CopyData&         copy_data) {
      FEValues<dim>& fe_values = scratch_data.fe_values;
      fe_values.reinit(cell);

      std::vector<Tensor<1, dim>> gradients(fe_values.n_quadrature_points);
      fe_values.get_function_gradients(solution[0], gradients);

      copy_data.cell_index = cell->active_cell_index();

      float max_gradient = 0.0;
      for(unsigned q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          float gradient_norm_square =
            gradients[q][0]*gradients[q][0] + gradients[q][1]*gradients[q][1];
          max_gradient = std::max(gradient_norm_square, max_gradient);
        }
      copy_data.value = std::sqrt(max_gradient);
    };

    auto copier = [&](const CopyData &copy_data) {
      if(copy_data.cell_index != numbers::invalid_unsigned_int)
        error_estimate[copy_data.cell_index] += copy_data.value;
    };

    const UpdateFlags cell_flags = update_gradients | update_quadrature_points | update_JxW_values;
    ScratchData<dim> scratch_data(*fe[0], n_q_points_amr_1d, cell_flags);
    CopyData copy_data;
    MeshWorker::mesh_loop(dof[0]->begin_active(),
                          dof[0]->end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }

#elif defined AMR_VORTICITY
  // The second error estimator uses the vorticity. This can detect
  // vortices, eddies and fronts.
  // Differently from before we need to evaluate the gradient of a
  // multicomponent solution vector and we store it in a two
  // dimensional vector of Tensors, with the quadrature points
  // along the lines and the dim-components along the columns.
  template <int dim, typename Number>
    void AmrTuner::estimate_error(
      const std::vector<FESystem<dim>*>                             &fe,
      const std::vector<DoFHandler<dim>*>                           &dof,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &solution,
      const Function<dim>                                           &/*data*/,
      Vector<float>                                                 &error_estimate) const
  {
    const unsigned int n_q_points_amr_1d        = fe[1]->degree;
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    auto cell_worker = [&](const Iterator   &cell,
                           ScratchData<dim> &scratch_data,
                           CopyData&         copy_data) {
      FEValues<dim>& fe_values = scratch_data.fe_values;
      fe_values.reinit(cell);

      std::vector<std::vector<Tensor<1, dim>>>
        gradients(fe_values.n_quadrature_points, std::vector<Tensor<1, dim>>(dim));
      fe_values.get_function_gradients(solution[1], gradients);

      copy_data.cell_index = cell->active_cell_index();

      float max_vorticity = 0.0;
      for(unsigned q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          float vorticity =
            std::abs(gradients[q][1][0] - gradients[q][0][1]);
          max_vorticity = std::max(vorticity, max_vorticity);
        }
      copy_data.value = max_vorticity;
    };

    auto copier = [&](const CopyData &copy_data) {
      if(copy_data.cell_index != numbers::invalid_unsigned_int)
        error_estimate[copy_data.cell_index] += copy_data.value;
    };

    const UpdateFlags cell_flags = update_gradients | update_quadrature_points | update_JxW_values;
    ScratchData<dim> scratch_data(*fe[1], n_q_points_amr_1d, cell_flags);
    CopyData copy_data;
    MeshWorker::mesh_loop(dof[1]->begin_active(),
                          dof[1]->end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }

#elif defined AMR_TRACERGRADIENT
  // The third error estimator is the the gradient of a tracer (its norm).
  // This is recommended to improve the resolution near tracer fronts.
  // The cell_worker computes the finite element gradient
  // of the tracer for all quadrature points and assign to
  // each cell the maximum value. In case of multiple tracers, for now,
  // the adaptation criteria is only driven by the first tracer-
  // The number of quadrature points is equal to the finite element
  // degree; for linear polynomials the gradient is constant and one
  // point it is enough to evaluate it, etc .... Please note that since the
  // tracers and the free-surface are co-located we can use the same dof
  // structure.
  template <int dim, typename Number>
    void AmrTuner::estimate_error(
      const std::vector<FESystem<dim>*>                             &fe,
      const std::vector<DoFHandler<dim>*>                           &dof,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &solution,
      const Function<dim>                                           &/*data*/,
      Vector<float>                                                 &error_estimate) const
  {
    const unsigned int n_q_points_amr_1d        = fe[0]->degree;
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    auto cell_worker = [&](const Iterator   &cell,
                           ScratchData<dim> &scratch_data,
                           CopyData&         copy_data) {
      FEValues<dim>& fe_values = scratch_data.fe_values;
      fe_values.reinit(cell);

      std::vector<Tensor<1, dim>> gradients(fe_values.n_quadrature_points);
      fe_values.get_function_gradients(solution[2], gradients);

      copy_data.cell_index = cell->active_cell_index();

      float max_gradient = 0.0;
      for(unsigned q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          float gradient_norm_square =
            gradients[q][0]*gradients[q][0] + gradients[q][1]*gradients[q][1];
          max_gradient = std::max(gradient_norm_square, max_gradient);
        }
      copy_data.value = std::sqrt(max_gradient);
    };

    auto copier = [&](const CopyData &copy_data) {
      if(copy_data.cell_index != numbers::invalid_unsigned_int)
        error_estimate[copy_data.cell_index] += copy_data.value;
    };

    const UpdateFlags cell_flags = update_gradients | update_quadrature_points | update_JxW_values;
    ScratchData<dim> scratch_data(*fe[0], n_q_points_amr_1d, cell_flags);
    CopyData copy_data;
    MeshWorker::mesh_loop(dof[0]->begin_active(),
                          dof[0]->end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }

#elif defined AMR_BATHYMETRY
  // The fourth error estimator is more a refinement indicator.
  // It can be used for static mesh refinement as it uses the
  // bathymetry to increase the resolution in coastal regions, near
  // the shoreline or along channels.
  // As you may note the number of quadrature points is not related
  // to the finite element solution as we are evaluating an external
  // field not projected on a finite dimensional space. The chosen
  // number is quite high (10 points) because of the fact that one
  // may start a very coarse mesh and shallow channels may not be
  // detected at all with a low order quadrature formula. The extra
  // cost of evaluating the bathymetry on a large number of points
  // is taken only during the initial refinement stage and we expect
  // to not impact much the final computational time.
  template <int dim, typename Number>
    void AmrTuner::estimate_error(
      const std::vector<FESystem<dim>*>                             &fe,
      const std::vector<DoFHandler<dim>*>                           &dof,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &/*solution*/,
      const Function<dim>                                           &data,
      Vector<float>                                                 &error_estimate) const
  {
    const unsigned int n_q_points_amr_1d        = 10;
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    auto cell_worker = [&](const Iterator   &cell,
                           ScratchData<dim> &scratch_data,
                           CopyData&         copy_data) {
      FEValues<dim>& fe_values = scratch_data.fe_values;
      fe_values.reinit(cell);

      copy_data.cell_index = cell->active_cell_index();

      double max_bathy = 0.0;
      for(unsigned q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          const Number zb_q = data.value(fe_values.quadrature_point(q), 0);
          max_bathy = std::max(std::abs(zb_q), max_bathy);
        }
      copy_data.value = max_bathy;
    };

    auto copier = [&](const CopyData &copy_data) {
      if(copy_data.cell_index != numbers::invalid_unsigned_int)
        error_estimate[copy_data.cell_index] += copy_data.value;
    };

    const UpdateFlags cell_flags = update_quadrature_points;
    ScratchData<dim> scratch_data(*fe[0], n_q_points_amr_1d, cell_flags);
    CopyData copy_data;
    MeshWorker::mesh_loop(dof[0]->begin_active(),
                          dof[0]->end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }

#elif defined AMR_FROMFILE
  // This is also a refinement indicator as it is not related
  // to the finit element solution but it is read directly from
  // file. It can be used, as before, for static mesh refinement.
  // but, with to respect to the latter, it gives more freedom.
  // You could think to create a file with a higher error around
  // sharp bathymetry gradients or to refine in a certain
  // masked area. Pleae note that the InterpolatedUniformGridData
  // works with double and there is a hiddend conversion of the
  // imported data to float.
  // As before we use a large number of quadrature points
  // to be sure to detect all features, even starting from a
  // very coarse mesh.
  template <int dim, typename Number>
    void AmrTuner::estimate_error(
      const std::vector<FESystem<dim>*>                             &fe,
      const std::vector<DoFHandler<dim>*>                           &dof,
      const std::vector<LinearAlgebra::distributed::Vector<Number>> &/*solution*/,
      const Function<dim>                                           &/*data*/,
      Vector<float>                                                 &error_estimate) const
  {
    const unsigned int n_q_points_amr_1d        = 10;
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    IO::TxtDataReader<dim> data_reader(refinement_filename);
    const Functions::InterpolatedUniformGridData<dim> data(
      data_reader.endpoints,
      data_reader.n_intervals,
      Table<dim, double>(data_reader.n_intervals.front()+1,
                         data_reader.n_intervals.back()+1,
                         data_reader.get_data(data_reader.filename).begin()));

    auto cell_worker = [&](const Iterator   &cell,
                           ScratchData<dim> &scratch_data,
                           CopyData&         copy_data) {
      FEValues<dim>& fe_values = scratch_data.fe_values;
      fe_values.reinit(cell);

      copy_data.cell_index = cell->active_cell_index();

      double max_error = 0.0;
      for(unsigned q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          const Number err_q = data.value(fe_values.quadrature_point(q));
          max_error= std::max(std::abs(err_q), max_error);
        }
      copy_data.value = max_error;
    };

    auto copier = [&](const CopyData &copy_data) {
      if(copy_data.cell_index != numbers::invalid_unsigned_int)
        error_estimate[copy_data.cell_index] += copy_data.value;
    };

    const UpdateFlags cell_flags = update_quadrature_points;
    ScratchData<dim> scratch_data(*fe[0], n_q_points_amr_1d, cell_flags);
    CopyData copy_data;
    MeshWorker::mesh_loop(dof[0]->begin_active(),
                          dof[0]->end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells);
  }
#endif

} // namespace Amr
#endif //AMR_HPP
