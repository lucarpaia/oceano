#include <deal.II/grid/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

// Auxiliary structs for mesh adaptation procedure
//
template<int dim>
struct ScratchData {
  ScratchData(const FiniteElement<dim>& fe,
              const unsigned int        quadrature_degree,
              const UpdateFlags         update_flags): fe_values(fe, QGauss<dim>(quadrature_degree), update_flags) {}

  ScratchData(const ScratchData<dim>& scratch_data): fe_values(scratch_data.fe_values.get_fe(),
                                                               scratch_data.fe_values.get_quadrature(),
                                                               scratch_data.fe_values.get_update_flags()) {}
  FEValues<dim> fe_values;
};


struct CopyData {
  CopyData() : cell_index(numbers::invalid_unsigned_int), value(0.0)  {}

  CopyData(const CopyData &) = default;

  unsigned int cell_index;
  double       value;
};

// @sect{ <code>NavierStokesProjection::refine_mesh</code>}

// After finding a good initial guess on the coarse mesh, we hope to
// decrease the error through refining the mesh. We also need to transfer the current solution to the
// next mesh using the SolutionTransfer class.
//
template <int dim>
void NavierStokesProjection<dim>::refine_mesh() {
  TimerOutput::Scope t(time_table, "Refine mesh");

  /*--- We first create a proper vector for computing estimator ---*/
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler_levset, locally_relevant_dofs);
  LinearAlgebra::distributed::Vector<double> tmp_levset;
  tmp_levset.reinit(dof_handler_levset.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
  tmp_levset = F_n;
  tmp_levset.update_ghost_values();

  using Iterator = typename DoFHandler<dim>::active_cell_iterator;
  Vector<float> estimated_indicator_per_cell(triangulation.n_active_cells());

  /*--- This is basically the indicator per cell computation (see step-50). Since it is not so complciated
        we implement it through a lambda expression ---*/
  auto cell_worker = [&](const Iterator&   cell,
                         ScratchData<dim>& scratch_data,
                         CopyData&         copy_data) {
    FEValues<dim>& fe_values = scratch_data.fe_values; /*--- Here we finally use the 'FEValues' inside ScratchData ---*/
    fe_values.reinit(cell);

    /*--- Compute the gradients for all quadrature points ---*/
    std::vector<Tensor<1, dim>> gradients(fe_values.n_quadrature_points);
    fe_values.get_function_gradients(tmp_levset, gradients);
    copy_data.cell_index = cell->active_cell_index();
    /*-- Criterion is based on the gradient of the level set function ---*/
    double max_gradient_F_norm_square = 0.0;
    for(unsigned k = 0; k < fe_values.n_quadrature_points; ++k) {
      double gradient_F_norm_square = (gradients[k][0]*gradients[k][0] + gradients[k][1]*gradients[k][1]);
      max_gradient_F_norm_square    = std::max(gradient_F_norm_square, max_gradient_F_norm_square);
    }
    copy_data.value = std::sqrt(max_gradient_F_norm_square);
  };

  const UpdateFlags cell_flags = update_gradients | update_quadrature_points | update_JxW_values;

  auto copier = [&](const CopyData &copy_data) {
    if(copy_data.cell_index != numbers::invalid_unsigned_int)
      estimated_indicator_per_cell[copy_data.cell_index] += copy_data.value;
  };

  /*--- Now everything is 'automagically' handled by 'mesh_loop' ---*/
  ScratchData<dim> scratch_data(fe_levset, EquationData::degree_F + 1, cell_flags);
  CopyData copy_data;
  MeshWorker::mesh_loop(dof_handler_levset.begin_active(),
                        dof_handler_levset.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells);

  /*--- Refine grid. In case the refinement level is above a certain value (or the coarsening level is below)
        we clear the flags. ---*/
  GridRefinement::refine(triangulation, estimated_indicator_per_cell, 10.0);
  GridRefinement::coarsen(triangulation, estimated_indicator_per_cell, 5.0);
  for(const auto& cell: triangulation.active_cell_iterators()) {
    if(cell->refine_flag_set() && cell->level() == max_loc_refinements) {
      cell->clear_refine_flag();
    }
    if(cell->coarsen_flag_set() && cell->level() == min_loc_refinements) {
      cell->clear_coarsen_flag();
    }
  }
  triangulation.prepare_coarsening_and_refinement();

  /*--- Now we prepare the object for transfering, basically saving the old quantities using SolutionTransfer.
        Since the 'prepare_for_coarsening_and_refinement' method can be called only once, but we have two vectors
        for dof_handler_velocity, we need to put them in an auxiliary vector. ---*/
  std::vector<const LinearAlgebra::distributed::Vector<double>*> velocities;
  velocities.push_back(&u_n);
  velocities.push_back(&u_n_minus_1);
  parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
  solution_transfer_velocity(dof_handler_velocity);
  solution_transfer_velocity.prepare_for_coarsening_and_refinement(velocities);

  parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
  solution_transfer_pressure(dof_handler_pressure);
  solution_transfer_pressure.prepare_for_coarsening_and_refinement(pres_n);

  parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
  solution_transfer_levset(dof_handler_levset);
  solution_transfer_levset.prepare_for_coarsening_and_refinement(F_n);

  triangulation.execute_coarsening_and_refinement(); /*--- Effectively perform the remeshing ---*/

  /*--- First DoFHandler objects are set up within the new grid ----*/
  setup_dofs();

  /*--- Interpolate current solutions to new mesh. This is done using auxliary vectors just for safety,
        but the new u_n or pres_n could be used. Again, the only point is that the function 'interpolate'
        can be called once and so the vectors related to 'dof_handler_velocity' have to collected in an auxiliary vector. ---*/
  LinearAlgebra::distributed::Vector<double> transfer_velocity,
                                             transfer_velocity_minus_1,
                                             transfer_pressure,
                                             transfer_levset;
  transfer_velocity.reinit(u_n);
  transfer_velocity.zero_out_ghosts();
  transfer_velocity_minus_1.reinit(u_n_minus_1);
  transfer_velocity_minus_1.zero_out_ghosts();

  transfer_pressure.reinit(pres_n);
  transfer_pressure.zero_out_ghosts();

  transfer_levset.reinit(F_n);
  transfer_levset.zero_out_ghosts();

  std::vector<LinearAlgebra::distributed::Vector<double>*> transfer_velocities;
  transfer_velocities.push_back(&transfer_velocity);
  transfer_velocities.push_back(&transfer_velocity_minus_1);
  solution_transfer_velocity.interpolate(transfer_velocities);
  transfer_velocity.update_ghost_values();
  transfer_velocity_minus_1.update_ghost_values();

  solution_transfer_pressure.interpolate(transfer_pressure);
  transfer_pressure.update_ghost_values();

  solution_transfer_levset.interpolate(transfer_levset);
  transfer_levset.update_ghost_values();

  u_n         = transfer_velocity;
  u_n_minus_1 = transfer_velocity_minus_1;

  pres_n      = transfer_pressure;

  F_n         = transfer_levset;
}
