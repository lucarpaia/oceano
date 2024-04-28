/* Author: Giuseppe Orlando, 2024. */

// @sect{Include files}

// We start by including all the necessary deal.II header files
//
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/timer.h>

#include <deal.II/fe/mapping_q.h>

#include "SW_operator.h"

/*--- Include headers related to multigrid ---*/
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

using namespace SW;

// @sect{The <code>SWSolver</code> class}

// Now for the main class of the program. It implements the solver for the
// Euler equations using the discretization previously implemented.
//
template<int dim>
class SWSolver {
public:
  SWSolver(RunTimeParameters::Data_Storage& data); /*--- Class constructor ---*/

  void run(const bool verbose = false, const unsigned int output_interval = 10);
  /*--- The run function which actually runs the simulation ---*/

protected:
  const double t0;         /*--- Initial time auxiliary variable ----*/
  const double T;          /*--- Final time auxiliary variable ----*/
  unsigned int IMEX_stage; /*--- Flag to check at which current stage of the IMEX we are ---*/
  double       dt;         /*--- Time step auxiliary variable ---*/

  parallel::distributed::Triangulation<dim> triangulation; /*--- The variable which stores the mesh ---*/

  /*--- Finite element spaces for all the variables ---*/
  FESystem<dim> fe_height;
  FESystem<dim> fe_discharge;
  FESystem<dim> fe_tracer;

  /*--- Degrees of freedom handlers for all the variables ---*/
  DoFHandler<dim> dof_handler_height;
  DoFHandler<dim> dof_handler_discharge;
  DoFHandler<dim> dof_handler_tracer;

  /*--- Auxiliary mapping for possible curved boundary ---*/
  MappingQ1<dim> mapping;
  MappingQ1<dim> mapping_mg; /*--- Auxiliary mapping for multigrid for the sake of generality ---*/
  std::map<types::global_dof_index, Point<dim>> dof_location_map; /*--- Map between global dof index and real space location ---*/

  /*--- Auxiliary quadratures for the variables for which we can keep track ---*/
  QGaussLobatto<dim> quadrature_height;
  QGaussLobatto<dim> quadrature_discharge;
  QGaussLobatto<dim> quadrature_tracer;

  /*--- Variables for the height ---*/
  LinearAlgebra::distributed::Vector<double> h_old;
  LinearAlgebra::distributed::Vector<double> h_tmp_2;
  LinearAlgebra::distributed::Vector<double> h_tmp_3;
  LinearAlgebra::distributed::Vector<double> h_curr;
  LinearAlgebra::distributed::Vector<double> rhs_h;

  LinearAlgebra::distributed::Vector<double> zeta_old;
  LinearAlgebra::distributed::Vector<double> zeta_tmp_2;
  LinearAlgebra::distributed::Vector<double> zeta_tmp_3;

  /*--- Variables for the discharge ---*/
  LinearAlgebra::distributed::Vector<double> hu_old;
  LinearAlgebra::distributed::Vector<double> hu_tmp_2;
  LinearAlgebra::distributed::Vector<double> hu_tmp_3;
  LinearAlgebra::distributed::Vector<double> hu_curr;
  LinearAlgebra::distributed::Vector<double> rhs_hu;

  /*--- Variables for the tracer ---*/
  LinearAlgebra::distributed::Vector<double> hc_old;
  LinearAlgebra::distributed::Vector<double> hc_tmp_2;
  LinearAlgebra::distributed::Vector<double> hc_tmp_3;
  LinearAlgebra::distributed::Vector<double> rhs_hc;

  DeclException2(ExcInvalidTimeStep,
                 double,
                 double,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  void create_triangulation(const unsigned int n_refines); /*--- Function to create the grid ---*/

  void setup_dofs(); /*--- Function to set the dofs ---*/

  void initialize(); /*--- Function to initialize the fields ---*/

  void update_height(); /*--- Function to update the height ---*/

  void update_discharge(); /*--- Function to update the discharge ---*/

  void update_tracer(); /*--- Function to update the tracer ---*/

  void output_results(const unsigned int step); /*--- Function to save the results ---*/

  void analyze_results(); /*--- Function to compute the errors (we have an analytical solutino here) ---*/

private:
  /*--- Function to set the initial conditions ---*/
  EquationData::Height<dim>             h_exact;
  EquationData::Discharge<dim>          hu_exact;
  EquationData::Tracer<dim>             hc_exact;
  EquationData::Bathymetry<dim, double> zb_exact;

  /*--- Auxiliary structures for the matrix-free ---*/
  std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

  SWOperator<dim, EquationData::n_stages,
             EquationData::degree_h, EquationData::degree_hu, EquationData::degree_hc,
             2*EquationData::degree_h + 1 + EquationData::extra_quadrature_degree,
             2*EquationData::degree_hu + 1 + EquationData::extra_quadrature_degree,
             2*EquationData::degree_hc + 1 + EquationData::extra_quadrature_degree,
             LinearAlgebra::distributed::Vector<double>> SW_matrix;

  MGLevelObject<SWOperator<dim, EquationData::n_stages,
                           EquationData::degree_h, EquationData::degree_hu, EquationData::degree_hc,
                           2*EquationData::degree_h + 1 + EquationData::extra_quadrature_degree,
                           2*EquationData::degree_hu + 1 + EquationData::extra_quadrature_degree,
                           2*EquationData::degree_hc + 1 + EquationData::extra_quadrature_degree,
                           LinearAlgebra::distributed::Vector<float>>> mg_matrices_SW;

  std::vector<const DoFHandler<dim>*> dof_handlers; /*--- Auxiliary container for the matrix-free ---*/

  std::vector<const AffineConstraints<double>*> constraints; /*--- Auxiliary container for the matrix-free ---*/
  AffineConstraints<double> constraints_height,
                            constraints_discharge,
                            constraints_tracer;

  std::vector<QGauss<1>> quadratures; /*--- Auxiliary container for the quadrature in matrix-free ---*/

  unsigned int max_its; /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
  double       eps;     /*--- Auxiliary variable for the tolerance of linear solvers ---*/

  unsigned int n_refines; /*-- Number of initial global refinements ---*/

  std::string saving_dir; /*--- Auxiliary variable for the directory to save the results ---*/

  /*--- Now we declare a bunch of variables for text output ---*/
  ConditionalOStream pcout;

  std::ofstream      time_out;
  ConditionalOStream ptime_out;
  TimerOutput        time_table;

  std::ofstream output_error_height;
  std::ofstream output_error_discharge;
  std::ofstream output_error_tracer;

  Vector<double> L1_error_per_cell_h,
                 Linfty_error_per_cell_h,
                 L1_error_per_cell_hu,
                 Linfty_error_per_cell_hu,
                 L1_error_per_cell_hc,
                 Linfty_error_per_cell_hc;  /*--- Auxiliary variables to compute the errors ---*/

  /*MGLevelObject<LinearAlgebra::distributed::Vector<float>> level_projection_h,
                                                           level_projection_hu;*/ /*--- Auxiliary variables for multigrid purposes ---*/

  double get_min_height(); /*--- Get minimum height ---*/

  double get_max_height(); /*--- Get maximum height ---*/

  std::tuple<double, double, double> compute_max_C_x_y(); /*--- Get maximum Courant numbers along x and z ---*/
};


// @sect{ <code>SWSolver::SWSolver</code> }

// In the constructor, we just read all the data from the
// <code>Data_Storage</code> object that is passed as an argument, verify that
// the data we read are reasonable and, finally, create the triangulation and
// load the initial data.
//
template<int dim>
SWSolver<dim>::SWSolver(RunTimeParameters::Data_Storage& data):
  t0(data.initial_time),
  T(data.final_time),
  IMEX_stage(1),             /*--- Initialize the flag for the IMEX scheme stage ---*/
  dt(data.dt),
  triangulation(MPI_COMM_WORLD,
                parallel::distributed::Triangulation<dim>::limit_level_difference_at_vertices,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe_height(FE_DGQ<dim>(EquationData::degree_h), 1),
  fe_discharge(FE_DGQ<dim>(EquationData::degree_hu), dim),
  fe_tracer(FE_DGQ<dim>(EquationData::degree_hc), 1),
  dof_handler_height(triangulation),
  dof_handler_discharge(triangulation),
  dof_handler_tracer(triangulation),
  mapping(),
  mapping_mg(),
  quadrature_height(EquationData::degree_h + 1),
  quadrature_discharge(EquationData::degree_hu + 1),
  quadrature_tracer(EquationData::degree_hc + 1),
  h_exact(data.initial_time),
  hu_exact(data.initial_time),
  hc_exact(data.initial_time),
  zb_exact(data.initial_time),
  SW_matrix(data),
  max_its(data.max_iterations),
  eps(data.eps),
  n_refines(data.n_global_refines),
  saving_dir(data.dir),
  pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_out("./" + data.dir + "/time_analysis_" +
           Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
  ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
  output_error_height("./" + data.dir + "/error_analysis_h.dat", std::ofstream::out),
  output_error_discharge("./" + data.dir + "/error_analysis_hu.dat", std::ofstream::out),
  output_error_tracer("./" + data.dir + "/error_analysis_hc.dat", std::ofstream::out) {
    AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

    matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

    dof_handlers.clear();

    constraints.clear();
    constraints_height.clear();
    constraints_discharge.clear();
    constraints_tracer.clear();

    quadratures.clear();

    create_triangulation(n_refines);
    setup_dofs();
    initialize();
}


// @sect{<code>SWSolver::create_triangulation_and_dofs</code>}

// The method that creates the triangulation.
//
template<int dim>
void SWSolver<dim>::create_triangulation(const unsigned int n_refines) {
  TimerOutput::Scope t(time_table, "Create triangulation");

  Point<dim> lower_left;
  lower_left[0] = 0.0;
  lower_left[1] = 0.0;
  Point<dim> upper_right;
  upper_right[0] = 10.0;
  upper_right[1] = 10.0;

  GridGenerator::subdivided_hyper_rectangle(triangulation, {16, 16}, lower_left, upper_right, true);

  /*--- Consider periodic conditions along the horizontal direction ---*/
  std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>> periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
  GridTools::collect_periodic_faces(triangulation, 2, 3, 1, periodic_faces);
  triangulation.add_periodicity(periodic_faces);

  triangulation.refine_global(n_refines);
}


// After creating the triangulation, it creates the mesh dependent
// data, i.e. it distributes degrees of freedom, and
// initializes the matrices and vectors that we will use.
//
template<int dim>
void SWSolver<dim>::setup_dofs() {
  TimerOutput::Scope t(time_table, "Setup dofs");

  pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

  /*--- Set degrees of freedom ---*/
  dof_handler_height.distribute_dofs(fe_height);
  dof_handler_discharge.distribute_dofs(fe_discharge);
  dof_handler_tracer.distribute_dofs(fe_tracer);

  pcout << "dim (space height) = " << dof_handler_height.n_dofs()
        << std::endl
        << "dim (space discharge) = " << dof_handler_discharge.n_dofs()
        << std::endl
        << "dim (space tracer) = " << dof_handler_tracer.n_dofs()
        << std::endl
        << std::endl;

  /*--- Set additional data to check which variables neeed to be updated ---*/
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags                = (update_values | update_gradients |
                                                         update_JxW_values | update_quadrature_points);
  additional_data.mapping_update_flags_inner_faces    = (update_values | update_JxW_values |
                                                         update_quadrature_points | update_normal_vectors);
  additional_data.mapping_update_flags_boundary_faces = update_default;
  additional_data.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;

  /*--- Set the container with the dof handlers ---*/
  dof_handlers.push_back(&dof_handler_height);
  dof_handlers.push_back(&dof_handler_discharge);
  dof_handlers.push_back(&dof_handler_tracer);

  /*--- Set the container with the constraints. Each entry is empty (no Dirichlet and weak imposition in general)
        and this is necessary only for compatibilty reasons ---*/
  constraints.push_back(&constraints_height);
  constraints.push_back(&constraints_discharge);
  constraints.push_back(&constraints_tracer);

  /*--- Set the quadrature formula to compute the integrals for assembling bilinear and linear forms ---*/
  quadratures.push_back(QGauss<1>(2*EquationData::degree_hu + 1 + EquationData::extra_quadrature_degree));

  /*--- Initialize the matrix-free structure with DofHandlers, Constraints, Quadratures and AdditionalData ---*/
  matrix_free_storage->reinit(mapping, dof_handlers, constraints, quadratures, additional_data);

  /*--- Initialize the variables related to the height ---*/
  matrix_free_storage->initialize_dof_vector(h_old, 0);
  matrix_free_storage->initialize_dof_vector(h_tmp_2, 0);
  matrix_free_storage->initialize_dof_vector(h_tmp_3, 0);
  matrix_free_storage->initialize_dof_vector(h_curr, 0);
  matrix_free_storage->initialize_dof_vector(rhs_h, 0);

  matrix_free_storage->initialize_dof_vector(zeta_old, 0);
  matrix_free_storage->initialize_dof_vector(zeta_tmp_2, 0);
  matrix_free_storage->initialize_dof_vector(zeta_tmp_3, 0);

  /*--- Initialize the variables related to the discharge ---*/
  matrix_free_storage->initialize_dof_vector(hu_old, 1);
  matrix_free_storage->initialize_dof_vector(hu_tmp_2, 1);
  matrix_free_storage->initialize_dof_vector(hu_tmp_3, 1);
  matrix_free_storage->initialize_dof_vector(hu_curr, 1);
  matrix_free_storage->initialize_dof_vector(rhs_hu, 1);

  /*--- Initialize the variables related to the tracer ---*/
  matrix_free_storage->initialize_dof_vector(hc_old, 2);
  matrix_free_storage->initialize_dof_vector(hc_tmp_2, 2);
  matrix_free_storage->initialize_dof_vector(hc_tmp_3, 2);
  matrix_free_storage->initialize_dof_vector(rhs_hc, 2);

  /*--- Initialize the auxiliary variables to check the errors ---*/
  Vector<double> error_per_cell_tmp(triangulation.n_active_cells());
  L1_error_per_cell_h.reinit(error_per_cell_tmp);
  Linfty_error_per_cell_h.reinit(error_per_cell_tmp);
  L1_error_per_cell_hu.reinit(error_per_cell_tmp);
  Linfty_error_per_cell_hu.reinit(error_per_cell_tmp);
  L1_error_per_cell_hc.reinit(error_per_cell_tmp);
  Linfty_error_per_cell_hc.reinit(error_per_cell_tmp);

  /*--- Initialize the multigrid structure ---*/
  mg_matrices_SW.clear_elements();
  dof_handler_height.distribute_mg_dofs();
  dof_handler_discharge.distribute_mg_dofs();
  dof_handler_tracer.distribute_mg_dofs();

  /*level_projection_h = MGLevelObject<LinearAlgebra::distributed::Vector<float>>(0, triangulation.n_global_levels() - 1);
  level_projection_hu   = MGLevelObject<LinearAlgebra::distributed::Vector<float>>(0, triangulation.n_global_levels() - 1);*/
  mg_matrices_SW.resize(0, triangulation.n_global_levels() - 1);
  for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
    /*typename MatrixFree<dim, float>::AdditionalData additional_data_mg;
    additional_data_mg.mg_level = level;

    std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
    mg_mf_storage_level->reinit(mapping_mg, dof_handlers, constraints, quadratures, additional_data_mg);
    mg_mf_storage_level->initialize_dof_vector(level_projection_h[level], 0);
    mg_mf_storage_level->initialize_dof_vector(level_projection_hu[level], 1);*/

    mg_matrices_SW[level].set_dt(dt);
  }

  /*--- Set the map between dofs indices and theri real space location ---*/
  dof_location_map = DoFTools::map_dofs_to_support_points(mapping, dof_handler_height);
}


// @sect{ <code>SWSolver::initialize</code> }

// This method loads the initial data
//
template<int dim>
void SWSolver<dim>::initialize() {
  TimerOutput::Scope t(time_table, "Initialize state");

  VectorTools::interpolate(mapping, dof_handler_height, h_exact, h_old);
  VectorTools::interpolate(mapping, dof_handler_discharge, hu_exact, hu_old);
  VectorTools::interpolate(mapping, dof_handler_tracer, hc_exact, hc_old);

  /*VectorTools::project(mapping, dof_handler_height, constraints_height, QGauss<dim>(EquationData::degree_h + 1), h_exact, h_old);
  VectorTools::project(mapping, dof_handler_discharge, constraints_discharge, QGauss<dim>(EquationData::degree_hu + 1), hu_exact, hu_old);
  VectorTools::project(mapping, dof_handler_tracer, constraints_tracer, QGauss<dim>(EquationData::degree_hc + 1), hc_exact, hc_old);*/

  /*--- Create the auxiliary variable for the total height ---*/
  zeta_old.equ(1.0, h_old);
  for(const auto& cell: dof_handler_height.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      std::vector<types::global_dof_index> dof_indices(fe_height.dofs_per_cell);
      cell->get_dof_indices(dof_indices);
      for(unsigned int idx = 0; idx < dof_indices.size(); ++idx) {
        zeta_old(dof_indices[idx]) -= zb_exact.value(dof_location_map[dof_indices[idx]]);
      }
    }
  }
}


// @sect{<code>SWSolver::update_height</code>}

// This implements the update of the height
//
template<int dim>
void SWSolver<dim>::update_height() {
  TimerOutput::Scope t(time_table, "Update height");

  const std::vector<unsigned int> tmp = {0};
  SW_matrix.initialize(matrix_free_storage, tmp, tmp);

  if(IMEX_stage == 2) {
    SW_matrix.set_SW_stage(1);
    SW_matrix.vmult_rhs_h(rhs_h, {h_old, hu_old});
  }
  else if(IMEX_stage == 3) {
    SW_matrix.set_SW_stage(1);
    SW_matrix.vmult_rhs_h(rhs_h, {h_old, hu_old,
                                  h_tmp_2, hu_tmp_2});
  }
  else {
    SW_matrix.set_SW_stage(4);
    SW_matrix.vmult_rhs_h(rhs_h, {h_old, hu_old,
                                  h_tmp_2, hu_tmp_2,
                                  h_tmp_3, hu_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_h.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  /*--- Compute multigrid preconditioner for the height equation ---*/
  MGTransferMatrixFree<dim, float> mg_transfer;
  mg_transfer.build(dof_handler_height);
  using SmootherType = PreconditionChebyshev<SWOperator<dim, EquationData::n_stages,
                                                        EquationData::degree_h,
                                                        EquationData::degree_hu,
                                                        EquationData::degree_hc,
                                                        2*EquationData::degree_h + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_hu + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_hc + 1 + EquationData::extra_quadrature_degree,
                                                        LinearAlgebra::distributed::Vector<float>>,
                                             LinearAlgebra::distributed::Vector<float>>;
  mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, triangulation.n_global_levels() - 1);
  for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
    typename MatrixFree<dim, float>::AdditionalData additional_data_mg;
    additional_data_mg.tasks_parallel_scheme               = MatrixFree<dim, float>::AdditionalData::none;
    additional_data_mg.mapping_update_flags                = (update_values | update_quadrature_points | update_JxW_values);
    additional_data_mg.mapping_update_flags_inner_faces    = update_default;
    additional_data_mg.mapping_update_flags_boundary_faces = update_default;
    additional_data_mg.mg_level                            = level;

    std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
    mg_mf_storage_level->reinit(mapping_mg, dof_handlers, constraints, quadratures, additional_data_mg);
    mg_matrices_SW[level].initialize(mg_mf_storage_level, tmp, tmp);
    if(IMEX_stage == EquationData::n_stages + 1) {
      mg_matrices_SW[level].set_SW_stage(4);
    }
    else {
      mg_matrices_SW[level].set_SW_stage(1);
    }

    if(level > 0) {
      smoother_data[level].smoothing_range     = 15.0;
      smoother_data[level].degree              = 3;
      smoother_data[level].eig_cg_n_iterations = 10;
    }
    else {
      smoother_data[0].smoothing_range     = 2e-2;
      smoother_data[0].degree              = numbers::invalid_unsigned_int;
      smoother_data[0].eig_cg_n_iterations = mg_matrices_SW[0].m();
    }
    mg_matrices_SW[level].compute_diagonal();
    smoother_data[level].preconditioner = mg_matrices_SW[level].get_matrix_diagonal_inverse();
  }
  mg_smoother.initialize(mg_matrices_SW, smoother_data);

  MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
  mg_coarse.initialize(mg_smoother);
  mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices_SW);
  Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  PreconditionMG<dim,
                 LinearAlgebra::distributed::Vector<float>,
                 MGTransferMatrixFree<dim, float>> preconditioner(dof_handler_height, mg, mg_transfer);

  /*--- Solve the linear system for the height ---*/
  if(IMEX_stage == 2) {
    h_tmp_2.equ(1.0, h_old);
    cg.solve(SW_matrix, h_tmp_2, rhs_h, preconditioner);

    /*--- Update the total height ---*/
    zeta_tmp_2.equ(1.0, h_tmp_2);
    for(const auto& cell: dof_handler_height.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        std::vector<types::global_dof_index> dof_indices(fe_height.dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        for(unsigned int idx = 0; idx < dof_indices.size(); ++idx) {
          zeta_tmp_2(dof_indices[idx]) -= zb_exact.value(dof_location_map[dof_indices[idx]]);
        }
      }
    }
  }
  else if(IMEX_stage == 3) {
    h_tmp_3.equ(1.0, h_tmp_2);
    cg.solve(SW_matrix, h_tmp_3, rhs_h, preconditioner);

    /*--- Update the total height ---*/
    zeta_tmp_3.equ(1.0, h_tmp_3);
    for(const auto& cell: dof_handler_height.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        std::vector<types::global_dof_index> dof_indices(fe_height.dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        for(unsigned int idx = 0; idx < dof_indices.size(); ++idx) {
          zeta_tmp_3(dof_indices[idx]) -= zb_exact.value(dof_location_map[dof_indices[idx]]);
        }
      }
    }
  }
  else {
    h_curr.equ(1.0, h_tmp_3);
    cg.solve(SW_matrix, h_curr, rhs_h, preconditioner);
  }
}


// @sect{<code>SWSolver::update_discharge</code>}

// This implements the update of the discharge
//
template<int dim>
void SWSolver<dim>::update_discharge() {
  TimerOutput::Scope t(time_table, "Update discharge");

  const std::vector<unsigned int> tmp = {1};
  SW_matrix.initialize(matrix_free_storage, tmp, tmp);

  if(IMEX_stage == 2) {
    SW_matrix.set_SW_stage(2);
    SW_matrix.vmult_rhs_hu(rhs_hu, {h_old, hu_old, zeta_old});
  }
  else if(IMEX_stage == 3) {
    SW_matrix.set_SW_stage(2);
    SW_matrix.vmult_rhs_hu(rhs_hu, {h_old, hu_old, zeta_old,
                                    h_tmp_2, hu_tmp_2, zeta_tmp_2});
  }
  else {
    SW_matrix.set_SW_stage(5);
    SW_matrix.vmult_rhs_hu(rhs_hu, {h_old, hu_old, zeta_old,
                                    h_tmp_2, hu_tmp_2, zeta_tmp_2,
                                    h_tmp_3, hu_tmp_3, zeta_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_hu.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  /*--- Compute multigrid preconditioner for the height equation ---*/
  MGTransferMatrixFree<dim, float> mg_transfer;
  mg_transfer.build(dof_handler_discharge);
  using SmootherType = PreconditionChebyshev<SWOperator<dim, EquationData::n_stages,
                                                        EquationData::degree_h,
                                                        EquationData::degree_hu,
                                                        EquationData::degree_hc,
                                                        2*EquationData::degree_h + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_hu + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_hc + 1 + EquationData::extra_quadrature_degree,
                                                        LinearAlgebra::distributed::Vector<float>>,
                                             LinearAlgebra::distributed::Vector<float>>;
  mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, triangulation.n_global_levels() - 1);
  for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
    typename MatrixFree<dim, float>::AdditionalData additional_data_mg;
    additional_data_mg.tasks_parallel_scheme               = MatrixFree<dim, float>::AdditionalData::none;
    additional_data_mg.mapping_update_flags                = (update_values | update_quadrature_points | update_JxW_values);
    additional_data_mg.mapping_update_flags_inner_faces    = update_default;
    additional_data_mg.mapping_update_flags_boundary_faces = update_default;
    additional_data_mg.mg_level                            = level;

    std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
    mg_mf_storage_level->reinit(mapping_mg, dof_handlers, constraints, quadratures, additional_data_mg);
    mg_matrices_SW[level].initialize(mg_mf_storage_level, tmp, tmp);
    if(IMEX_stage == EquationData::n_stages + 1) {
      mg_matrices_SW[level].set_SW_stage(5);
    }
    else {
      mg_matrices_SW[level].set_SW_stage(2);
    }

    if(level > 0) {
      smoother_data[level].smoothing_range     = 15.0;
      smoother_data[level].degree              = 3;
      smoother_data[level].eig_cg_n_iterations = 10;
    }
    else {
      smoother_data[0].smoothing_range     = 2e-2;
      smoother_data[0].degree              = numbers::invalid_unsigned_int;
      smoother_data[0].eig_cg_n_iterations = mg_matrices_SW[0].m();
    }
    mg_matrices_SW[level].compute_diagonal();
    smoother_data[level].preconditioner = mg_matrices_SW[level].get_matrix_diagonal_inverse();
  }
  mg_smoother.initialize(mg_matrices_SW, smoother_data);

  MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
  mg_coarse.initialize(mg_smoother);
  mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices_SW);
  Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  PreconditionMG<dim,
                 LinearAlgebra::distributed::Vector<float>,
                 MGTransferMatrixFree<dim, float>> preconditioner(dof_handler_discharge, mg, mg_transfer);

  /*--- Solve the linear system for the discharge ---*/
  if(IMEX_stage == 2) {
    hu_tmp_2.equ(1.0, hu_old);
    cg.solve(SW_matrix, hu_tmp_2, rhs_hu, preconditioner);
  }
  else if(IMEX_stage == 3) {
    hu_tmp_3.equ(1.0, hu_tmp_2);
    cg.solve(SW_matrix, hu_tmp_3, rhs_hu, preconditioner);
  }
  else {
    hu_curr.equ(1.0, hu_tmp_3);
    cg.solve(SW_matrix, hu_curr, rhs_hu, preconditioner);
  }
}


// @sect{<code>SWSolver::update_tracer</code>}

// This implements the update of the tracer
//
template<int dim>
void SWSolver<dim>::update_tracer() {
  TimerOutput::Scope t(time_table, "Update tracer");

  const std::vector<unsigned int> tmp = {2};
  SW_matrix.initialize(matrix_free_storage, tmp, tmp);

  if(IMEX_stage == 2) {
    SW_matrix.set_SW_stage(3);
    SW_matrix.vmult_rhs_hc(rhs_hc, {h_old, hu_old, hc_old});
  }
  else if(IMEX_stage == 3) {
    SW_matrix.set_SW_stage(3);
    SW_matrix.vmult_rhs_hc(rhs_hc, {h_old, hu_old, hc_old,
                                    h_tmp_2, hu_tmp_2, hc_tmp_2});
  }
  else {
    SW_matrix.set_SW_stage(6);
    SW_matrix.vmult_rhs_hc(rhs_hc, {h_old, hu_old, hc_old,
                                    h_tmp_2, hu_tmp_2, hc_tmp_2,
                                    h_tmp_3, hu_tmp_3, hc_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_hc.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  /*--- Compute multigrid preconditioner for the tracer equation ---*/
  MGTransferMatrixFree<dim, float> mg_transfer;
  mg_transfer.build(dof_handler_tracer);
  using SmootherType = PreconditionChebyshev<SWOperator<dim, EquationData::n_stages,
                                                        EquationData::degree_h,
                                                        EquationData::degree_hu,
                                                        EquationData::degree_hc,
                                                        2*EquationData::degree_h + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_hu + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_hc + 1 + EquationData::extra_quadrature_degree,
                                                        LinearAlgebra::distributed::Vector<float>>,
                                             LinearAlgebra::distributed::Vector<float>>;
  mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, triangulation.n_global_levels() - 1);
  for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
    typename MatrixFree<dim, float>::AdditionalData additional_data_mg;
    additional_data_mg.tasks_parallel_scheme               = MatrixFree<dim, float>::AdditionalData::none;
    additional_data_mg.mapping_update_flags                = (update_values | update_quadrature_points | update_JxW_values);
    additional_data_mg.mapping_update_flags_inner_faces    = update_default;
    additional_data_mg.mapping_update_flags_boundary_faces = update_default;
    additional_data_mg.mg_level                            = level;

    std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
    mg_mf_storage_level->reinit(mapping_mg, dof_handlers, constraints, quadratures, additional_data_mg);
    mg_matrices_SW[level].initialize(mg_mf_storage_level, tmp, tmp);
    if(IMEX_stage == EquationData::n_stages + 1) {
      mg_matrices_SW[level].set_SW_stage(6);
    }
    else {
      mg_matrices_SW[level].set_SW_stage(3);
    }

    if(level > 0) {
      smoother_data[level].smoothing_range     = 15.0;
      smoother_data[level].degree              = 3;
      smoother_data[level].eig_cg_n_iterations = 10;
    }
    else {
      smoother_data[0].smoothing_range     = 2e-2;
      smoother_data[0].degree              = numbers::invalid_unsigned_int;
      smoother_data[0].eig_cg_n_iterations = mg_matrices_SW[0].m();
    }
    mg_matrices_SW[level].compute_diagonal();
    smoother_data[level].preconditioner = mg_matrices_SW[level].get_matrix_diagonal_inverse();
  }
  mg_smoother.initialize(mg_matrices_SW, smoother_data);

  MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>> mg_coarse;
  mg_coarse.initialize(mg_smoother);
  mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices_SW);
  Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  PreconditionMG<dim,
                 LinearAlgebra::distributed::Vector<float>,
                 MGTransferMatrixFree<dim, float>> preconditioner(dof_handler_tracer, mg, mg_transfer);

  /*--- Solve the linear system for the tracer ---*/
  if(IMEX_stage == 2) {
    hc_tmp_2.equ(1.0, hc_old);
    cg.solve(SW_matrix, hc_tmp_2, rhs_hc, preconditioner);
  }
  else if(IMEX_stage == 3) {
    hc_tmp_3.equ(1.0, hc_tmp_2);
    cg.solve(SW_matrix, hc_tmp_3, rhs_hc, preconditioner);
  }
  else {
    hc_old.equ(1.0, hc_tmp_3);
    cg.solve(SW_matrix, hc_old, rhs_hc, preconditioner);
  }
}


// @sect{ <code>SWSolver::output_results</code> }

// This method plots the current solution.
//
template<int dim>
void SWSolver<dim>::output_results(const unsigned int step) {
  TimerOutput::Scope t(time_table, "Output results");

  /*--- Save the fields ---*/
  DataOut<dim> data_out;

  h_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_height, h_old, "h", {DataComponentInterpretation::component_is_scalar});

  std::vector<std::string> velocity_names(dim, "hu");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
  hu_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_discharge, hu_old, velocity_names, component_interpretation_velocity);

  hc_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_tracer, hc_old, "hc", {DataComponentInterpretation::component_is_scalar});

  data_out.build_patches(mapping, EquationData::degree_hu);

  const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

  /*--- Call this function to be sure to be able to write again on these fields ---*/
  h_old.zero_out_ghost_values();
  hu_old.zero_out_ghost_values();
  hc_old.zero_out_ghost_values();
}


// Since we have solved a problem with analytic solution, we want to verify
// the correctness of our implementation by computing the errors of the
// numerical result against the analytic solution.
//
template <int dim>
void SWSolver<dim>::analyze_results() {
  TimerOutput::Scope t(time_table, "Analysis results: computing errrors");

  /*--- Errors for the height ---*/
  VectorTools::integrate_difference(dof_handler_height, h_old, h_exact,
                                    L1_error_per_cell_h, quadrature_height, VectorTools::L1_norm);
  const double error_h_L1 = VectorTools::compute_global_error(triangulation, L1_error_per_cell_h, VectorTools::L1_norm);
  pcout << "Verification via L1 error height:    " << error_h_L1 << std::endl;

  VectorTools::integrate_difference(dof_handler_height, h_old, h_exact,
                                    Linfty_error_per_cell_h, quadrature_height, VectorTools::Linfty_norm);
  const double error_h_Linfty = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_h, VectorTools::Linfty_norm);
  pcout << "Verification via Linfty error height:    " << error_h_Linfty << std::endl;

  /*--- Errors for the discharge ---*/
  VectorTools::integrate_difference(dof_handler_discharge, hu_old, hu_exact,
                                    L1_error_per_cell_hu, quadrature_discharge, VectorTools::L1_norm);
  const double error_hu_L1 = VectorTools::compute_global_error(triangulation, L1_error_per_cell_hu, VectorTools::L1_norm);
  pcout << "Verification via L1 error discharge:    " << error_hu_L1 << std::endl;

  VectorTools::integrate_difference(dof_handler_discharge, hu_old, hu_exact,
                                    Linfty_error_per_cell_hu, quadrature_discharge, VectorTools::Linfty_norm);
  const double error_hu_Linfty = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_hu, VectorTools::Linfty_norm);
  pcout << "Verification via Linfty error discharge:    " << error_hu_Linfty << std::endl;

  /*--- Errors for the tracer ---*/
  VectorTools::integrate_difference(dof_handler_tracer, hc_old, hc_exact,
                                    L1_error_per_cell_hc, quadrature_tracer, VectorTools::L1_norm);
  const double error_hc_L1 = VectorTools::compute_global_error(triangulation, L1_error_per_cell_hc, VectorTools::L1_norm);
  pcout << "Verification via L1 error tracer:    " << error_hc_L1 << std::endl;

  VectorTools::integrate_difference(dof_handler_tracer, hc_old, hc_exact,
                                    Linfty_error_per_cell_hc, quadrature_tracer, VectorTools::Linfty_norm);
  const double error_hc_Linfty = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_hc, VectorTools::Linfty_norm);
  pcout << "Verification via Linfty error tracer:    " << error_hc_Linfty << std::endl;

  /*--- Save errors ---*/
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_error_height    << error_h_L1      << std::endl;
    output_error_height    << error_h_Linfty  << std::endl;
    output_error_discharge << error_hu_L1     << std::endl;
    output_error_discharge << error_hu_Linfty << std::endl;
    output_error_tracer    << error_hc_L1     << std::endl;
    output_error_tracer    << error_hc_Linfty << std::endl;
  }
}


// The following function is used in determining the minimum height
//
template<int dim>
double SWSolver<dim>::get_min_height() {
  const unsigned int n_q_points = quadrature_height.size();

  FEValues<dim> fe_values(fe_height, quadrature_height, update_values);
  std::vector<double> solution_values(n_q_points);

  double min_local_height = std::numeric_limits<double>::max();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_height.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      if(IMEX_stage == 2) {
        fe_values.get_function_values(h_tmp_2, solution_values);
      }
      else if(IMEX_stage == 3) {
        fe_values.get_function_values(h_tmp_3, solution_values);
      }
      else {
        fe_values.get_function_values(h_old, solution_values);
      }

      for(unsigned int q = 0; q < n_q_points; ++q) {
        min_local_height = std::min(min_local_height, solution_values[q]);
      }
    }
  }

  return Utilities::MPI::min(min_local_height, MPI_COMM_WORLD);
}


// The following function is used in determining the maximum height
//
template<int dim>
double SWSolver<dim>::get_max_height() {
  const unsigned int n_q_points = quadrature_height.size();

  FEValues<dim> fe_values(fe_height, quadrature_height, update_values);
  std::vector<double> solution_values(n_q_points);

  double max_local_height = std::numeric_limits<double>::min();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_height.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      if(IMEX_stage == 2) {
        fe_values.get_function_values(h_tmp_2, solution_values);
      }
      else if(IMEX_stage == 3) {
        fe_values.get_function_values(h_tmp_3, solution_values);
      }
      else {
        fe_values.get_function_values(h_old, solution_values);
      }

      for(unsigned int q = 0; q < n_q_points; ++q) {
        max_local_height = std::max(max_local_height, solution_values[q]);
      }
    }
  }

  return Utilities::MPI::max(max_local_height, MPI_COMM_WORLD);
}


// The following function is used in determining the maximum Courant numbers along the two directions
//
template<int dim>
std::tuple<double, double, double> SWSolver<dim>::compute_max_C_x_y() {
  FEValues<dim>               fe_values_h(fe_height, quadrature_height, update_values);
  std::vector<double>         solution_values_height(quadrature_height.size(), update_values);

  FEValues<dim>               fe_values_hu(fe_discharge, quadrature_height, update_values);
  std::vector<Vector<double>> solution_values_discharge(quadrature_height.size(), Vector<double>(dim));

  double max_C_x = std::numeric_limits<double>::min();
  double max_C_y = std::numeric_limits<double>::min();
  double max_C   = std::numeric_limits<double>::min();

  /*--- Loop over all cells ---*/
  auto tmp_cell = dof_handler_discharge.begin_active();
  for(const auto& cell: dof_handler_height.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values_h.reinit(cell);
      fe_values_h.get_function_values(h_old, solution_values_height);

      fe_values_hu.reinit(tmp_cell);
      fe_values_hu.get_function_values(hu_old, solution_values_discharge);

      for(unsigned int q = 0; q < quadrature_height.size(); ++q) {
        const auto h_q = solution_values_height[q];
        auto vel_q     = solution_values_discharge[q];
        vel_q         /= h_q;

        max_C_x = std::max(max_C_x,
                           EquationData::degree_hu*(std::abs(vel_q(0)) + std::sqrt(EquationData::g*h_q))*dt/cell->extent_in_direction(0));

        max_C_y = std::max(max_C_y,
                           EquationData::degree_hu*(std::abs(vel_q(1)) + std::sqrt(EquationData::g*h_q))*dt/cell->extent_in_direction(1));

        max_C   = std::max(max_C,
                           EquationData::degree_hu*(vel_q.l2_norm() + std::sqrt(EquationData::g*h_q))*dt/cell->diameter(mapping));
      }
    }
    ++tmp_cell;
  }

  return std::make_tuple(Utilities::MPI::max(max_C_x, MPI_COMM_WORLD),
                         Utilities::MPI::max(max_C_y, MPI_COMM_WORLD),
                         Utilities::MPI::max(max_C, MPI_COMM_WORLD));
}


// @sect{ <code>SWSolver::run</code> }

// This is the time marching function, which starting at <code>t0</code>
// advances in time using the projection method with time step <code>dt</code>
// until <code>T</code>.
//
// Its second parameter, <code>verbose</code> indicates whether the function
// should output information what it is doing at any given moment:
// we use the ConditionalOStream class to do that for us.
//
template<int dim>
void SWSolver<dim>::run(const bool verbose, const unsigned int output_interval) {
  ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  double time    = t0;
  unsigned int n = 0;

  analyze_results();
  output_results(0);

  while(std::abs(T - time) > 1e-10) {
    time += dt;
    n++;
    pcout << "Step = " << n << " Time = " << time << std::endl;

    /*--- Second stage of IMEX scheme ---*/
    IMEX_stage = 2;
    SW_matrix.set_IMEX_stage(IMEX_stage);

    verbose_cout << "  Update height stage 2" << std::endl;
    update_height();
    pcout << "Minimum height " << get_min_height() << std::endl;
    pcout << "Maximum height " << get_max_height() << std::endl;

    verbose_cout << "  Update discharge stage 2" << std::endl;
    /*SW_matrix.set_h_curr(h_tmp_2);
    SW_matrix.set_hu_curr(hu_tmp_2);
    MGTransferMatrixFree<dim, float> mg_transfer;
    mg_transfer.build(dof_handler_height);
    mg_transfer.interpolate_to_mg(dof_handler_height, level_projection_h, h_tmp_2);
    mg_transfer.build(dof_handler_discharge);
    mg_transfer.interpolate_to_mg(dof_handler_discharge, level_projection_hu, hu_tmp_2);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      mg_matrices_SW[level].set_h_curr(level_projection_h[level]);
      mg_matrices_SW[level].set_hu_curr(level_projection_hu[level]);
    }*/
    update_discharge();

    verbose_cout << "  Update tracer stage 2" << std::endl;
    update_tracer();

    /*--- Third stage of IMEX scheme ---*/
    IMEX_stage = 3;
    SW_matrix.set_IMEX_stage(IMEX_stage);

    verbose_cout << "  Update height stage 3" << std::endl;
    update_height();
    pcout << "Minimum height " << get_min_height() << std::endl;
    pcout << "Maximum height " << get_max_height() << std::endl;

    verbose_cout << "  Update discharge stage 3" << std::endl;
    /*SW_matrix.set_h_curr(h_tmp_3);
    SW_matrix.set_hu_curr(hu_tmp_3);
    mg_transfer.build(dof_handler_height);
    mg_transfer.interpolate_to_mg(dof_handler_height, level_projection_h, h_tmp_3);
    mg_transfer.build(dof_handler_discharge);
    mg_transfer.interpolate_to_mg(dof_handler_discharge, level_projection_hu, hu_tmp_3);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      mg_matrices_SW[level].set_h_curr(level_projection_h[level]);
      mg_matrices_SW[level].set_hu_curr(level_projection_hu[level]);
    }*/
    update_discharge();

    verbose_cout << "  Update tracer stage 3" << std::endl;
    update_tracer();

    /*--- Final stage of RK scheme to update ---*/
    IMEX_stage = 4;
    SW_matrix.set_IMEX_stage(IMEX_stage);

    verbose_cout << "  Update density" << std::endl;
    update_height();
    pcout << "Minimum height " << get_min_height() << std::endl;
    pcout << "Maximum height " << get_max_height() << std::endl;

    verbose_cout << "  Update discharge" << std::endl;
    update_discharge();

    verbose_cout << "  Update tracer" << std::endl;
    update_tracer();

    /*--- Update for next step ---*/
    h_old.equ(1.0, h_curr);
    zeta_old.equ(1.0, h_old);
    for(const auto& cell: dof_handler_height.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        std::vector<types::global_dof_index> dof_indices(fe_height.dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        for(unsigned int idx = 0; idx < dof_indices.size(); ++idx) {
          zeta_old(dof_indices[idx]) -= zb_exact.value(dof_location_map[dof_indices[idx]]);
        }
      }
    }
    hu_old.equ(1.0, hu_curr);

    /*--- Analyze the results ---*/
    analyze_results();

    const auto max_C_x_y = compute_max_C_x_y();
    pcout << "C_x = " << std::get<0>(max_C_x_y) << std::endl;
    pcout << "C_y = " << std::get<1>(max_C_x_y) << std::endl;
    pcout << "C   = " << std::get<2>(max_C_x_y) << std::endl;

    /*--- Save the results each 'output_interval' steps ---*/
    if(n % output_interval == 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
    }

    /*--- Restore time step towards the end of simulation if needed ---*/
    if(T - time < dt && T - time > 1e-10) {
      dt = T - time;
      SW_matrix.set_dt(dt);
      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
        mg_matrices_SW[level].set_dt(dt);
      }
    }
  }

  /*--- Save the final results if not previously done ---*/
  if(n % output_interval != 0) {
    verbose_cout << "Plotting Solution final" << std::endl;
    output_results(n);
  }
}


// @sect{ The main function }

// The main function is quite standard. We just need to declare the SWSolver
// instance and let the simulation run.
//

int main(int argc, char *argv[]) {
  try {
    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    SWSolver<2> test(data);
    test.run(data.verbose, data.output_interval);

    if(curr_rank == 0)
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Apparently everything went fine!" << std::endl
                << "Don't forget to brush your teeth :-)" << std::endl
                << std::endl;

    return 0;
  }
  catch(std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

}
