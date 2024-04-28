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
  FESystem<dim> fe_velocity;
  FESystem<dim> fe_tracer;

  /*--- Degrees of freedom handlers for all the variables ---*/
  DoFHandler<dim> dof_handler_height;
  DoFHandler<dim> dof_handler_velocity;
  DoFHandler<dim> dof_handler_tracer;

  /*--- Auxiliary mapping for possible curved boundary ---*/
  MappingQ1<dim> mapping;
  MappingQ1<dim> mapping_mg; /*--- Auxiliary mapping for multigrid for the sake of generality ---*/

  /*--- Auxiliary quadratures for the variables for which we can keep track ---*/
  QGaussLobatto<dim> quadrature_height;
  QGaussLobatto<dim> quadrature_velocity;
  QGaussLobatto<dim> quadrature_tracer;

  /*--- Variables for the height ---*/
  LinearAlgebra::distributed::Vector<double> zeta_old;
  LinearAlgebra::distributed::Vector<double> zeta_tmp_2;
  LinearAlgebra::distributed::Vector<double> zeta_tmp_3;
  LinearAlgebra::distributed::Vector<double> zeta_curr;
  LinearAlgebra::distributed::Vector<double> rhs_zeta;

  /*--- Variables for the velocity ---*/
  LinearAlgebra::distributed::Vector<double> u_old;
  LinearAlgebra::distributed::Vector<double> u_tmp_2;
  LinearAlgebra::distributed::Vector<double> u_tmp_3;
  LinearAlgebra::distributed::Vector<double> u_curr;
  LinearAlgebra::distributed::Vector<double> rhs_u;

  /*--- Variables for the tracer ---*/
  LinearAlgebra::distributed::Vector<double> c_old;
  LinearAlgebra::distributed::Vector<double> c_tmp_2;
  LinearAlgebra::distributed::Vector<double> c_tmp_3;
  LinearAlgebra::distributed::Vector<double> rhs_c;

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

  void update_velocity(); /*--- Function to update the velocity ---*/

  void update_tracer(); /*--- Function to update the tracer ---*/

  void output_results(const unsigned int step); /*--- Function to save the results ---*/

  void analyze_results(); /*--- Function to compute the errors (we have an analytical solutino here) ---*/

private:
  /*--- Function to set the initial conditions ---*/
  EquationData::TotalHeight<dim>        zeta_exact;
  EquationData::Velocity<dim>           u_exact;
  EquationData::Tracer<dim>             c_exact;
  EquationData::Bathymetry<dim, double> zb_exact;

  /*--- Auxiliary structures for the matrix-free ---*/
  std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

  SWOperator<dim, EquationData::n_stages,
             EquationData::degree_zeta, EquationData::degree_u, EquationData::degree_c,
             2*EquationData::degree_zeta + 1 + EquationData::extra_quadrature_degree,
             2*EquationData::degree_u + 1 + EquationData::extra_quadrature_degree,
             2*EquationData::degree_c + 1 + EquationData::extra_quadrature_degree,
             LinearAlgebra::distributed::Vector<double>> SW_matrix;

  MGLevelObject<SWOperator<dim, EquationData::n_stages,
                           EquationData::degree_zeta, EquationData::degree_u, EquationData::degree_c,
                           2*EquationData::degree_zeta + 1 + EquationData::extra_quadrature_degree,
                           2*EquationData::degree_u + 1 + EquationData::extra_quadrature_degree,
                           2*EquationData::degree_c + 1 + EquationData::extra_quadrature_degree,
                           LinearAlgebra::distributed::Vector<float>>> mg_matrices_SW;

  std::vector<const DoFHandler<dim>*> dof_handlers; /*--- Auxiliary container for the matrix-free ---*/

  std::vector<const AffineConstraints<double>*> constraints; /*--- Auxiliary container for the matrix-free ---*/
  AffineConstraints<double> constraints_height,
                            constraints_velocity,
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
  std::ofstream output_error_velocity;
  std::ofstream output_error_tracer;

  Vector<double> L1_error_per_cell_zeta,
                 Linfty_error_per_cell_zeta,
                 L1_error_per_cell_u,
                 Linfty_error_per_cell_u,
                 L1_error_per_cell_c,
                 Linfty_error_per_cell_c;  /*--- Auxiliary variables to compute the errors ---*/

  MGLevelObject<LinearAlgebra::distributed::Vector<float>> level_projection_zeta;
  //MGLevelObject<LinearAlgebra::distributed::Vector<float>> level_projection_u; /*--- Auxiliary variables for multigrid purposes ---*/

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
  fe_height(FE_DGQ<dim>(EquationData::degree_zeta), 1),
  fe_velocity(FE_DGQ<dim>(EquationData::degree_u), dim),
  fe_tracer(FE_DGQ<dim>(EquationData::degree_c), 1),
  dof_handler_height(triangulation),
  dof_handler_velocity(triangulation),
  dof_handler_tracer(triangulation),
  mapping(),
  mapping_mg(),
  quadrature_height(EquationData::degree_zeta + 1),
  quadrature_velocity(EquationData::degree_u + 1),
  quadrature_tracer(EquationData::degree_c + 1),
  zeta_exact(data.initial_time),
  u_exact(data.initial_time),
  c_exact(data.initial_time),
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
  output_error_velocity("./" + data.dir + "/error_analysis_u.dat", std::ofstream::out),
  output_error_tracer("./" + data.dir + "/error_analysis_c.dat", std::ofstream::out) {
    AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

    matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

    dof_handlers.clear();

    constraints.clear();
    constraints_height.clear();
    constraints_velocity.clear();
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
  dof_handler_velocity.distribute_dofs(fe_velocity);
  dof_handler_tracer.distribute_dofs(fe_tracer);

  pcout << "dim (space height) = " << dof_handler_height.n_dofs()
        << std::endl
        << "dim (space velocity) = " << dof_handler_velocity.n_dofs()
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
  dof_handlers.push_back(&dof_handler_velocity);
  dof_handlers.push_back(&dof_handler_tracer);

  /*--- Set the container with the constraints. Each entry is empty (no Dirichlet and weak imposition in general)
        and this is necessary only for compatibilty reasons ---*/
  constraints.push_back(&constraints_height);
  constraints.push_back(&constraints_velocity);
  constraints.push_back(&constraints_tracer);

  /*--- Set the quadrature formula to compute the integrals for assembling bilinear and linear forms ---*/
  quadratures.push_back(QGauss<1>(2*EquationData::degree_u + 1 + EquationData::extra_quadrature_degree));

  /*--- Initialize the matrix-free structure with DofHandlers, Constraints, Quadratures and AdditionalData ---*/
  matrix_free_storage->reinit(mapping, dof_handlers, constraints, quadratures, additional_data);

  /*--- Initialize the variables related to the height ---*/
  matrix_free_storage->initialize_dof_vector(zeta_old, 0);
  matrix_free_storage->initialize_dof_vector(zeta_tmp_2, 0);
  matrix_free_storage->initialize_dof_vector(zeta_tmp_3, 0);
  matrix_free_storage->initialize_dof_vector(zeta_curr, 0);
  matrix_free_storage->initialize_dof_vector(rhs_zeta, 0);

  /*--- Initialize the variables related to the velocity ---*/
  matrix_free_storage->initialize_dof_vector(u_old, 1);
  matrix_free_storage->initialize_dof_vector(u_tmp_2, 1);
  matrix_free_storage->initialize_dof_vector(u_tmp_3, 1);
  matrix_free_storage->initialize_dof_vector(u_curr, 1);
  matrix_free_storage->initialize_dof_vector(rhs_u, 1);

  /*--- Initialize the variables related to the tracer ---*/
  matrix_free_storage->initialize_dof_vector(c_old, 2);
  matrix_free_storage->initialize_dof_vector(c_tmp_2, 2);
  matrix_free_storage->initialize_dof_vector(c_tmp_3, 2);
  matrix_free_storage->initialize_dof_vector(rhs_c, 2);

  /*--- Initialize the auxiliary variables to check the errors ---*/
  Vector<double> error_per_cell_tmp(triangulation.n_active_cells());
  L1_error_per_cell_zeta.reinit(error_per_cell_tmp);
  Linfty_error_per_cell_zeta.reinit(error_per_cell_tmp);
  L1_error_per_cell_u.reinit(error_per_cell_tmp);
  Linfty_error_per_cell_u.reinit(error_per_cell_tmp);
  L1_error_per_cell_c.reinit(error_per_cell_tmp);
  Linfty_error_per_cell_c.reinit(error_per_cell_tmp);

  /*--- Initialize the multigrid structure ---*/
  mg_matrices_SW.clear_elements();
  dof_handler_height.distribute_mg_dofs();
  dof_handler_velocity.distribute_mg_dofs();
  dof_handler_tracer.distribute_mg_dofs();

  level_projection_zeta = MGLevelObject<LinearAlgebra::distributed::Vector<float>>(0, triangulation.n_global_levels() - 1);
  //level_projection_u   = MGLevelObject<LinearAlgebra::distributed::Vector<float>>(0, triangulation.n_global_levels() - 1);
  mg_matrices_SW.resize(0, triangulation.n_global_levels() - 1);
  for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
    typename MatrixFree<dim, float>::AdditionalData additional_data_mg;
    additional_data_mg.mg_level = level;

    std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
    mg_mf_storage_level->reinit(mapping_mg, dof_handlers, constraints, quadratures, additional_data_mg);
    mg_mf_storage_level->initialize_dof_vector(level_projection_zeta[level], 0);
    //mg_mf_storage_level->initialize_dof_vector(level_projection_u[level], 1);

    mg_matrices_SW[level].set_dt(dt);
  }
}


// @sect{ <code>SWSolver::initialize</code> }

// This method loads the initial data
//
template<int dim>
void SWSolver<dim>::initialize() {
  TimerOutput::Scope t(time_table, "Initialize state");

  VectorTools::interpolate(mapping, dof_handler_height, zeta_exact, zeta_old);
  VectorTools::interpolate(mapping, dof_handler_velocity, u_exact, u_old);
  VectorTools::interpolate(mapping, dof_handler_tracer, c_exact, c_old);

  /*VectorTools::project(mapping, dof_handler_height, constraints_height, QGauss<dim>(EquationData::degree_zeta + 1), zeta_exact, zeta_old);
  VectorTools::project(mapping, dof_handler_velocity, constraints_velocity, QGauss<dim>(EquationData::degree_u + 1), u_exact, u_old);
  VectorTools::project(mapping, dof_handler_tracer, constraints_tracer, QGauss<dim>(EquationData::degree_c + 1), c_exact, c_old);*/
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
    SW_matrix.vmult_rhs_zeta(rhs_zeta, {zeta_old, u_old});
  }
  else if(IMEX_stage == 3) {
    SW_matrix.set_SW_stage(1);
    SW_matrix.vmult_rhs_zeta(rhs_zeta, {zeta_old, u_old,
                                        zeta_tmp_2, u_tmp_2});
  }
  else {
    SW_matrix.set_SW_stage(4);
    SW_matrix.vmult_rhs_zeta(rhs_zeta, {zeta_old, u_old,
                                        zeta_tmp_2, u_tmp_2,
                                        zeta_tmp_3, u_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_zeta.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  /*--- Compute multigrid preconditioner for the height equation ---*/
  MGTransferMatrixFree<dim, float> mg_transfer;
  mg_transfer.build(dof_handler_height);
  using SmootherType = PreconditionChebyshev<SWOperator<dim, EquationData::n_stages,
                                                        EquationData::degree_zeta,
                                                        EquationData::degree_u,
                                                        EquationData::degree_c,
                                                        2*EquationData::degree_zeta + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_u + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_c + 1 + EquationData::extra_quadrature_degree,
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
    zeta_tmp_2.equ(1.0, zeta_old);
    cg.solve(SW_matrix, zeta_tmp_2, rhs_zeta, preconditioner);
  }
  else if(IMEX_stage == 3) {
    zeta_tmp_3.equ(1.0, zeta_tmp_2);
    cg.solve(SW_matrix, zeta_tmp_3, rhs_zeta, preconditioner);
  }
  else {
    zeta_curr.equ(1.0, zeta_tmp_3);
    cg.solve(SW_matrix, zeta_curr, rhs_zeta, preconditioner);
  }
}


// @sect{<code>SWSolver::update_velocity</code>}

// This implements the update of the velocity
//
template<int dim>
void SWSolver<dim>::update_velocity() {
  TimerOutput::Scope t(time_table, "Update velocity");

  const std::vector<unsigned int> tmp = {1};
  SW_matrix.initialize(matrix_free_storage, tmp, tmp);

  if(IMEX_stage == 2) {
    SW_matrix.set_SW_stage(2);
    SW_matrix.vmult_rhs_hu(rhs_u, {zeta_old, u_old});
  }
  else if(IMEX_stage == 3) {
    SW_matrix.set_SW_stage(2);
    SW_matrix.vmult_rhs_hu(rhs_u, {zeta_old, u_old,
                                   zeta_tmp_2, u_tmp_2});
  }
  else {
    SW_matrix.set_SW_stage(5);
    SW_matrix.vmult_rhs_hu(rhs_u, {zeta_old, u_old,
                                   zeta_tmp_2, u_tmp_2,
                                   zeta_tmp_3, u_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_u.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  /*--- Compute multigrid preconditioner for the height equation ---*/
  MGTransferMatrixFree<dim, float> mg_transfer;
  mg_transfer.build(dof_handler_velocity);
  using SmootherType = PreconditionChebyshev<SWOperator<dim, EquationData::n_stages,
                                                        EquationData::degree_zeta,
                                                        EquationData::degree_u,
                                                        EquationData::degree_c,
                                                        2*EquationData::degree_zeta + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_u + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_c + 1 + EquationData::extra_quadrature_degree,
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
                 MGTransferMatrixFree<dim, float>> preconditioner(dof_handler_velocity, mg, mg_transfer);

  /*--- Solve the linear system for the velocity ---*/
  if(IMEX_stage == 2) {
    u_tmp_2.equ(1.0, u_old);
    cg.solve(SW_matrix, u_tmp_2, rhs_u, preconditioner);
  }
  else if(IMEX_stage == 3) {
    u_tmp_3.equ(1.0, u_tmp_2);
    cg.solve(SW_matrix, u_tmp_3, rhs_u, preconditioner);
  }
  else {
    u_curr.equ(1.0, u_tmp_3);
    cg.solve(SW_matrix, u_curr, rhs_u, preconditioner);
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
    SW_matrix.vmult_rhs_hc(rhs_c, {zeta_old, u_old, c_old});
  }
  else if(IMEX_stage == 3) {
    SW_matrix.set_SW_stage(3);
    SW_matrix.vmult_rhs_hc(rhs_c, {zeta_old, u_old, c_old,
                                   zeta_tmp_2, u_tmp_2, c_tmp_2});
  }
  else {
    SW_matrix.set_SW_stage(6);
    SW_matrix.vmult_rhs_hc(rhs_c, {zeta_old, u_old, c_old,
                                   zeta_tmp_2, u_tmp_2, c_tmp_2,
                                   zeta_tmp_3, u_tmp_3, c_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_c.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  /*--- Compute multigrid preconditioner for the tracer equation ---*/
  MGTransferMatrixFree<dim, float> mg_transfer;
  mg_transfer.build(dof_handler_tracer);
  using SmootherType = PreconditionChebyshev<SWOperator<dim, EquationData::n_stages,
                                                        EquationData::degree_zeta,
                                                        EquationData::degree_u,
                                                        EquationData::degree_c,
                                                        2*EquationData::degree_zeta + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_u + 1 + EquationData::extra_quadrature_degree,
                                                        2*EquationData::degree_c + 1 + EquationData::extra_quadrature_degree,
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
    c_tmp_2.equ(1.0, c_old);
    cg.solve(SW_matrix, c_tmp_2, rhs_c, preconditioner);
  }
  else if(IMEX_stage == 3) {
    c_tmp_3.equ(1.0, c_tmp_2);
    cg.solve(SW_matrix, c_tmp_3, rhs_c, preconditioner);
  }
  else {
    c_old.equ(1.0, c_tmp_3);
    cg.solve(SW_matrix, c_old, rhs_c, preconditioner);
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

  zeta_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_height, zeta_old, "zeta", {DataComponentInterpretation::component_is_scalar});

  std::vector<std::string> velocity_names(dim, "u");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
  u_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_velocity, u_old, velocity_names, component_interpretation_velocity);

  c_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_tracer, c_old, "c", {DataComponentInterpretation::component_is_scalar});

  data_out.build_patches(mapping, EquationData::degree_u);

  const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

  /*--- Call this function to be sure to be able to write again on these fields ---*/
  zeta_old.zero_out_ghost_values();
  u_old.zero_out_ghost_values();
  c_old.zero_out_ghost_values();
}


// Since we have solved a problem with analytic solution, we want to verify
// the correctness of our implementation by computing the errors of the
// numerical result against the analytic solution.
//
template <int dim>
void SWSolver<dim>::analyze_results() {
  TimerOutput::Scope t(time_table, "Analysis results: computing errrors");

  /*--- Errors for the height ---*/
  VectorTools::integrate_difference(dof_handler_height, zeta_old, zeta_exact,
                                    L1_error_per_cell_zeta, quadrature_height, VectorTools::L1_norm);
  const double error_zeta_L1 = VectorTools::compute_global_error(triangulation, L1_error_per_cell_zeta, VectorTools::L1_norm);
  pcout << "Verification via L1 error height:    " << error_zeta_L1 << std::endl;

  VectorTools::integrate_difference(dof_handler_height, zeta_old, zeta_exact,
                                    Linfty_error_per_cell_zeta, quadrature_height, VectorTools::Linfty_norm);
  const double error_zeta_Linfty = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_zeta, VectorTools::Linfty_norm);
  pcout << "Verification via Linfty error height:    " << error_zeta_Linfty << std::endl;

  /*--- Errors for the velocity ---*/
  VectorTools::integrate_difference(dof_handler_velocity, u_old, u_exact,
                                    L1_error_per_cell_u, quadrature_velocity, VectorTools::L1_norm);
  const double error_u_L1 = VectorTools::compute_global_error(triangulation, L1_error_per_cell_u, VectorTools::L1_norm);
  pcout << "Verification via L1 error velocity:    " << error_u_L1 << std::endl;

  VectorTools::integrate_difference(dof_handler_velocity, u_old, u_exact,
                                    Linfty_error_per_cell_u, quadrature_velocity, VectorTools::Linfty_norm);
  const double error_u_Linfty = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_u, VectorTools::Linfty_norm);
  pcout << "Verification via Linfty error velocity:    " << error_u_Linfty << std::endl;

  /*--- Errors for the tracer ---*/
  VectorTools::integrate_difference(dof_handler_tracer, c_old, c_exact,
                                    L1_error_per_cell_c, quadrature_tracer, VectorTools::L1_norm);
  const double error_c_L1 = VectorTools::compute_global_error(triangulation, L1_error_per_cell_c, VectorTools::L1_norm);
  pcout << "Verification via L1 error tracer:    " << error_c_L1 << std::endl;

  VectorTools::integrate_difference(dof_handler_tracer, c_old, c_exact,
                                    Linfty_error_per_cell_c, quadrature_tracer, VectorTools::Linfty_norm);
  const double error_c_Linfty = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_c, VectorTools::Linfty_norm);
  pcout << "Verification via Linfty error tracer:    " << error_c_Linfty << std::endl;

  /*--- Save errors ---*/
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_error_height   << error_zeta_L1     << std::endl;
    output_error_height   << error_zeta_Linfty << std::endl;
    output_error_velocity << error_u_L1        << std::endl;
    output_error_velocity << error_u_Linfty    << std::endl;
    output_error_tracer   << error_c_L1        << std::endl;
    output_error_tracer   << error_c_Linfty    << std::endl;
  }
}


// The following function is used in determining the minimum height
//
template<int dim>
double SWSolver<dim>::get_min_height() {
  const unsigned int n_q_points = quadrature_height.size();

  FEValues<dim> fe_values(fe_height, quadrature_height, update_values | update_quadrature_points);
  std::vector<double> solution_values(n_q_points);

  double min_local_height = std::numeric_limits<double>::max();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_height.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      if(IMEX_stage == 2) {
        fe_values.get_function_values(zeta_tmp_2, solution_values);
      }
      else if(IMEX_stage == 3) {
        fe_values.get_function_values(zeta_tmp_3, solution_values);
      }
      else {
        fe_values.get_function_values(zeta_old, solution_values);
      }

      for(unsigned int q = 0; q < n_q_points; ++q) {
        min_local_height = std::min(min_local_height, solution_values[q] + zb_exact.value(fe_values.quadrature_point(q)));
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

  FEValues<dim> fe_values(fe_height, quadrature_height, update_values | update_quadrature_points);
  std::vector<double> solution_values(n_q_points);

  double max_local_height = std::numeric_limits<double>::min();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_height.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      if(IMEX_stage == 2) {
        fe_values.get_function_values(zeta_tmp_2, solution_values);
      }
      else if(IMEX_stage == 3) {
        fe_values.get_function_values(zeta_tmp_3, solution_values);
      }
      else {
        fe_values.get_function_values(zeta_old, solution_values);
      }

      for(unsigned int q = 0; q < n_q_points; ++q) {
        max_local_height = std::max(max_local_height, solution_values[q] + zb_exact.value(fe_values.quadrature_point(q)));
      }
    }
  }

  return Utilities::MPI::max(max_local_height, MPI_COMM_WORLD);
}


// The following function is used in determining the maximum Courant numbers along the two directions
//
template<int dim>
std::tuple<double, double, double> SWSolver<dim>::compute_max_C_x_y() {
  FEValues<dim>               fe_values_zeta(fe_height, quadrature_height, update_values | update_quadrature_points);
  std::vector<double>         solution_values_height(quadrature_height.size(), update_values);

  FEValues<dim>               fe_values_u(fe_velocity, quadrature_height, update_values);
  std::vector<Vector<double>> solution_values_velocity(quadrature_height.size(), Vector<double>(dim));

  double max_C_x = std::numeric_limits<double>::min();
  double max_C_y = std::numeric_limits<double>::min();
  double max_C   = std::numeric_limits<double>::min();

  /*--- Loop over all cells ---*/
  auto tmp_cell = dof_handler_velocity.begin_active();
  for(const auto& cell: dof_handler_height.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values_zeta.reinit(cell);
      fe_values_zeta.get_function_values(zeta_old, solution_values_height);

      fe_values_u.reinit(tmp_cell);
      fe_values_u.get_function_values(u_old, solution_values_velocity);

      for(unsigned int q = 0; q < quadrature_height.size(); ++q) {
        const auto h_q   = solution_values_height[q] + zb_exact.value(fe_values_zeta.quadrature_point(q));
        const auto vel_q = solution_values_velocity[q];

        max_C_x = std::max(max_C_x,
                           EquationData::degree_u*(std::abs(vel_q(0)) + std::sqrt(EquationData::g*h_q))*dt/cell->extent_in_direction(0));

        max_C_y = std::max(max_C_y,
                           EquationData::degree_u*(std::abs(vel_q(1)) + std::sqrt(EquationData::g*h_q))*dt/cell->extent_in_direction(1));

        max_C   = std::max(max_C,
                           EquationData::degree_u*(vel_q.l2_norm() + std::sqrt(EquationData::g*h_q))*dt/cell->diameter(mapping));
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

    verbose_cout << "  Update velocity stage 2" << std::endl;
    SW_matrix.set_zeta_curr(zeta_tmp_2);
    //SW_matrix.set_u_curr(u_tmp_2);
    MGTransferMatrixFree<dim, float> mg_transfer;
    mg_transfer.build(dof_handler_height);
    mg_transfer.interpolate_to_mg(dof_handler_height, level_projection_zeta, zeta_tmp_2);
    /*mg_transfer.build(dof_handler_velocity);
    mg_transfer.interpolate_to_mg(dof_handler_velocity, level_projection_u, u_tmp_2);*/
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      mg_matrices_SW[level].set_zeta_curr(level_projection_zeta[level]);
      //mg_matrices_SW[level].set_u_curr(level_projection_u[level]);
    }
    update_velocity();

    verbose_cout << "  Update tracer stage 2" << std::endl;
    update_tracer();

    /*--- Third stage of IMEX scheme ---*/
    IMEX_stage = 3;
    SW_matrix.set_IMEX_stage(IMEX_stage);

    verbose_cout << "  Update height stage 3" << std::endl;
    update_height();
    pcout << "Minimum height " << get_min_height() << std::endl;
    pcout << "Maximum height " << get_max_height() << std::endl;

    verbose_cout << "  Update velocity stage 3" << std::endl;
    SW_matrix.set_zeta_curr(zeta_tmp_3);
    //SW_matrix.set_u_curr(u_tmp_3);
    mg_transfer.build(dof_handler_height);
    mg_transfer.interpolate_to_mg(dof_handler_height, level_projection_zeta, zeta_tmp_3);
    /*mg_transfer.build(dof_handler_velocity);
    mg_transfer.interpolate_to_mg(dof_handler_velocity, level_projection_u, u_tmp_3);*/
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      mg_matrices_SW[level].set_zeta_curr(level_projection_zeta[level]);
      //mg_matrices_SW[level].set_u_curr(level_projection_u[level]);
    }
    update_velocity();

    verbose_cout << "  Update tracer stage 3" << std::endl;
    update_tracer();

    /*--- Final stage of RK scheme to update ---*/
    IMEX_stage = 4;
    SW_matrix.set_IMEX_stage(IMEX_stage);

    verbose_cout << "  Update density" << std::endl;
    update_height();
    pcout << "Minimum height " << get_min_height() << std::endl;
    pcout << "Maximum height " << get_max_height() << std::endl;

    verbose_cout << "  Update velocity" << std::endl;
    SW_matrix.set_zeta_curr(zeta_curr);
    mg_transfer.build(dof_handler_height);
    mg_transfer.interpolate_to_mg(dof_handler_height, level_projection_zeta, zeta_curr);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      mg_matrices_SW[level].set_zeta_curr(level_projection_zeta[level]);
    }
    update_velocity();

    verbose_cout << "  Update tracer" << std::endl;
    update_tracer();

    /*--- Update for next step ---*/
    zeta_old.equ(1.0, zeta_curr);
    u_old.equ(1.0, u_curr);

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
