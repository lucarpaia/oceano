/* Author: Giuseppe Orlando, 2024. */

// We start by including all the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
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

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/timer.h>

#include "advection_operator.h"

using namespace Advection;

// @sect{The <code>AdvectionSolver</code> class}

// Now we are ready for the main class of the program. It implements the calls to the various steps
// of the implicit time discretization scheme for advection.
//
template<int dim>
class AdvectionSolver {
public:
  AdvectionSolver(RunTimeParameters::Data_Storage& data); /*--- Class constructor ---*/

  void run(const bool verbose = false, const unsigned int output_interval = 10); /*--- Main routine to effectively run a simulation ---*/

protected:
  const double t0; /*--- Initial time ---*/
  const double T;  /*--- Final time ---*/

  double dt; /*--- Time step ---*/

  EquationData::Density<dim>  rho_init; /*--- Instance of 'Density' class to initialize the advected field. ---*/
  EquationData::Velocity<dim> u_init;   /*--- Instance of 'Velocity' class to initialize the advecting field. ---*/

  parallel::distributed::Triangulation<dim> triangulation; /*--- Variable that stores the triangulation ---*/

  /*--- Finite Element space ---*/
  FESystem<dim> fe_rho;
  FESystem<dim> fe_u;

  /*--- Handlers for dofs ---*/
  DoFHandler<dim> dof_handler_rho;
  DoFHandler<dim> dof_handler_u;

  /*--- Auxiliary quadrature formula to compute the maximum velocity ---*/
  QGaussLobatto<dim> quadrature_u;

  /*--- Mapping for the sake of generality ---*/
  MappingQ1<dim> mapping;

  /*--- Now we define all the vectors for the solution.  ---*/
  LinearAlgebra::distributed::Vector<double> rho_curr;
  LinearAlgebra::distributed::Vector<double> rho_old;
  LinearAlgebra::distributed::Vector<double> rhs_rho;

  LinearAlgebra::distributed::Vector<double> u;

  DeclException2(ExcInvalidTimeStep,
                 double,
                 double,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  /*--- Set of functions to be executed ---*/
  void create_triangulation(const unsigned int n_refines); /*--- Create the grid ---*/

  void setup_dofs(); /*--- Set degrees of freedom ---*/

  void initialize(); /*--- Initialize all the fields ---*/

  void update_field(); /*--- Update the advected field ---*/

  void output_results(const unsigned int step); /*--- Save results ---*/

  double get_max_velocity(); /*--- Compute maximum velocity ---*/

private:
  /*--- Technical member to handle the various steps ---*/
  std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

  /*--- Now we need an instance of the class implemented before with the weak form ---*/
  AdvectionImplicitOperator<dim,
                            EquationData::degree,
                            EquationData::degree + 1,
                            LinearAlgebra::distributed::Vector<double>> advection_matrix;

  /*--- Here we define the 'AffineConstraints' instance.
        This is just a technical issue, due to MatrixFree requirements. In general
        this class is used to impose boundary conditions (or any kind of constraints), but in this case, since
        we are using a weak imposition of bcs, everything is already in the weak forms and so these instances
        will be default constructed ---*/
  AffineConstraints<double> constraints_rho,
                            constraints_u;

  /*--- Now a bunch of variables handled by 'ParamHandler' introduced at the beginning of the code ---*/
  unsigned int max_its;
  double       eps;

  std::string saving_dir;

  /*--- Finally, some output related streams ---*/
  ConditionalOStream pcout;

  std::ofstream      time_out;
  ConditionalOStream ptime_out;
  TimerOutput        time_table;

  std::ofstream output_linear_solver_iterations;
};


// In the constructor, we just read all the data from the
// <code>Data_Storage</code> object that is passed as an argument, verify that
// the data we read are reasonable and, finally, create the triangulation and
// load the initial data.
//
template<int dim>
AdvectionSolver<dim>::AdvectionSolver(RunTimeParameters::Data_Storage& data):
  t0(data.initial_time),
  T(data.final_time),
  dt(data.dt),
  rho_init(data.initial_time),
  u_init(data.initial_time),
  triangulation(MPI_COMM_WORLD,
                parallel::distributed::Triangulation<dim>::limit_level_difference_at_vertices,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe_rho(FE_DGQ<dim>(EquationData::degree), 1),
  fe_u(FE_DGQ<dim>(EquationData::degree), dim),
  dof_handler_rho(triangulation),
  dof_handler_u(triangulation),
  quadrature_u(EquationData::degree + 1),
  mapping(),
  advection_matrix(data),
  max_its(data.max_iterations),
  eps(data.eps),
  saving_dir(data.dir),
  pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_out("./" + data.dir + "/time_analysis_" +
           Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
  ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
  output_linear_solver_iterations("./" + data.dir + "/GMRES_iterations.dat", std::ofstream::out) {
    AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

    matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

    create_triangulation(data.n_refines);
    setup_dofs();
    initialize();
}


// The method that creates the triangulation and refines it the needed number
// of times.
//
template<int dim>
void AdvectionSolver<dim>::create_triangulation(const unsigned int n_refines) {
  TimerOutput::Scope t(time_table, "Create triangulation");

  GridGenerator::subdivided_hyper_cube(triangulation, 15, -0.5, 0.5, true);

  /*--- Consider periodic boundary conditions ---*/
  std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>> periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
  GridTools::collect_periodic_faces(triangulation, 2, 3, 1, periodic_faces);
  triangulation.add_periodicity(periodic_faces);

  pcout << "Number of refines = " << n_refines << std::endl;
  triangulation.refine_global(n_refines);
}


// After creating the triangulation, it creates the mesh dependent
// data, i.e. it distributes degrees of freedom, and
// initializes the vectors that we will use.
//
template<int dim>
void AdvectionSolver<dim>::setup_dofs() {
  TimerOutput::Scope t(time_table, "Setup dofs");

  pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

  /*--- Distribute dofs ---*/
  dof_handler_rho.distribute_dofs(fe_rho);
  dof_handler_u.distribute_dofs(fe_u);

  pcout << "dim (X_h) = " << dof_handler_rho.n_dofs()
        << std::endl
        << std::endl;

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags                = (update_values | update_gradients | update_JxW_values | update_quadrature_points);
  additional_data.mapping_update_flags_inner_faces    = (update_values | update_JxW_values | update_normal_vectors | update_quadrature_points);
  additional_data.mapping_update_flags_boundary_faces = update_default;
  additional_data.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;

  std::vector<const DoFHandler<dim>*> dof_handlers; /*--- Vector of dof_handlers to feed the 'MatrixFree'. Here the order
                                                          counts and enters into the game as parameter of FEEvaluation and
                                                          FEFaceEvaluation in the previous class ---*/
  dof_handlers.push_back(&dof_handler_rho);
  dof_handlers.push_back(&dof_handler_u);

  /*--- Focus now on the constraints ---*/
  constraints_rho.clear();
  constraints_rho.close();
  constraints_u.clear();
  constraints_u.close();
  std::vector<const AffineConstraints<double>*> constraints;
  constraints.push_back(&constraints_rho);
  constraints.push_back(&constraints_u);

  std::vector<QGauss<1>> quadratures; /*--- We cannot directly use something such as 'quadrature_rho',
                                            because the 'MatrixFree' structure wants a quadrature formula for 1D. ---*/
  quadratures.push_back(QGauss<1>(EquationData::degree + 1));

  /*--- Initialize the matrix-free structure and size properly the vectors. Here again the
        second input argument of the 'initialize_dof_vector' method depends on the order of 'dof_handlers' ---*/
  matrix_free_storage->reinit(mapping, dof_handlers, constraints, quadratures, additional_data);

  /*--- Initialize vectors related to the field ---*/
  matrix_free_storage->initialize_dof_vector(rho_old);
  matrix_free_storage->initialize_dof_vector(rho_curr);
  matrix_free_storage->initialize_dof_vector(rhs_rho);

  /*--- Initialize the vectro with the velocity ---*/
  matrix_free_storage->initialize_dof_vector(u, 1);
}


// This method loads the initial data.
//
template<int dim>
void AdvectionSolver<dim>::initialize() {
  TimerOutput::Scope t(time_table, "Initialize field");

  VectorTools::interpolate(mapping, dof_handler_rho, rho_init, rho_old);

  VectorTools::interpolate(mapping, dof_handler_u, u_init, u);
}


// We are finally ready to update the advected field.
//
template<int dim>
void AdvectionSolver<dim>::update_field() {
  TimerOutput::Scope t(time_table, "Update field");

  /*--- We initialize with the propoer dof_handler ---*/
  const std::vector<unsigned int> tmp = {0};
  advection_matrix.initialize(matrix_free_storage, tmp, tmp);

  /*--- Now, we compute the right-hand side. ---*/
  advection_matrix.vmult_rhs(rhs_rho, rho_old);

  /*--- Build the linear solver; in this case we specifiy the maximum number of iterations and residual ---*/
  SolverControl solver_control(max_its, eps*rhs_rho.l2_norm());
  SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);

  /*--- Build the preconditioner ---*/
  PreconditionJacobi<AdvectionImplicitOperator<dim,
                                               EquationData::degree,
                                               EquationData::degree + 1,
                                               LinearAlgebra::distributed::Vector<double>>> preconditioner;
  advection_matrix.compute_diagonal();
  preconditioner.initialize(advection_matrix);

  /*--- Solve the linear system ---*/
  rho_curr.equ(1.0, rho_old);
  gmres.solve(advection_matrix, rho_curr, rhs_rho, preconditioner);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_linear_solver_iterations << solver_control.last_step() << std::endl;
  }
}


// This method plots the current solution.
//
template<int dim>
void AdvectionSolver<dim>::output_results(const unsigned int step) {
  TimerOutput::Scope t(time_table, "Output results");

  DataOut<dim> data_out;

  /*--- Save the advected field ---*/
  rho_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_rho, rho_old, "Field", {DataComponentInterpretation::component_is_scalar});

  /*--- Save data ---*/
  data_out.build_patches(mapping, EquationData::degree);

  const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

  /*--- Make sure you can write on these vectors ---*/
  rho_old.zero_out_ghost_values();
}


// The following function is used in determining the maximum velocity
// in order to compute the Courant number
//
template<int dim>
double AdvectionSolver<dim>::get_max_velocity() {
  FEValues<dim> fe_values_velocity(mapping, fe_u, quadrature_u, update_values);
  std::vector<Vector<double>> velocity_values(quadrature_u.size(), Vector<double>(dim));

  double max_local_velocity = 0.0;

  /*--- Loop over all cells ---*/
  for(const auto& cell : dof_handler_u.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values_velocity.reinit(cell);

      fe_values_velocity.get_function_values(u, velocity_values);

      /*--- Loop over quadrature points (GaussLobatto with degree + 1 coincides with nodes) ---*/
      for(unsigned int q = 0; q < quadrature_u.size(); q++) {
        max_local_velocity = std::max(max_local_velocity, std::sqrt(velocity_values[q][0]*velocity_values[q][0] +
                                                                    velocity_values[q][1]*velocity_values[q][1]));
      }
    }
  }

  /*--- Compute the maximum velocity in parallel ---*/
  const double max_velocity = Utilities::MPI::max(max_local_velocity, MPI_COMM_WORLD);

  return max_velocity;
}


// @sect{ <code>AdvectionSolver::run</code> }

// This is the time marching function, which starting at <code>t0</code>
// advances in time using the projection method with time step <code>dt</code>
// until <code>T</code>.
//
// Its second parameter, <code>verbose</code> indicates whether the function
// should output information what it is doing at any given moment:
// we use the ConditionalOStream class to do that for us.
//
template<int dim>
void AdvectionSolver<dim>::run(const bool verbose, const unsigned int output_interval) {
  ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  /*--- Save results ---*/
  output_results(0);

  /*--- Initialize variables to check when we are ---*/
  double time    = t0;
  unsigned int n = 0;

  /*--- Perform the temporal loop ---*/
  while(std::abs(T - time) > 1e-10) {
    time += dt;
    n++;
    pcout << "Step = " << n << " Time = " << time << std::endl;

    /*--- Advect the field ---*/
    verbose_cout << "  Advecting field" << std::endl;
    advection_matrix.set_advecting_field(u);
    update_field();

    pcout << "CFL_u = " << dt*get_max_velocity()*EquationData::degree*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation, mapping) << std::endl;

    /*--- Update for next step ---*/
    rho_old.equ(1.0, rho_curr);

    /*--- Save results ---*/
    if(n % output_interval == 0) {
      verbose_cout << "Plotting final solution" << std::endl;
      output_results(n);
    }

    /*--- In case dt is not a multiple of T, we reduce dt in order to end up at T ---*/
    if(T - time < dt && T - time > 1e-10) {
      dt = T - time;
      advection_matrix.set_dt(dt);
    }
  }

  if(n % output_interval != 0) {
    verbose_cout << "Plotting final solution" << std::endl;
    output_results(n);
  }
}


// @sect{ The main function }

// The main function looks very much like in all the other tutorial programs. We first initialize MPI,
// we initialize the class 'AdvectionSolver' with the dimension as template parameter and then
// let the method 'run' do the job.
//
int main(int argc, char *argv[]) {
  try {
    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    AdvectionSolver<2> test(data);
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
