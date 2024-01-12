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
 * Author: Martin Kronbichler, 2020
 */

// Run-time polymorphism can be an elegant solution do deal within a single code 
// with multiple physical models and different numerical choices. One for example 
// would like to test different numerical fluxes or time integrators. Also the 
// initial and boundary conditions specific to each test case can be resolved with
// a pointer base class which act as an interface with virtual functions defined in it. 
// However, calls to virtual functions (model flux, numerical flux, boundary conditions)
// happens at each quadrature points and this can be the cause of a bit of overhead.
// Moreover these choices are tested in a development phase and typically 
// both the model and the numerics are kept fixed by users. Also boundary and initial 
// conditions for real case scenario consists in reading external data and not in 
// using analytical functions. In order to not introduce in the optimized deal.II code 
// call to virtual functions that would point, almost always, to the same derived class,
// we use instead c++ preprocessors. C++ preprocessor allows to avoid interface classes
// and to define just the classes actually used.

// The following are the preprocessors that select the initial and boundary conditions.
// We implement two different test cases. The first one is an analytical
// solution in 2d, whereas the second is a channel flow around a cylinder as
// described in the introduction.
#define ICBC_ISENTROPICVORTEX
#undef  ICBC_FLOWAROUNDCYLINDER
// and numerical flux (Riemann solver) at the faces between cells. For this
// program, we have implemented a modified variant of the Lax--Friedrichs
// flux and the Harten--Lax--van Leer (HLL) flux:
#define  NUMERICALFLUX_LAXFRIEDRICHSMODIFIED
#undef    NUMERICALFLUX_HARTENVANLEER
// The include files are similar to the previous matrix-free tutorial programs
// step-37, step-48, and step-59
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iomanip>
#include <iostream>

// The following file includes the CellwiseInverseMassMatrix data structure
// that we will use for the mass matrix inversion, the only new include
// file for this tutorial program:
#include <deal.II/matrix_free/operators.h>

// The following files include the oceano libraries:
#include <time_integrator/LowStorageRungeKuttaIntegrator.h>
#include <space_discretization/EulerDG.h>
#include <io/CommandLineParser.h>
// The following files are included depending on
// the Preprocessor keys. This is necessary because 
// we have done a limited use of virtual classes; on the contrary 
// each of these header files contains the same class definition, so they
// cannot be linked together. 
#if defined ICBC_ISENTROPICVORTEX
#include <icbc/Icbc_IsentropicVortex.h>
#elif defined ICBC_FLOWAROUNDCYLINDER
#include <icbc/Icbc_FlowAroundCylinder.h>
#endif

namespace Problem
{
  using namespace dealii;

  // We collect some parameters that control the execution of the program at the top of the
  // file. These are parameters that the user should not change for operational 
  // simuations. They can be thus considered known at compile time leading, I guess,
  // to some optimization. Besides the dimension and polynomial degree we want to run with, we
  // also specify a number of points in the Gaussian quadrature formula we
  // want to use for the nonlinear terms in the Euler equations.
  constexpr unsigned int dimension            = 2;
  constexpr unsigned int n_variables          = dimension + 2;  
  constexpr unsigned int fe_degree            = 1;
  constexpr unsigned int n_q_points_1d        = fe_degree + 2;

  using Number = double;

  // Next off are some details of the time integrator, namely a Courant number
  // that scales the time step size in terms of the formula $\Delta t =
  // \text{Cr} n_\text{stages} \frac{h}{(p+1)^{1.5} (\|\mathbf{u} +
  // c)_\text{max}}$, as well as a selection of a few low-storage Runge--Kutta
  // methods. We specify the Courant number per stage of the Runge--Kutta
  // scheme, as this gives a more realistic expression of the numerical cost
  // for schemes of various numbers of stages.
  const double courant_number = 0.15 / std::pow(fe_degree, 1.5);
  constexpr TimeIntegrator::LowStorageRungeKuttaScheme lsrk_scheme = TimeIntegrator::stage_5_order_4;



  // @sect3{The EulerProblem class}

  // This class combines the EulerOperator class with the time integrator and
  // the usual global data structures such as FiniteElement and DoFHandler, to
  // actually run the simulations of the Euler problem.
  //
  // The member variables are a triangulation, a finite element, a mapping (to
  // create high-order curved surfaces, see e.g. step-10), and a DoFHandler to
  // describe the degrees of freedom. In addition, we keep an instance of the
  // EulerOperator described above around, which will do all heavy lifting in
  // terms of integrals, and some parameters for time integration like the
  // current time or the time step size.
  //
  // Furthermore, we use a PostProcessor instance to write some additional
  // information to the output file, in similarity to what was done in
  // step-33. The interface of the DataPostprocessor class is intuitive,
  // requiring us to provide information about what needs to be evaluated
  // (typically only the values of the solution, except for the Schlieren plot
  // that we only enable in 2d where it makes sense), and the names of what
  // gets evaluated. Note that it would also be possible to extract most
  // information by calculator tools within visualization programs such as
  // ParaView, but it is so much more convenient to do it already when writing
  // the output.
  //
  // The class is templated with the dimension and the number of variables.
  // Both are important because all templated members and methods
  // (Tensor and Functions for example) that are defined in nested classes
  // needs to inherit `dim` and `n_vars`. 
  template <int dim, int n_vars>
  class EulerProblem
  {
  public:
    EulerProblem(IO::ParameterHandler       &,
                 ICBC::BcBase<dim, n_vars>* bc);

    void run();

  private:
    void make_grid_and_dofs();

    LinearAlgebra::distributed::Vector<Number> solution;

    ParameterHandler &prm;

    ConditionalOStream pcout;

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    FESystem<dim>   fe;
    MappingQ<dim>   mapping;
    DoFHandler<dim> dof_handler;

    TimerOutput timer;

    SpaceDiscretization::EulerOperator<dim, n_vars, fe_degree, n_q_points_1d>
      euler_operator;

    double time, time_step;

   public:
    class Postprocessor : public DataPostprocessor<dim>
    {
    public:
      Postprocessor(IO::ParameterHandler &param);

      virtual void evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override;

      virtual std::vector<std::string> get_names() const override;

      virtual std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const override;

      virtual UpdateFlags get_needed_update_flags() const override;

    private:
      const bool do_schlieren_plot;

    public:
      bool do_error;
      double output_tick;
      std::string output_filename;

      Model::Euler euler;
    };

   private:
    void output_results(
      Postprocessor      &postprocessor,
      const unsigned int  result_number);
  };


  // The constructor of the postprocessor class takes as arguments the parameters handler
  // class in order to read the output parameters defined from the parameter file. These
  // parameters are stored as class members.
  template <int dim, int n_vars>
  EulerProblem<dim, n_vars>::Postprocessor::Postprocessor(
    IO::ParameterHandler &prm)
    : do_schlieren_plot(dim == 2)
    , euler(prm)
  {
    prm.enter_subsection("Output parameters");
    do_error = prm.get_double("Output_error");
    output_tick = prm.get_double("Output_tick");
    output_filename = prm.get("Output_filename");
    prm.leave_subsection();
  }



  // For the main evaluation of the field variables, we first check that the
  // lengths of the arrays equal the expected values (the lengths `2*dim+4` or
  // `2*dim+5` are derived from the sizes of the names we specify in the
  // get_names() function below). Then we loop over all evaluation points and
  // fill the respective information: First we fill the primal solution
  // variables of density $\rho$, momentum $\rho \mathbf{u}$ and energy $E$,
  // then we compute the derived velocity $\mathbf u$, the pressure $p$, the
  // speed of sound $c=\sqrt{\gamma p / \rho}$, as well as the Schlieren plot
  // showing $s = |\nabla \rho|^2$ in case it is enabled. (See step-69 for
  // another example where we create a Schlieren plot.)
  template <int dim, int n_vars>
  void EulerProblem<dim, n_vars>::Postprocessor::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &               computed_quantities) const
  {
    const unsigned int n_evaluation_points = inputs.solution_values.size();
    
    if (do_schlieren_plot == true)
      Assert(inputs.solution_gradients.size() == n_evaluation_points,
             ExcInternalError());

    Assert(computed_quantities.size() == n_evaluation_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == n_vars, ExcInternalError());
    Assert(computed_quantities[0].size() ==
             n_vars + (do_schlieren_plot == true ? 1 : 0),
           ExcInternalError());

    for (unsigned int p = 0; p < n_evaluation_points; ++p)
      {
        Tensor<1, n_vars> solution;
        for (unsigned int d = 0; d < n_vars; ++d)
          solution[d] = inputs.solution_values[p](d);

        const double         density  = solution[0];
        const Tensor<1, dim> velocity = euler.velocity<dim, n_vars>(solution);
        const double         pressure = euler.pressure<dim, n_vars>(solution);

        for (unsigned int d = 0; d < dim; ++d)
          computed_quantities[p](d) = velocity[d];
        computed_quantities[p](dim)     = pressure;
        computed_quantities[p](dim + 1) = std::sqrt(euler.gamma * pressure / density);

        if (do_schlieren_plot == true)
          computed_quantities[p](n_vars) =
            inputs.solution_gradients[p][0] * inputs.solution_gradients[p][0];
      }
  }



  template <int dim, int n_vars>
  std::vector<std::string> EulerProblem<dim, n_vars>::Postprocessor::get_names() const
  {
    std::vector<std::string> names;
    for (unsigned int d = 0; d < dim; ++d)
      names.emplace_back("velocity");
    names.emplace_back("pressure");
    names.emplace_back("speed_of_sound");

    if (do_schlieren_plot == true)
      names.emplace_back("schlieren_plot");

    return names;
  }



  // For the interpretation of quantities, we have scalar density, energy,
  // pressure, speed of sound, and the Schlieren plot, and vectors for the
  // momentum and the velocity.
  template <int dim, int n_vars>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  EulerProblem<dim, n_vars>::Postprocessor::get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation;
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    if (do_schlieren_plot == true)
      interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }



  // With respect to the necessary update flags, we only need the values for
  // all quantities but the Schlieren plot, which is based on the density
  // gradient.
  template <int dim, int n_vars>
  UpdateFlags EulerProblem<dim, n_vars>::Postprocessor::get_needed_update_flags() const
  {
    if (do_schlieren_plot == true)
      return update_values | update_gradients;
    else
      return update_values;
  }



  // The constructor for this class is unsurprising: We set up a parallel
  // triangulation based on the `MPI_COMM_WORLD` communicator, a vector finite
  // element with `n_vars` components for density, momentum, and energy, a
  // high-order mapping of the same degree as the underlying finite element,
  // and initialize the time and time step to zero.
  template <int dim, int n_vars>
  EulerProblem<dim, n_vars>::EulerProblem(IO::ParameterHandler      &param,
                                          ICBC::BcBase<dim, n_vars>* bc)
    : prm(param)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
    , triangulation(MPI_COMM_WORLD)
#endif
    , fe(FE_DGQ<dim>(fe_degree), n_vars)
    , mapping(fe_degree)
    , dof_handler(triangulation)
    , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    , euler_operator(param, bc, timer)
    , time(0)
    , time_step(0)
  {}



  // It is possible to create the mesh inside 
  // deal.II using the functions in the namespace GridGenearator. However 
  // here we use only the possibility to import the mesh from an external mesher, Gmsh. 
  // Gmsh is the smallest and most quickly set up open source tool we are aware of. 
  // One of the issues is that deal.II, at least until version 9.2, 
  // can only deal with meshes that only consist of quadrilaterals and hexahedra - 
  // tetrahedral meshes were not supported and will likely not be supported with all 
  // of the features deal.II offers for quadrilateral and hexahedral meshes for several 
  // versions following the 9.3 release that introduced support for simplicial and 
  // mixed meshes first. Gmsh can generate unstructured 2d quad meshes.
  //
  // Furthermore,
  // we also add a gravity force in vertical direction. Having the base mesh
  // in place (including the manifolds set by
  // GridGenerator::channel_with_cylinder()), we can then perform the
  // specified number of global refinements, create the unknown numbering from
  // the DoFHandler, and hand the DoFHandler and Mapping objects to the
  // initialization of the EulerOperator.
  template <int dim, int n_vars>
  void EulerProblem<dim, n_vars>::make_grid_and_dofs()
  {

    euler_operator.bc->set_body_force(
      std::make_unique<Functions::ConstantFunction<dim>>(
        std::vector<double>({0., 0.})));

    // The class GridIn can read many different mesh formats from a 
    // file from disk. In order to read a grid from a file, we generate an object 
    // of data type GridIn and associate the triangulation to it (i.e. we tell 
    // it to fill our triangulation object when we ask it to read the file). 
    // Then we open the respective file and initialize the triangulation with 
    // the data in the file
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
  
    char cwd[1024];
    Assert(getcwd(cwd, sizeof(cwd)) != NULL,
           ExcInternalError());   
    std::string current_working_directory(cwd);
    std::string slash("/");

    prm.enter_subsection("Mesh & geometry parameters");
    const std::string file_msh = prm.get("Mesh_filename");
    const unsigned int n_global_refinements = prm.get_integer("Number_of_refinements");
    prm.leave_subsection();
         
    std::ifstream f(current_working_directory+slash+file_msh);
    pcout << "Reading mesh file: " << file_msh << std::endl;

    gridin.read_msh(f);
 
    euler_operator.bc->set_boundary_conditions();

    triangulation.refine_global(n_global_refinements);

    dof_handler.distribute_dofs(fe);

    euler_operator.reinit(mapping, dof_handler);
    euler_operator.initialize_vector(solution);

    // In the following, we output some statistics about the problem. Because we
    // often end up with quite large numbers of cells or degrees of freedom, we
    // would like to print them with a comma to separate each set of three
    // digits. This can be done via "locales", although the way this works is
    // not particularly intuitive. step-32 explains this in slightly more
    // detail.
    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << " ( = " << (n_vars) << " [vars] x "
          << triangulation.n_global_active_cells() << " [cells] x "
          << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
          << std::endl;
    pcout.get_stream().imbue(s);
  }



  // For output, we first let the Euler operator compute the errors of the
  // numerical results. More precisely, we compute the error against the
  // analytical result for the analytical solution case, whereas we compute
  // the deviation against the background field with constant density and
  // energy and constant velocity in $x$ direction for the second test case.
  //
  // The next step is to create output. This is similar to what is done in
  // step-33: We let the postprocessor defined above control most of the
  // output, except for the primal field that we write directly. For the
  // analytical solution test case, we also perform another projection of the
  // analytical solution and print the difference between that field and the
  // numerical solution. Once we have defined all quantities to be written, we
  // build the patches for output. Similarly to step-65, we create a
  // high-order VTK output by setting the appropriate flag, which enables us
  // to visualize fields of high polynomial degrees. Finally, we call the
  // `DataOutInterface::write_vtu_in_parallel()` function to write the result
  // to the given file name. This function uses special MPI parallel write
  // facilities, which are typically more optimized for parallel file systems
  // than the standard library's `std::ofstream` variants used in most other
  // tutorial programs. A particularly nice feature of the
  // `write_vtu_in_parallel()` function is the fact that it can combine output
  // from all MPI ranks into a single file, making it unnecessary to have a
  // central record of all such files (namely, the "pvtu" file).
  //
  // For parallel programs, it is often instructive to look at the partitioning
  // of cells among processors. To this end, one can pass a vector of numbers
  // to DataOut::add_data_vector() that contains as many entries as the
  // current processor has active cells; these numbers should then be the
  // rank of the processor that owns each of these cells. Such a vector
  // could, for example, be obtained from
  // GridTools::get_subdomain_association(). On the other hand, on each MPI
  // process, DataOut will only read those entries that correspond to locally
  // owned cells, and these of course all have the same value: namely, the rank
  // of the current process. What is in the remaining entries of the vector
  // doesn't actually matter, and so we can just get away with a cheap trick: We
  // just fill *all* values of the vector we give to DataOut::add_data_vector()
  // with the rank of the current MPI process. The key is that on each process,
  // only the entries corresponding to the locally owned cells will be read,
  // ignoring the (wrong) values in other entries. The fact that every process
  // submits a vector in which the correct subset of entries is correct is all
  // that is necessary.
  //
  // @note As of 2023, Visit 3.3.3 can still not deal with higher-order cells.
  //   Rather, it simply reports that there is no data to show. To view the
  //   results of this program with Visit, you will want to comment out the
  //   line that sets `flags.write_higher_order_cells = true;`. On the other
  //   hand, Paraview is able to understand VTU files with higher order cells
  //   just fine.
  template <int dim, int n_vars>
  void EulerProblem<dim, n_vars>::output_results(
    Postprocessor      &postprocessor,
    const unsigned int  result_number)
  {
    const std::array<double, 3> errors =
      euler_operator.compute_errors(
        ICBC::ExactSolution<dimension, n_variables>(time), solution);

    const std::string quantity_name =
      postprocessor.do_error == 1 ? "error" : "norm";

    pcout << "Time:" << std::setw(8) << std::setprecision(3) << time
          << ", dt: " << std::setw(8) << std::setprecision(2) << time_step
          << ", " << quantity_name << " rho: " << std::setprecision(4)
          << std::setw(10) << errors[0] << ", rho * u: " << std::setprecision(4)
          << std::setw(10) << errors[1] << ", energy:" << std::setprecision(4)
          << std::setw(10) << errors[2] << std::endl;

    {
      TimerOutput::Scope t(timer, "output");

      DataOut<dim>  data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_dof_handler(dof_handler);
      {
        std::vector<std::string> names;
        names.emplace_back("density");
        for (unsigned int d = 0; d < dim; ++d)
          names.emplace_back("momentum");
        names.emplace_back("energy");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          interpretation;
        interpretation.push_back(
          DataComponentInterpretation::component_is_scalar);
        for (unsigned int d = 0; d < dim; ++d)
          interpretation.push_back(
            DataComponentInterpretation::component_is_part_of_vector);
        interpretation.push_back(
          DataComponentInterpretation::component_is_scalar);

        data_out.add_data_vector(dof_handler, solution, names, interpretation);
      }
      data_out.add_data_vector(solution, postprocessor);

      LinearAlgebra::distributed::Vector<Number> reference;

      if (postprocessor.do_error && dim == 2)
        {
          reference.reinit(solution);
          euler_operator.project(
            ICBC::ExactSolution<dimension, n_variables>(time), reference);
          reference.sadd(-1., 1, solution);
          std::vector<std::string> names;
          names.emplace_back("error_density");
          for (unsigned int d = 0; d < dim; ++d)
            names.emplace_back("error_momentum");
          names.emplace_back("error_energy");

          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation;
          interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);
          for (unsigned int d = 0; d < dim; ++d)
            interpretation.push_back(
              DataComponentInterpretation::component_is_part_of_vector);
          interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);

          data_out.add_data_vector(dof_handler,
                                   reference,
                                   names,
                                   interpretation);
        }

      Vector<double> mpi_owner(triangulation.n_active_cells());
      mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      data_out.add_data_vector(mpi_owner, "owner");

      data_out.build_patches(mapping,
                             fe.degree,
                             DataOut<dim>::curved_inner_cells);

      const std::string filename =
        postprocessor.output_filename + "_"
        + Utilities::int_to_string(result_number, 3) + ".vtu";
      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
    }
  }



  // The EulerProblem::run() function puts all pieces together. It starts off
  // by calling the function that creates the mesh and sets up data structures,
  // and then initializing the time integrator and the two temporary vectors of
  // the low-storage integrator. We call these vectors `rk_register_1` and
  // `rk_register_2`, and use the first vector to represent the quantity
  // $\mathbf{r}_i$ and the second one for $\mathbf{k}_i$ in the formulas for
  // the Runge--Kutta scheme outlined in the introduction. Before we start the
  // time loop, we compute the time step size by the
  // `EulerOperator::compute_cell_transport_speed()` function. For reasons of
  // comparison, we compare the result obtained there with the minimal mesh
  // size and print them to screen. For velocities and speeds of sound close
  // to unity as in this tutorial program, the predicted effective mesh size
  // will be close, but they could vary if scaling were different.
  template <int dim, int n_vars>
  void EulerProblem<dim, n_vars>::run()
  {
    {
      const unsigned int n_vect_number = VectorizedArray<Number>::size();
      const unsigned int n_vect_bits   = 8 * sizeof(Number) * n_vect_number;

      pcout << "Running with "
            << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
            << " MPI processes" << std::endl;
      pcout << "Vectorization over " << n_vect_number << ' '
            << (std::is_same<Number, double>::value ? "doubles" : "floats")
            << " = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ')'
            << std::endl;
    }

    make_grid_and_dofs();

    const TimeIntegrator::LowStorageRungeKuttaIntegrator integrator(lsrk_scheme);

    LinearAlgebra::distributed::Vector<Number> rk_register_1;
    LinearAlgebra::distributed::Vector<Number> rk_register_2;
    rk_register_1.reinit(solution);
    rk_register_2.reinit(solution);

    euler_operator.project(ICBC::Ic<dimension, n_variables>(), solution);

    double min_vertex_distance = std::numeric_limits<double>::max();
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        min_vertex_distance =
          std::min(min_vertex_distance, cell->minimum_vertex_distance());
    min_vertex_distance =
      Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

    time_step = courant_number * integrator.n_stages() /
                euler_operator.compute_cell_transport_speed(solution);
    pcout << "Time step size: " << time_step
          << ", minimal h: " << min_vertex_distance
          << ", initial transport scaling: "
          << 1. / euler_operator.compute_cell_transport_speed(solution)
          << std::endl
          << std::endl;

    // We have moved the constructor of the `Postprocessor` class in this
    // top level `run()`, so that we can pass it by reference to many
    // functions that postprocess the solution. For now there the postprocess
    // consists of only one function that print the solution and compute an
    // error or norm of the solution.
    Postprocessor postprocessor(prm);

    output_results(postprocessor, 0);

    // Now we are ready to start the time loop, which we run until the time
    // has reached the desired end time. Every 5 time steps, we compute a new
    // estimate for the time step -- since the solution is nonlinear, it is
    // most effective to adapt the value during the course of the
    // simulation. In case the Courant number was chosen too aggressively, the
    // simulation will typically blow up with time step NaN, so that is easy
    // to detect here. One thing to note is that roundoff errors might
    // propagate to the leading digits due to an interaction of slightly
    // different time step selections that in turn lead to slightly different
    // solutions. To decrease this sensitivity, it is common practice to round
    // or truncate the time step size to a few digits, e.g. 3 in this case. In
    // case the current time is near the prescribed 'tick' value for output
    // (e.g. 0.02), we also write the output. After the end of the time loop,
    // we summarize the computation by printing some statistics, which is
    // mostly done by the TimerOutput::print_wall_time_statistics() function.
    prm.enter_subsection("Time parameters");
    const double final_time = prm.get_double("Final_time");
    prm.leave_subsection();
    unsigned int timestep_number = 0;

    while (time < final_time - 1e-12)
      {
        ++timestep_number;
        if (timestep_number % 5 == 0)
          time_step =
            courant_number * integrator.n_stages() /
            Utilities::truncate_to_n_digits(
              euler_operator.compute_cell_transport_speed(solution), 3);

        {
          TimerOutput::Scope t(timer, "rk time stepping total");
          integrator.perform_time_step(euler_operator,
                                       time,
                                       time_step,
                                       solution,
                                       rk_register_1,
                                       rk_register_2);
        }

        time += time_step;

        if (static_cast<int>(time / postprocessor.output_tick) !=
              static_cast<int>((time - time_step) / postprocessor.output_tick) ||
            time >= final_time - 1e-12)
          output_results(
            postprocessor,
            static_cast<unsigned int>(std::round(time / postprocessor.output_tick)));
      }

    timer.print_wall_time_statistics(MPI_COMM_WORLD);
    pcout << std::endl;
  }

} // namespace Problem



// The main() function is not surprising and follows what was done in all
// previous MPI programs: As we run an MPI program, we need to call `MPI_Init()`
// and `MPI_Finalize()`, which we do through the
// Utilities::MPI::MPI_InitFinalize data structure. Note that we run the program
// only with MPI, and set the thread count to 1.
int main(int argc, char **argv)
{
  using namespace Problem;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      deallog.depth_console(0);

      // We define `CommandLineParser` objects that parse the command line
      IO::CommandLineParser prs;
      prs.parse_command_line(argc, argv);

      // We define `ParameterHandler` and `ParameterReader` objects, and let the latter 
      // read in the parameter values from a configuration textfile. The values so
      // read are then handed over to an instance of the EulerProblem class:
      IO::ParameterHandler prm;
      IO::ParameterReader  param(prm);      
      param.read_parameters(argv[2]);

      // The boundary condition class is the only class declared as a pointer.
      // This is because we want to realize run-time polymrophism. In the base class 
      // there are defined a bunch of members that are needed for all the test cases. 
      // Next, the pointer is allocated as a derived class specific 
      // to the test-case which will override the boundary conditions. 
      // The pointer is easily pass as argument to the `EulerProblem` class.
      ICBC::BcBase<dimension, n_variables> *bc;
      // The switch between the different test-cases is realized with Preprocessor keys.
      // The choice to use a Preprocessor also for the boundary conditions is because we
      // use it for the initial condition and we have mantained the same directive 
      // to easily switch both initial and boundary conditions depending on the test-case.
#if defined ICBC_ISENTROPICVORTEX          
      bc = new ICBC::BcIsentropicVortex<dimension, n_variables>;
#elif defined ICBC_FLOWAROUNDCYLINDER          
      bc = new ICBC::BcFlowAroundCylinder<dimension, n_variables>;
#else          
      Assert(false, ExcNotImplemented());
      return 0.;
#endif 
      
      // The EulerProblem class takes as argument the only two classes that have been
      // previously defined and that are filled at runtime. One is the pointer class
      // for the boundary conditions, the other is a reference to the parameter
      // class to read the paramteres from an external config file.     
      EulerProblem<dimension, n_variables> euler_problem(prm, bc);
      euler_problem.run();
      
      delete bc;
    }
  catch (std::exception &exc)
    {
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
  catch (...)
    {
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

  return 0;
}
