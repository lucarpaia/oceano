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
 *         Luca Arpaia,        2024
 *         Giuseppe Orlando,   2024
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
// and to define just the classes actually used. Each pre-processor must begins with
// the name of the class.

// First come the numerics. The following preprocessor select the time integrator. For
// now we have coded general explicit schemes that belong to the family of Runge-Kutta scheme.
#undef  TIMEINTEGRATOR_EXPLICITRUNGEKUTTA
#define TIMEINTEGRATOR_ADDITIVERUNGEKUTTA
// The numerical flux (Riemann solver) at the faces between cells. For this
// program, we have implemented a modified variant of the Lax--Friedrichs
// flux and the Harten--Lax--van Leer (HLL) flux:
#define NUMERICALFLUX_LAXFRIEDRICHSMODIFIED
#undef  NUMERICALFLUX_HARTENVANLEER
// The following are the preprocessors that select the initial and boundary conditions.
// We implement many test cases and a last generic class named as "Realistic".
// Here the data are imported from external files and it can target any shallow
// water simulation.
#undef  ICBC_ISENTROPICVORTEX
#undef  ICBC_FLOWAROUNDCYLINDER
#undef  ICBC_IMPULSIVEWAVE
#undef  ICBC_SHALLOWWATERVORTEX
#undef  ICBC_STOMMELGYRE
#undef  ICBC_LAKEATREST
#undef  ICBC_TRACERADVECTION
#undef  ICBC_CHANNELFLOW
#define ICBC_REALISTIC
// We have two models: a non-hydrostatic Euler model for perfect gas which was the
// original model coded in the deal.II example and the shallow water model. The Euler model
// is only used for debugging, to check consistency with the original deal.II example and
// it's not working in the present version.
// Additionally we can add tracers to the shallow water model.
#define MODEL_SHALLOWWATER
#undef  MODEL_SHALLOWWATERWITHTRACER
#undef  MODEL_EULER
// Next come the physics. With the following cpp keys one can switch between the different
// formulations of a given term in the right-hand side of the shallow water equations.
// For the bottom friction one has two formulations: a simple linear bottom friction and
// a non-linear Manning bottom friction. In both cases, in the icbc class must appear the
// definition of the drag coefficient, either the linear drag coefficient or the Manning
// number.
#undef  PHYSICS_BOTTOMFRICTIONLINEAR
#define PHYSICS_BOTTOMFRICTIONMANNING
// The followig key is for the wind stress. Either you can specify directly wind stress components
// or you can compute the wind stress with a quadratic formula
// For the wind stress either
#define PHYSICS_WINDSTRESSGENERAL
#undef  PHYSICS_WINDSTRESSQUADRATIC
// The following key is for the eddy diffusion/viscosity coefficient. You have two alternatives:
// a constant eddy coefficient or a more sophisticated turbulent model based on the mixing length,
// proposed by Smagorinsky in 1963. The next is for eddy viscosity:
#define PHYSICS_VISCOSITYCOEFFICIENTCONSTANT
#undef  PHYSICS_VISCOSITYCOEFFICIENTSMAGORINSKY
// and the following is for eddy diffusivity. In a test stage one may use different
// models for the diffusion and for the viscosity. However, momentum and tracer mixing are
// of course related and it is recommended to use the same model for both:
#define PHYSICS_DIFFUSIONCOEFFICIENTCONSTANT
#undef  PHYSICS_DIFFUSIONCOEFFICIENTSMAGORINSKY
// We end with a tuner class for the AMR:
#undef  AMR_HEIGHTGRADIENT
#define AMR_VORTICITY
#undef  AMR_TRACERGRADIENT
#undef  AMR_BATHYMETRY
#undef  AMR_FROMFILE
//
//
//
// Finally we define new preprocessors that do not identify a specific class but rather add a
// functionality to the scheme, for example the computation of tracers.
// For this reason it starts with `OCEANO_WITH`.
// In some case they are used just for a more compact `#if defined` statement.
#undef  OCEANO_WITH_TRACERS
#if defined MODEL_SHALLOWWATERWITHTRACER
#define OCEANO_WITH_TRACERS
#endif
#undef  OCEANO_WITH_MASSCONSERVATIONCHECK

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

// The following files include functionalities for the model input/outputs:
// these are some c++ functions to open and read files or to manipulate
// strings but also deal.ii classes that write the finite element solutions
// to files on distributed memories architectures
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/numerics/vector_tools.h>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

// The following file includes the CellwiseInverseMassMatrix data structure
// that we will use for the mass matrix inversion, the only new include
// file for this tutorial program:
#include <deal.II/matrix_free/operators.h>

// In order to refine our grids locally, we need a function from the library
// that decides which cells to flag for refinement or coarsening based on the
// error indicators we have computed. We need a class that help in transfering
// a discrete finite element function by interpolation while refining and/or
// coarsening a distributed grid and handles the necessary communication:
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

// The following files include the oceano libraries:
#include <space_discretization/OceanDG.h>
#include <space_discretization/OceanDGWithTracer.h>
#include <io/CommandLineParser.h>
#include <amr/AmrTuner.h>
// The following files are included depending on
// the Preprocessor keys. This is necessary because 
// we have done a limited use of virtual classes; on the contrary 
// each of these header files contains the same class definition, so they
// cannot be linked together.
#if defined TIMEINTEGRATOR_EXPLICITRUNGEKUTTA
#include <time_integrator/ExplicitRungeKuttaIntegrator.h>
#elif defined TIMEINTEGRATOR_ADDITIVERUNGEKUTTA
#include <time_integrator/AdditiveRungeKuttaIntegrator.h>
#endif
#if defined ICBC_ISENTROPICVORTEX
#include <icbc/Icbc_IsentropicVortex.h>
#elif defined ICBC_FLOWAROUNDCYLINDER
#include <icbc/Icbc_FlowAroundCylinder.h>
#elif defined ICBC_IMPULSIVEWAVE
#include <icbc/Icbc_ImpulsiveWave.h>
#elif defined ICBC_SHALLOWWATERVORTEX
#include <icbc/Icbc_ShallowWaterVortex.h>
#elif defined ICBC_STOMMELGYRE
#include <icbc/Icbc_StommelGyre.h>
#elif defined ICBC_LAKEATREST
#include <icbc/Icbc_LakeAtRest.h>
#elif defined ICBC_TRACERADVECTION
#include <icbc/Icbc_TracerAdvection.h>
#elif defined ICBC_CHANNELFLOW
#include <icbc/Icbc_ChannelFlow.h>
#elif defined ICBC_REALISTIC
#include <icbc/Icbc_Realistic.h>
#endif

namespace Problem
{
  using namespace dealii;

  // We collect some parameters that control the execution of the program at the top of the
  // file. These are parameters that the user should not change for operational 
  // simuations. They can be thus considered known at compile time leading, I guess,
  // to some optimization. Besides the dimension and polynomial degree we want to run with, we
  // also specify a number of points in the Gaussian quadrature formula we
  // want to use for the nonlinear terms in the shallow water equations.
  constexpr unsigned int dimension            = 2;
  constexpr unsigned int fe_degree            = 1;
  constexpr unsigned int n_q_points_1d        = floor(1.5*fe_degree) + 1;
#if defined MODEL_SHALLOWWATERWITHTRACER
  constexpr unsigned int n_tracers            = 1;
#endif

  using Number = double;

  // Next off is the choice of the time integrator scheme:
#if defined TIMEINTEGRATOR_EXPLICITRUNGEKUTTA
  constexpr TimeIntegrator::ExplicitRungeKuttaScheme rk_scheme = TimeIntegrator::stage_3_order_2;
#elif defined TIMEINTEGRATOR_ADDITIVERUNGEKUTTA
  constexpr TimeIntegrator::AdditiveRungeKuttaScheme rk_scheme = TimeIntegrator::stage_3_order_2;
#endif
  //
  //
  //
  // The user should stop here. The next lines are a consistency check between
  // the parameters and the preprocessors plus we set derived parameters that are
  // also known at compile time:
#if defined MODEL_EULER
  constexpr unsigned int n_tracers            = 1;
#elif defined MODEL_SHALLOWWATER
  constexpr unsigned int n_tracers            = 0;
#endif
  constexpr unsigned int n_variables          = dimension + 1 + n_tracers;

  // @sect3{The OceanoProblem class}

  // This class combines the OceanoOperator class with the time integrator and
  // the usual global data structures such as FiniteElement and DoFHandler, to
  // actually run the simulations of the shallow water problem.
  //
  // The member variables are a triangulation, a finite element, a mapping (to
  // create high-order curved surfaces, see e.g. step-10), and a DoFHandler to
  // describe the degrees of freedom. In addition, we keep an instance of the
  // OceanoOperator described above around, which will do all heavy lifting in
  // terms of integrals, and some parameters for time integration like the
  // current time or the time step size.
  //
  // Furthermore, we use a PostProcessor instance to write some additional
  // information to the output file, in similarity to what was done in
  // step-33. Differently from step-33, the Postprocessor does not inherit
  // from the deal.II class but has been re-written from skratch
  // to handle the multi-component case with many solution vectors. We have
  // however reused many functions.
  // The interface of the DataPostprocessor class is intuitive,
  // requiring us to provide information about what needs to be evaluated
  // (typically the solution fields, the solution at specific points, integrals
  // or an error), when it needs to be evaluated (at what frequency) and the names
  // of what gets evaluated. Note that it would also be possible to extract most
  // information by calculator tools within visualization programs such as
  // ParaView, but it is so much more convenient to do it already when writing
  // the output.
  //
  // The class is templated with the dimension and the number of tracers.
  // Both are important because they define the number of equation (and of
  // prognostic variables) in a multi-component system such as an ocean model,
  // which reads `1+dim+n_tra`.
  // FEEvaluation class needs to inherit both `dim` and `n_tra` and the same
  // holds for all all templated members and methods
  // (Tensor and Functions for example) that are defined in nested classes.
  template <int dim, int n_tra>
  class OceanoProblem
  {
  public:
    OceanoProblem(IO::ParameterHandler           &,
                  ICBC::BcBase<dim, 1+dim+n_tra> *bc);

    void run();

  private:
    void make_grid();
    void make_dofs();
    void make_ic(
      const ICBC::Ic<dim, 1+dim+n_tra>            ic,
      LinearAlgebra::distributed::Vector<Number> &postprocess_velocity);
    void refine_grid(
      const Amr::AmrTuner                        &amr_tuner,
      LinearAlgebra::distributed::Vector<Number> &postprocess_velocity);

    LinearAlgebra::distributed::Vector<Number> solution_height;
    LinearAlgebra::distributed::Vector<Number> solution_discharge;
    LinearAlgebra::distributed::Vector<Number> solution_tracer;

    ParameterHandler &prm;

    ConditionalOStream pcout;

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    FESystem<dim>   fe_height;
    FESystem<dim>   fe_discharge;
#ifdef OCEANO_WITH_TRACERS
    FESystem<dim>   fe_tracer;
#endif

    MappingQ<dim>   mapping;
    DoFHandler<dim> dof_handler_height;
    DoFHandler<dim> dof_handler_discharge;
    DoFHandler<dim> dof_handler_tracer;

    TimerOutput timer;

#ifndef OCEANO_WITH_TRACERS
    SpaceDiscretization::OceanoOperator<dim, n_tra, fe_degree, n_q_points_1d>
      oceano_operator;
#else
    SpaceDiscretization::OceanoOperatorWithTracer<dim, n_tra, fe_degree, n_q_points_1d>
      oceano_operator;
#endif

    double time, time_step;

   public:
    class Postprocessor
    {
    public:
      Postprocessor(IO::ParameterHandler     &param,
                    std::vector<std::string>  postproc_vars_name);

      ParameterHandler &prm;

      Utilities::MPI::RemotePointEvaluation<dim, dim> cache;

      std::vector<std::string> get_names() const;

      std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const;

      void write_history_gnuplot() const;

      std::string output_filename;

      std::map<double, std::vector<std::vector<double>>> pointHistory_data;
      std::vector<Point<dim>> point_vector;

      std::map<double, std::array<double, 2>> integralHistory_data;

      // variables
      std::vector<std::string> postproc_vars_name;
      unsigned int n_postproc_vars;
      // when
      double solution_tick;
      double pointHistory_tick;
      // what
      bool do_solution;
      bool do_pointHistory;
      bool do_integralHistory;
      bool do_error;
      bool do_meshsize;
    };

   private:
    void output_results(
      Postprocessor                                    &postprocessor,
      const unsigned int                                result_number,
      std::map<unsigned int, Point<dim>>               &dof_points,
      const LinearAlgebra::distributed::Vector<Number> &postprocess_velocity);
  };



  // The constructor of the postprocessor class takes as arguments the parameters handler
  // class in order to read the output parameters defined from the parameter file. These
  // parameters are stored as class members.
  template <int dim, int n_tra>
  OceanoProblem<dim, n_tra>::Postprocessor::Postprocessor(
      IO::ParameterHandler     &prm,
      std::vector<std::string>  postproc_vars_name)
    : prm(prm)
    , postproc_vars_name(postproc_vars_name)
  {
    prm.enter_subsection("Output parameters");
    output_filename = prm.get("Output_filename");
    solution_tick = prm.get_double("Solution_tick");
    pointHistory_tick = prm.get_double("Point_history_tick");

    std::vector<std::string> info;
    for (unsigned int num = 1; num < 20; ++num)
      {
         std::string pointHistory_info = prm.get("Point_history_"+std::to_string(num));
         if (pointHistory_info == "There is no entry Point_history in the parameter file")
           {
             continue;
           }
         boost::split(info, pointHistory_info, boost::is_any_of(":"));
         point_vector.push_back(Point<dim>(std::stof(info[0]), std::stof(info[1])));
      }

    do_solution        = true;
    do_pointHistory    = !point_vector.empty();
    do_integralHistory = false;
    do_error           = prm.get_bool("Output_error");
    do_meshsize        = prm.get_bool("Output_meshsize");

    prm.leave_subsection();

    n_postproc_vars = postproc_vars_name.size();
  }



  template <int dim, int n_tra>
  std::vector<std::string> OceanoProblem<dim, n_tra>::Postprocessor::get_names() const
  {
    return postproc_vars_name;
  }



  // For the interpretation of the postprocessed quantities, we have first the
  // velocity vector and then the scalar quantities.
  template <int dim, int n_tra>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  OceanoProblem<dim, n_tra>::Postprocessor::get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation;
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int v = 0; v < postproc_vars_name.size()-dim; ++v)
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }


  // The user can request output files to be generated for the point or integral
  // history. These files are in Gnuplot format but are basically just regular text
  // and can easily be imported into other programs well, for example into spreadsheets.
  // We write out a series of gnuplot files named "point_history" + "-01.gpl", etc.
  // The data file gives information about where the points and interpreting the
  // data. The names of the data columns is supplied, depending on
  // the model class.
  //
  // For each point, open a file to be written to, put helpful info about the
  // point into the file as comments, write general data stored.
  // Note that for the file name we use two digits, so up two 100 points can be
  // written.
  //
  // For the integral history we need high precision number to check accurately
  // mass conservation. The total mass in the computational domain is typically a
  // big number and we use 12 digits. The total mass flux across the boundary is
  // smaller, we use 6 digits.
  template <int dim, int n_tra>
  void OceanoProblem<dim, n_tra>::Postprocessor::write_history_gnuplot() const
  {
    typename std::vector<Point<dim>>::const_iterator point = point_vector.begin();
    for (unsigned int point_vector_index = 0;
         point != point_vector.end();
         ++point, ++point_vector_index)
      {
        const std::string filename_point =
          output_filename + "_pointHistory_"
          + Utilities::int_to_string(point_vector_index+1, 2) + ".gpl";

        std::ofstream to_gnuplot(filename_point);

        to_gnuplot << "# Requested location: " << *point
                   << '\n';
        to_gnuplot << "# Requested variables: " << "free_surface"; // lrp: TODO read from model class
        for (unsigned int v = 0; v < n_postproc_vars; ++v)
          {
             to_gnuplot << " " << postproc_vars_name[v];
          }
        to_gnuplot << '\n';
        to_gnuplot << "#\n";

        for (auto i : pointHistory_data)
          {
            to_gnuplot << i.first;
            to_gnuplot << " " << i.second[0][point_vector_index];
            for (unsigned int v = 0; v < n_postproc_vars; ++v)
              {
                to_gnuplot << " " << i.second[v+1][point_vector_index];
              }
            to_gnuplot << '\n';
          }
        to_gnuplot.close();
      }
#ifdef OCEANO_WITH_MASSCONSERVATIONCHECK
      const std::string filename_integral =
        output_filename + "_integralHistory.gpl";

      std::ofstream to_gnuplot(filename_integral);

      to_gnuplot << "# Requested variables: " << "depth\n";
      to_gnuplot << "# Time " << " " << "total_mass" << " " << "boundary_flux\n";

      for (auto i : integralHistory_data)
        {
          to_gnuplot << i.first << " " << std::setprecision(14) << i.second[0]
                                << " " << std::setprecision(12) << i.second[1]
                                << '\n';
        }
      to_gnuplot.close();
#endif
  }



  // The constructor for this class is unsurprising: We set up a parallel
  // triangulation based on the `MPI_COMM_WORLD` communicator. The
  // Triangulation constructor takes an argument specifying whether a
  // smoothing step shall be performed on the grid. Some degradation of
  // approximation properties has been observed for grids which are too
  // unstructured which are typical in ocean applications. For this reason
  // we initialize the smoothing, with a flag `eliminate_unrefined_islands`.
  // The we set up a vector finite element with `1+dim+n_tra` components
  // for water level, momentum, and tracers, a first-order mapping and initialize
  // the time and time step to zero. Deal.ii supports also high order mappings,
  // in principle of the same degree as the underlying finite element.
  template <int dim, int n_tra>
  OceanoProblem<dim, n_tra>::OceanoProblem(IO::ParameterHandler           &param,
                                           ICBC::BcBase<dim, 1+dim+n_tra> *bc)
    : prm(param)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
    , triangulation(MPI_COMM_WORLD, Triangulation<dim>::MeshSmoothing::eliminate_unrefined_islands)
#endif
    , fe_height(FE_DGQ<dim>(fe_degree), 1)
    , fe_discharge(FE_DGQ<dim>(fe_degree), dim)
#ifdef OCEANO_WITH_TRACERS
    , fe_tracer(FE_DGQ<dim>(fe_degree), n_tra)
#endif
    , mapping(1)
    , dof_handler_height(triangulation)
    , dof_handler_discharge(triangulation)
    , dof_handler_tracer(triangulation)
    , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    , oceano_operator(param, bc, timer)
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
  // Having the base mesh in place (including the manifolds set by
  // GridGenerator::channel_with_cylinder()), we can then perform the
  // specified number of global refinements, create the unknown numbering from
  // the DoFHandler, and hand the DoFHandler and Mapping objects to the
  // initialization of the OceanoOperator.
  //
  // Furthermore, the shallow water equations may have a source term, which contains bottom
  // friction, wind stress ... Friction force, as other terms coming from
  // boundary conditions, depends on an external and given data, e.g the friction coefficient.
  // More in general these data may be spatially and time varying. Wind forcing or bathymetry
  // can be other examples. Time and space functions are represented in deal.ii with a Function
  // class. Here we set a pointer to Functions that defines such external data, both for
  // the boundary conditions and for data associated to it.
  template <int dim, int n_tra>
  void OceanoProblem<dim, n_tra>::make_grid()
  {
    oceano_operator.bc->set_problem_data(std::make_unique<ICBC::ProblemData<dim>>(prm));

    // The class GridIn can read many different mesh formats from a 
    // file from disk. In order to read a grid from a file, we generate an object 
    // of data type GridIn and associate the triangulation to it (i.e. we tell 
    // it to fill our triangulation object when we ask it to read the file). 
    // Then we open the respective file and initialize the triangulation with 
    // the data in the file. The path to the file is reconstructed from the current
    // directory, that means that the mesh file must exist in the same directory where
    // the executable is lunched.
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
  
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == NULL)
      ExcInternalError();
    std::string current_working_directory(cwd);
    std::string slash("/");

    prm.enter_subsection("Mesh & geometry parameters");
    const std::string file_msh = prm.get("Mesh_filename");
    const unsigned int n_global_refinements = prm.get_integer("Number_of_refinements");
    prm.leave_subsection();

    std::ifstream f(current_working_directory+slash+file_msh);
    pcout << "Reading mesh file: " << file_msh << std::endl;
    gridin.read_msh(f);
 
    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Initial number of cells: " << std::setw(8) << triangulation.n_global_active_cells()
          << std::endl;

    triangulation.refine_global(n_global_refinements);

    pcout << "Number of cells after global refinement: " << std::setw(8)
          << triangulation.n_global_active_cells() << std::endl;
    pcout.get_stream().imbue(s);

    oceano_operator.bc->set_boundary_conditions();
  }



  // The degrees of freedom and the solution vector needs to be
  // initialized. We take advantage of this function to initialize
  // also static data (typically computed at the quadrature points),
  // that does not change during the simulation.
  template <int dim, int n_tra>
  void OceanoProblem<dim, n_tra>::make_dofs()
  {
    dof_handler_height.distribute_dofs(fe_height);
    dof_handler_discharge.distribute_dofs(fe_discharge);
#ifdef OCEANO_WITH_TRACERS
    dof_handler_tracer.distribute_dofs(fe_tracer);
#endif
    oceano_operator.reinit(mapping, dof_handler_height,
                                    dof_handler_discharge,
                                    dof_handler_tracer);
    oceano_operator.initialize_vector(solution_height, 0);
    oceano_operator.initialize_vector(solution_discharge, 1);
#ifdef OCEANO_WITH_TRACERS
    oceano_operator.initialize_vector(solution_tracer, 2);
#endif
    oceano_operator.initialize_data();
  }



  // Given the initial condition class and some other structure,
  // we initialize the solution vectors and the postprocessed velocity.
  template <int dim, int n_tra>
  void OceanoProblem<dim, n_tra>::make_ic(
    const ICBC::Ic<dim, 1+dim+n_tra>            ic,
    LinearAlgebra::distributed::Vector<Number> &postprocess_velocity)
  {
    oceano_operator.project_hydro(
      ic, solution_height, solution_discharge);
#ifdef OCEANO_WITH_TRACERS
    oceano_operator.project_tracers(
      ic, solution_tracer);
#endif
    oceano_operator.evaluate_velocity_field(
      solution_height,
      solution_discharge,
      postprocess_velocity);
  }



  // This function takes care of the adaptive mesh refinement. The tasks this
  // function performs are the classical one of AMR: estimate the error, mark the
  // cells for refinement/coarsening, execute the remeshing and transfer the solution
  // onto the new grid.
  // The preprocessor point out that we have implemented the AMR only with P4est, a tool that
  // handle mesh refinement on distributed architecture. Without a deal.ii version compiled
  // with P4est mesh refinement is not active and a warning system is raised.
  // We have a look to each task into more details.
  template <int dim, int n_tra>
  void OceanoProblem<dim, n_tra>::refine_grid(
    const Amr::AmrTuner                        &amr_tuner,
    LinearAlgebra::distributed::Vector<Number> &postprocess_velocity)
  {
    {
      TimerOutput::Scope t(timer, "amr - remesh + remap");

#ifdef DEAL_II_WITH_P4EST
      // We estimate the error. Since this operation is case-dependent it is left
      // to a specific class and to its member function `estimate_error`.
      // This computes a cellwise error based on all the components of the solution.
      // To reduce at the miniumum the calibration of the refinement/coarsen thresholds,
      // we use dimensionless thresholds and we multiply them by the maximum error into
      // the domain.
      Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

      std::vector<DoFHandler<dim>*> dof_handlers;
      dof_handlers.push_back(&dof_handler_height);
      dof_handlers.push_back(&dof_handler_discharge);

      std::vector<FESystem<dim>*> finite_elements;
      finite_elements.push_back(&fe_height);
      finite_elements.push_back(&fe_discharge);

      amr_tuner.estimate_error<dim,Number>(finite_elements,
                                           dof_handlers,
                                           {solution_height,
                                           postprocess_velocity,
                                           solution_tracer},
                                           *oceano_operator.bc->problem_data,
                                           estimated_error_per_cell);

      float max_error = estimated_error_per_cell.linfty_norm();
      max_error = Utilities::MPI::max(max_error, MPI_COMM_WORLD);

      // Next we find out which cells to refine/coarsen: we use two functions from
      // a class that implements several different algorithms to refine a triangulation
      // based on cell-wise error indicators. This function are very simple: mark for
      // refinement if the error is above a given threshold, mark for coarsening if the
      // error is below another minimal threshold.
      // We control the minimum mesh size thanks to two parameters, set by the user:
      // the maximum level of refinement and the minimum mesh size. Above these thresholds
      // a cell cannot be refined. A successive intermediate step is
      // necessary to make sure that no two cells are adjacent with a refinement level
      // differing with more than one.
      GridRefinement::refine(
        triangulation, estimated_error_per_cell, amr_tuner.threshold_refinement * max_error);
      GridRefinement::coarsen(
        triangulation, estimated_error_per_cell, amr_tuner.threshold_coarsening * max_error);

      const unsigned int max_grid_level = amr_tuner.max_level_refinement;
      const unsigned int min_grid_size = amr_tuner.min_mesh_size;
      if (triangulation.n_levels() > max_grid_level)
        for (const auto &cell :
             triangulation.active_cell_iterators_on_level(max_grid_level))
          cell->clear_refine_flag();
      if (min_grid_size > 0.)
        for (const auto &cell :
             triangulation.active_cell_iterators())
          if (cell->minimum_vertex_distance() < min_grid_size)
            cell->clear_refine_flag();

      triangulation.prepare_coarsening_and_refinement();

      // As part of mesh refinement we need to transfer the solution vectors from
      // the old mesh to the new one. To this end we use the SolutionTransfer class and
      // the solution vectors that should be transferred to the new grid. Consequently, we
      // we have to initialize a SolutionTransfer object by attaching it to the old
      // DoF handler. We then prepare the data vector containing the old solution for
      // refinement.
      parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<Number>>
        solution_transfer_height(dof_handler_height);
      solution_transfer_height.prepare_for_coarsening_and_refinement(solution_height);

      parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<Number>>
        solution_transfer_discharge(dof_handler_discharge);
      solution_transfer_discharge.prepare_for_coarsening_and_refinement(solution_discharge);

#ifdef OCEANO_WITH_TRACERS
      parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<Number>>
        solution_transfer_tracer(dof_handler_tracer);
      solution_transfer_tracer.prepare_for_coarsening_and_refinement(solution_tracer);
#endif

      // We actually do the refinement and recreate the DoF structure on the new grid.
      triangulation.execute_coarsening_and_refinement();
      make_dofs();

      // We transfer the solution vectors between the two different grids. We initialize
      // a temporary vector to store the interpolated solution. Please note that in parallel
      // computations the interpolation operates only on locally owned dofs. We thus zero out
      // the ghost dofs of the source vector and after the interpolation we need to syncronize
      // the ghost dofs owned on other processor with an update.
      LinearAlgebra::distributed::Vector<Number> transfer_height;
      transfer_height.reinit(solution_height);
      transfer_height.zero_out_ghost_values();
      solution_transfer_height.interpolate(transfer_height);
      transfer_height.update_ghost_values();

      LinearAlgebra::distributed::Vector<Number> transfer_discharge;
      transfer_discharge.reinit(solution_discharge);
      transfer_discharge.zero_out_ghost_values();
      solution_transfer_discharge.interpolate(transfer_discharge);
      transfer_discharge.update_ghost_values();

      solution_height = transfer_height;
      solution_discharge = transfer_discharge;

#ifdef OCEANO_WITH_TRACERS
      LinearAlgebra::distributed::Vector<Number> transfer_tracer;
      transfer_tracer.reinit(solution_tracer);
      transfer_tracer.zero_out_ghost_values();
      solution_transfer_tracer.interpolate(transfer_tracer);
      transfer_tracer.update_ghost_values();

      solution_tracer = transfer_tracer;
#endif
      postprocess_velocity.reinit(solution_discharge);
      oceano_operator.evaluate_velocity_field(solution_height,
                                              solution_discharge,
                                              postprocess_velocity);
#else
    Assert(amr_tuner.max_level_refinement > 0
            || amr_tuner.remesh_tick < 10000000000.,
           ExcInternalError());
#endif
    }
  }



  // This method collects all the model outputs (to screen and to file).
  // The input argument is the preprocessor class that contains useful
  // tools that help in the outputting.
  // We examine the outputs one by one. 
  //
  // At the beginning, we postprocess some variables depending on the model
  // class request. The call `evaluate_postprocess_field` do the job here.
  template <int dim, int n_tra>
  void OceanoProblem<dim, n_tra>::output_results(
    Postprocessor                                    &postprocessor,
    const unsigned int                                result_number,
    std::map<unsigned int, Point<dim>>               &dof_points,
    const LinearAlgebra::distributed::Vector<Number> &postprocess_velocity)
  {
    std::vector<LinearAlgebra::distributed::Vector<Number>>
      postprocess_scalar_variables(postprocessor.n_postproc_vars-dim, solution_height);
    oceano_operator.evaluate_postprocess_field(dof_handler_height,
                                               solution_height,
                                               dof_points,
                                               postprocess_scalar_variables);

    if (postprocessor.do_solution)
    {
      TimerOutput::Scope t(timer, "output solution");

      // We first let the Oceano operator compute the errors of the
      // numerical results. More precisely, we compute the error against the
      // a reference result (analytical or fine mesh computation), whereas we
      // compute the deviation against a background field if no reference result
      // is provided
      const std::array<double,2> errors_hydro =
        oceano_operator.compute_errors_hydro(
          ICBC::ExactSolution<dimension, n_variables>(time,prm),
          solution_height, solution_discharge);
#ifdef OCEANO_WITH_TRACERS
      const std::array<double,1> errors_tracers =
        oceano_operator.compute_errors_tracers(
          ICBC::ExactSolution<dimension, n_variables>(time,prm), solution_tracer);
#endif

      const std::string quantity_name =
        postprocessor.do_error == 1 ? "error" : "norm";

      std::vector<std::string> vars_names = oceano_operator.model.vars_name;

      pcout << "Time:"     << std::setw(8) << std::setprecision(3) << time
            << ", cells: " << std::setw(8) << triangulation.n_global_active_cells()
            << ", dt: "    << std::setw(8) << std::setprecision(2) << time_step
            << ", " << quantity_name  << " " + vars_names[0] + ": "
            << std::setprecision(4) << std::setw(10) << errors_hydro[0]
            << ", " + vars_names[1] + ": " << std::setprecision(4)
            << std::setw(10) << errors_hydro[1];

#ifdef OCEANO_WITH_TRACERS
      pcout << ", "+ vars_names[dim+0+1] + ":" << std::setprecision(4)
            << std::setw(10) << errors_tracers[0];
#endif
      pcout << std::endl;

      // The next step is to create a visualization of the results.
      // Once we have defined all quantities to be written, we
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
      std::vector<std::string> names_postproc = postprocessor.get_names();
      std::vector<std::string> names_vector;
      for (unsigned int d = 0; d < dim; ++d)
        names_vector.emplace_back(names_postproc[d]);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation_postproc = postprocessor.get_data_component_interpretation();
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation_vector;
      for (unsigned int d = 0; d < dim; ++d)
        interpretation_vector.push_back(interpretation_postproc[d]);

      DataOut<dim>  data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_dof_handler(dof_handler_height);
      {
        data_out.add_data_vector(dof_handler_height,
                                 solution_height,
                                 vars_names[0]);

        data_out.add_data_vector(dof_handler_discharge,
                                 postprocess_velocity,
                                 names_vector,
                                 interpretation_vector);

#ifdef OCEANO_WITH_TRACERS
        // lrp: not nice the next if statement
        std::vector<std::string> names_tracer;
        for (unsigned int t = 0; t < n_tra; ++t)
          names_tracer.emplace_back(vars_names[1+dim+t]);
        if (n_tra>1)
          data_out.add_data_vector(dof_handler_tracer,
                                   solution_tracer,
                                   names_tracer);
        else data_out.add_data_vector(dof_handler_tracer,
                                     solution_tracer,
                                     vars_names[1+dim]);
#endif

        for (unsigned int v = 0; v < postprocessor.n_postproc_vars-dim; ++v)
          data_out.add_data_vector(dof_handler_height,
                                   postprocess_scalar_variables[v],
                                   {names_postproc[dim+v]},
                                   {interpretation_postproc[dim+v]});
      }

      // For the analytical solution cases, we also perform another projection of
      // the analytical solution and print the difference between that field and
      // the numerical solution.
      LinearAlgebra::distributed::Vector<Number> reference_height;
      LinearAlgebra::distributed::Vector<Number> reference_discharge;
      LinearAlgebra::distributed::Vector<Number> reference_tracer;

      if (postprocessor.do_error && dim == 2)
        {
          reference_height.reinit(solution_height);
          reference_discharge.reinit(solution_discharge);
          oceano_operator.project_hydro(
            ICBC::ExactSolution<dimension, n_variables>(time,prm),
            reference_height, reference_discharge);
          reference_height.sadd(-1., 1, solution_height);
          reference_discharge.sadd(-1., 1, solution_discharge);

          std::vector<std::string> names;
          for (unsigned int d = 0; d < dim; ++d)
            names.emplace_back("error_"+ vars_names[d+1]);

          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation;
          for (unsigned int d = 0; d < dim; ++d)
            interpretation.push_back(
              DataComponentInterpretation::component_is_part_of_vector);

          data_out.add_data_vector(dof_handler_height,
                                   reference_height,
                                   "error_"+ vars_names[0],
                                   {DataComponentInterpretation::component_is_scalar});
          data_out.add_data_vector(dof_handler_discharge,
                                   reference_discharge,
                                   names,
                                   interpretation);
        }

//      //lrp: reconnect this when postproc will work with mpi
//      // commented because it is conflicting with fedegree=0
//      Vector<double> mpi_owner(triangulation.n_active_cells());
//      mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
//      data_out.add_data_vector(mpi_owner, "owner");

      data_out.build_patches(mapping,
                             fe_height.degree,
                             DataOut<dim>::curved_inner_cells);

      const std::string filename =
        postprocessor.output_filename + "_"
        + Utilities::int_to_string(result_number, 3) + ".vtu";
      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
    }

    if (postprocessor.do_pointHistory)
    {
      // PointHistory tackles the overhead of plotting time series of solution values
      // at specific points on the mesh. The user specifies the points which the solution
      // should be monitored at ahead of time, as well as giving each solution vector
      // that they want to record a mnemonic name. Then, for each step the user calls
      // `VectorTools::point_values` to store the data from each time step, and the
      // class extracts data for the requested points to store it.
      // What is hide in `VectorTools::point_values` is absolutely non-trivial. In fact
      // this need to perform two operations. Determine the cell and the reference
      // position within the cell for a given point. Then evaluate the
      // finite element solution. For a distributed mesh deal.II performs a
      // two-level-search approach based on bounding-boxes, r-trees and a minimization.
      // We cannot cover this topic here, we only mention that the search should be as
      // fast as possible becouse it must be repeated at any tringulation change, to be
      // compatible with adaptive mesh refinement.
      // The search algorithm described above is implemented in the cache class
      // `Utilities::MPI::RemotePointEvaluation`. Note that the cache is reinitialized
      // manually to take into account the latest triangulation. The call to
      // `VectorTools::point_values` is the cheapest one because it only evaluates the
      // finite element solution with the determined information.
      TimerOutput::Scope t(timer, "output point history");

      postprocessor.cache.reinit(postprocessor.point_vector, triangulation, mapping);

      const auto history_solution_height = VectorTools::point_values<1>(
        postprocessor.cache,
        dof_handler_height,
        solution_height);
      postprocessor.pointHistory_data[time].push_back( history_solution_height );

      for (unsigned int d = 0; d < dim; ++d)
        {
          auto history_postproc_vector = VectorTools::point_values<1>(
              postprocessor.cache,
              dof_handler_discharge,
              postprocess_velocity,
              VectorTools::EvaluationFlags::avg, d);
          postprocessor.pointHistory_data[time].push_back( history_postproc_vector );
        }

      for (unsigned int v = 0; v < postprocessor.n_postproc_vars-dim; ++v)
        {
          auto history_postproc_scalar = VectorTools::point_values<1>(
            postprocessor.cache,
            dof_handler_height,
            postprocess_scalar_variables[v]);
          postprocessor.pointHistory_data[time].push_back( history_postproc_scalar );
        }
    }

    if (postprocessor.do_integralHistory)
    {
      // This is an analogue to the previous but it outputs the integrals of a variable
      // instead of puctual values. For now it works only with the water depth to compute
      // the global mass balance of the scheme which is a delicate issue for coastal models,
      // both when mesh adaptation and/or wetting and drying is involved. The mass
      // balance is strictly related to the scheme, here we do almost nothing since much of
      // the work has been done into the space discretization class, `OceanoOperator`.
      postprocessor.integralHistory_data[time] =
        {{oceano_operator.check_mass_cell_integral, oceano_operator.check_mass_boundary_integral}};
    }

    if (postprocessor.do_meshsize)
    {
      // Outputting the meshsize is useful to keep under control the timestep.
      Vector<float> meshsize(triangulation.n_active_cells());
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          meshsize(cell->active_cell_index()) = cell->minimum_vertex_distance();

      DataOut<dim>  data_out;

      DataOutBase::VtkFlags flags;
      data_out.set_flags(flags);

      data_out.attach_triangulation(triangulation);
      data_out.add_data_vector(meshsize, "minimum_vertex_distance");
      data_out.build_patches();

      const std::string filename =
        postprocessor.output_filename + "_meshsize.vtu";
      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

      postprocessor.do_meshsize = false;
    }
  }



  // The OceanoProblem::run() function puts all pieces together. It starts off
  // by calling the function that creates the mesh and sets up data structures,
  // and then initializing the time integrator and the two temporary vectors of
  // the Runge--Kutta integrator. We call these vectors `rk_register_1` and
  // `rk_register_2`, and use the first vector to represent the quantity
  // $\mathbf{r}_i$ and the second one for $\mathbf{k}_i$ in the formulas for
  // the Runge--Kutta scheme outlined in the introduction. Before we start the
  // time loop, we compute the time step size by the
  // `OceanoOperator::compute_cell_transport_speed()` function. For reasons of
  // comparison, we compare the result obtained there with the minimal mesh
  // size and print them to screen.
  template <int dim, int n_tra>
  void OceanoProblem<dim, n_tra>::run()
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
      pcout << "Number of quadrature points along a line: " << std::setw(4)
            << n_q_points_1d << std::endl;
      pcout << "Number of quadrature points in a cell   : " << std::setw(4)
            << n_q_points_1d * n_q_points_1d  << std::endl;
    }

    make_grid();
    make_dofs();

    // It is the turn of the constructor of the AmrTuner. We have collected
    // all the  tuning parameters for the AMR in a separate class in order
    // to keep things in order. This class is controlled via a preprocessor
    // for the choice of the error estimate.
    Amr::AmrTuner amr_tuner(prm);

    // We introduce two auxiliary vectors to store the velocity and
    // the dof position. Velocities are derived from discharges and are
    // ubiquious (output, grid refinement, eddy viscosity/diffusivity).
    // We prefer to store them, rather than recompute locally in each code subroutine.
    // Moreover, having a solution vector also for the velocity, we can exploit
    // fast access to the projected velocity at the quadrature points thanks to
    // the capabilities of FEEvaluation.
    LinearAlgebra::distributed::Vector<Number> postprocess_velocity;
    postprocess_velocity.reinit(solution_discharge);
    std::map<unsigned int, Point<dim>> dof_points =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler_height);

    // We set the initial conditions. This is done in a scope to free
    // the memory associated to the data stored in the initial condition class.
    // For initial value problems we may proceed with mesh
    // refinement to the initial solution. The mesh is pushed
    // to the maximum refinement level since the start.
    {
      TimerOutput::Scope t(timer, "compute initial solution");
      ICBC::Ic<dimension, n_variables> ic(prm);
      make_ic(ic, postprocess_velocity);

      for (unsigned int lev = 0; lev < amr_tuner.max_level_refinement; ++lev)
        {
          refine_grid(amr_tuner, postprocess_velocity);
          make_ic(ic, postprocess_velocity);
        }
    }

    // For a general Runge--Kutta schemes with Butcher tableau we need a table
    // to store the updated residual vector at each stage. We blend one of the
    // two auxiliary vectors with this table in a single entity, so at the end
    // `rk_register_2` has size equal to the number of stage plus one. For
    // the Additive Runge--Kutta the number of stored residuals is doubled
    // because of the explicit and the implicit residuals. We blend the two residuals
    // per stage in a table, that should have size equal to twice the number of stages plus one.
    // However the last stage has common residuals so we can save one residual and
    // the final dimension is just twice the number of stages. Please note that, for now, ARK
    // is used only for the discharge equation.
#if defined TIMEINTEGRATOR_EXPLICITRUNGEKUTTA
    const TimeIntegrator::ExplicitRungeKuttaIntegrator integrator(rk_scheme);

    LinearAlgebra::distributed::Vector<Number> rk_register_height_1;
    LinearAlgebra::distributed::Vector<Number> rk_register_discharge_1;
    LinearAlgebra::distributed::Vector<Number> rk_register_tracer_1;
    std::vector<LinearAlgebra::distributed::Vector<Number>> rk_register_height_2(
                                                              integrator.n_stages()+1);
    std::vector<LinearAlgebra::distributed::Vector<Number>> rk_register_discharge_2(
                                                              integrator.n_stages()+1);
    std::vector<LinearAlgebra::distributed::Vector<Number>> rk_register_tracer_2(
                                                              integrator.n_stages()+1);

    integrator.reinit(solution_height, solution_discharge, solution_tracer,
                      rk_register_height_1,
                      rk_register_discharge_1,
                      rk_register_tracer_1,
                      rk_register_height_2,
                      rk_register_discharge_2,
                      rk_register_tracer_2);

#elif defined TIMEINTEGRATOR_ADDITIVERUNGEKUTTA
    const TimeIntegrator::AdditiveRungeKuttaIntegrator integrator(rk_scheme);

    LinearAlgebra::distributed::Vector<Number> rk_register_height_1;
    LinearAlgebra::distributed::Vector<Number> rk_register_discharge_1;
    LinearAlgebra::distributed::Vector<Number> rk_register_tracer_1;
    std::vector<LinearAlgebra::distributed::Vector<Number>> rk_register_height_2(
                                                              integrator.n_stages()+1);
    std::vector<LinearAlgebra::distributed::Vector<Number>> rk_register_discharge_2(
                                                              2*integrator.n_stages());
    std::vector<LinearAlgebra::distributed::Vector<Number>> rk_register_tracer_2(
                                                              integrator.n_stages()+1);

    integrator.reinit(solution_height, solution_discharge, solution_tracer,
                      rk_register_height_1,
                      rk_register_discharge_1,
                      rk_register_tracer_1,
                      rk_register_height_2,
                      rk_register_discharge_2,
                      rk_register_tracer_2);
#endif

    // In the following, we output some statistics about the problem. Because we
    // often end up with quite large numbers of cells or degrees of freedom, we
    // would like to print them with a comma to separate each set of three
    // digits. This can be done via "locales", although the way this works is
    // not particularly intuitive. step-32 explains this in slightly more
    // detail.
    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Number of cells after local  refinement: " << std::setw(8)
          << triangulation.n_global_active_cells() << std::endl;
    pcout << "Initial number of degrees of freedom: " << dof_handler_height.n_dofs()
             + dof_handler_discharge.n_dofs() + dof_handler_tracer.n_dofs()
          << " ( = " << ( 1 + dim + n_tra ) << " [vars] x "
          << triangulation.n_global_active_cells() << " [cells] x "
          << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
          << std::endl;
    pcout.get_stream().imbue(s);

    // Next off is the Courant number
    // that scales the time step size in terms of the formula $\Delta t =
    // \text{CFL}\frac{h}{(\|\mathbf{u} +
    // c)_\text{max}}$, as well as a selection of a few explicit Runge--Kutta
    // methods.
    // The Courant number depends on the number of stages, on the degree of the finite
    // element approximation as well as on the specific time integrator. The correct choice
    // of the courant number is left to the user that must specify it in the parameter file.
    // A possible expression can be $CFL=\frac{1}{p^{1.5}}*n_{stages}$. For TODO (Cockburn and Shu)
    // $CFL=\frac{1}{2p+1}$.
    prm.enter_subsection("Time parameters");
    const double final_time = prm.get_double("Final_time");
    const double courant_number = prm.get_double("CFL");
    prm.leave_subsection();

    double min_vertex_distance = std::numeric_limits<double>::max();
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        min_vertex_distance =
          std::min(min_vertex_distance, cell->minimum_vertex_distance());
    min_vertex_distance =
      Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

    time_step = courant_number  /
      oceano_operator.compute_cell_transport_speed(solution_height,
                                                   solution_discharge);
    pcout << "Time step size: " << time_step
          << ", initial minimal h: " << min_vertex_distance
          << ", initial transport scaling: "
          << 1. / oceano_operator.compute_cell_transport_speed(solution_height,
                                                               solution_discharge)
          << "\n" << std::endl;

    // We have moved the constructor of the `Postprocessor` class in this
    // top level `run()`, so that we can pass it by reference to many
    // functions that postprocess the solution. For now there the postprocess
    // consists of only one function that print the solution and compute an
    // error or norm of the solution.
    Postprocessor postprocessor(prm, oceano_operator.model.postproc_vars_name);

    output_results(postprocessor, 0, dof_points, postprocess_velocity);

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
    unsigned int timestep_number = 0;

    while (time < final_time - 1e-12)
      {
        ++timestep_number;

        if (timestep_number % 5 == 0)
          time_step =
            courant_number /
            Utilities::truncate_to_n_digits(
              oceano_operator.compute_cell_transport_speed(solution_height,
                                                           solution_discharge), 3);

        {
          TimerOutput::Scope t(timer, "rk time stepping total");
          integrator.perform_time_step(oceano_operator,
                                       time,
                                       time_step,
                                       solution_height,
                                       solution_discharge,
                                       solution_tracer,
                                       postprocess_velocity,
                                       rk_register_height_1,
                                       rk_register_discharge_1,
                                       rk_register_tracer_1,
                                       rk_register_height_2,
                                       rk_register_discharge_2,
                                       rk_register_tracer_2);

          oceano_operator.evaluate_velocity_field(solution_height,
                                                  solution_discharge,
                                                  postprocess_velocity);
        }

        time += time_step;

        if (static_cast<int>(time / amr_tuner.remesh_tick) !=
              static_cast<int>((time - time_step) / amr_tuner.remesh_tick))
          {
            refine_grid(amr_tuner, postprocess_velocity);

            integrator.reinit(solution_height, solution_discharge, solution_tracer,
              rk_register_height_1,
              rk_register_discharge_1,
              rk_register_tracer_1,
              rk_register_height_2,
              rk_register_discharge_2,
              rk_register_tracer_2);
          }

        postprocessor.do_solution =
          (static_cast<int>(time / postprocessor.solution_tick) !=
            static_cast<int>((time - time_step) / postprocessor.solution_tick) ||
              time >= final_time - 1e-12);
        postprocessor.do_pointHistory =
          (static_cast<int>(time / postprocessor.pointHistory_tick) !=
            static_cast<int>((time - time_step) / postprocessor.pointHistory_tick));
#ifdef OCEANO_WITH_MASSCONSERVATIONCHECK
        postprocessor.do_integralHistory = postprocessor.do_solution;
#endif
        if (postprocessor.do_solution || postprocessor.do_pointHistory)
          output_results(
            postprocessor,
            static_cast<unsigned int>(std::round(time / postprocessor.solution_tick)),
            dof_points,
            postprocess_velocity);
      }

    postprocessor.write_history_gnuplot();

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
      // read are then handed over to an instance of the OceanoProblem class:
      IO::ParameterHandler prm;
      IO::ParameterReader  param(prm);      
      param.read_parameters(argv[2]);

      // The boundary condition class is the only class declared as a pointer.
      // This is because we want to realize run-time polymrophism. In the base class 
      // there are defined a bunch of members that are needed for all the test cases. 
      // Next, the pointer is allocated as a derived class specific 
      // to the test-case which will override the boundary conditions. 
      // The pointer is easily pass as argument to the `OceanoProblem` class.
      ICBC::BcBase<dimension, n_variables> *bc;
      // The switch between the different test-cases is realized with Preprocessor keys.
      // The choice to use a Preprocessor also for the boundary conditions is because we
      // use it for the initial condition and we have mantained the same directive 
      // to easily switch both initial and boundary conditions depending on the test-case.
#if defined ICBC_ISENTROPICVORTEX
      bc = new ICBC::BcIsentropicVortex<dimension, n_variables>(prm);
#elif defined ICBC_FLOWAROUNDCYLINDER
      bc = new ICBC::BcFlowAroundCylinder<dimension, n_variables>(prm);
#elif defined ICBC_IMPULSIVEWAVE
      bc = new ICBC::BcImpulsiveWave<dimension, n_variables>(prm);
#elif defined ICBC_SHALLOWWATERVORTEX
      bc = new ICBC::BcShallowWaterVortex<dimension, n_variables>(prm);
#elif defined ICBC_STOMMELGYRE
      bc = new ICBC::BcStommelGyre<dimension, n_variables>(prm);
#elif defined ICBC_LAKEATREST
      bc = new ICBC::BcLakeAtRest<dimension, n_variables>(prm);
#elif defined ICBC_TRACERADVECTION
      bc = new ICBC::BcTracerAdvection<dimension, n_variables>(prm);
#elif defined ICBC_CHANNELFLOW
      bc = new ICBC::BcChannelFlow<dimension, n_variables>(prm);
#elif defined ICBC_REALISTIC
      bc = new ICBC::BcRealistic<dimension, n_variables>(prm);
#else
      Assert(false, ExcNotImplemented());
      return 0.;
#endif 
      
      // The OceanoProblem class takes as argument the only two classes that have been
      // previously defined and that are filled at runtime. One is the pointer class
      // for the boundary conditions, the other is a reference to the parameter
      // class to read the paramteres from an external config file.     
      OceanoProblem<dimension, n_tracers> oceano_problem(prm, bc);
      oceano_problem.run();
      
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
