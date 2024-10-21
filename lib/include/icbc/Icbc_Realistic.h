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
 *         Luca Arpaia,        2023
 */
#ifndef ICBC_REALISTIC_HPP
#define ICBC_REALISTIC_HPP

#include <deal.II/base/function.h>
// The following files include the oceano libraries
#include <icbc/IcbcBase.h>
/**
 * Namespace containing the initial and boundary conditions.
 */

namespace ICBC
{

  // These are the the initial and boundary conditions classes for
  // realistic coastal experiments.
  // With respect to other ICBC derived classes, the classes contained
  // in this file are not specific to a test case but rather generic.
  // In fact all initial solution and boundary forcings are all
  // imported by external files that can be modified outside the
  // oceano model.

  using namespace dealii;

  // We define global parameters that help in the definition of the
  // initial and boundary conditions. For this test we just need the
  // discharge and and the Manning coefficient:
  constexpr double z0      = 0.0;
  constexpr double q0      = 50./30.;

  // @sect3{Equation data}
  
  // Next we implement a few classes that are useful to represent functions
  // for the initial and boundary conditions. Deal.II has a class `Function`
  // which returns function of space and time, thus we simply create a
  // derived class.
  // We specialize a few different derived Function classes to handle data.
  // What changes essentially is the number of components, the type of
  // function, i.e. analytical or from observations saved into file, and
  // the independent variable, i.e is the function time or space dependent?
  // We review these Function classes.

  // First we need to handle the problem surface data. The number of
  // components for the data is fixed to `dim+3=5` scalar quantities.
  // We list them here:
  // begin{itemize}
  // \item the first component is the bathymetry.
  // \item the second is the bottom friction coefficient.
  // \item the third and fourth components are the cartesian components
  // of the wind velocity (in order, eastward and northward).
  // \item The fifth one is the Coriolis parameter.
  // end{itemize}
  // Surface data in general depends on both time and space, although for
  // now we have only implemented the space dependence. As you may see the
  // time variable is never retrieved.
  //
  // In some simplified context we may also want to assign constant value
  // to the data. The parameter handler class may seems redundant but it is
  // not! Constants that appears in you data may be easily recovered from
  // the configuration file.
  //
  // More often the data is space-varying and need to be interpolated from
  // a reference field given on a fine mesh. Deal.ii has a special `Function`
  // class that computes the values by (bi-, tri-)linear interpolation from
  // a set of point data that are arranged on a uniformly spaced tensor
  // product mesh. It is called `InterpolatedUniformGridData`. Considering
  // the two-dimensional case, let there be points $x_0,\,...,\,x_{K−1}$ that
  // result from a uniform subdivision of the interval $\[x_0,x_{K−1}\]$
  // into $K−1$ sub-intervals of size $\Delta x=\frac{x_{K−1}−x_0}/{K−1}$,
  // and similarly $y_0,\,...,\,y_{L−1}$. Also consider bathymetry data
  // $z_{kl}$ defined at point $\left(x_k,y_l\right)^T$, then evaluating the
  // function at a point $x=(x,y,z)$ will find the box so that
  // $x_k\le x \le x_{k+1},\, y_l \le y \le y_{l+1}$ and do a bilinear
  // interpolation of the data on this cell. Let us talk about the constructor
  // of this class. It takes as argument the interval_endpoints, the number
  // of subintervals in each coordinate direction and a `dim`-dimensional
  // table of data at each of the mesh points defined by the
  // coordinate arrays above. We have also used an auxiliary class that reads
  // the file. Thanks to this class we open, read and store the header lines,
  // then read the data values and close the file.
  //
  // Last, a note on the `value()` method that actually do the job,
  // specialize the base class method and return all the external data
  // necessary to complete the computation.
  template <int dim>  
  class ProblemData : public Function<dim>
  {
  public:
    ProblemData(IO::ParameterHandler &prm);
    ~ProblemData(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    std::string bathymetry_filename(IO::ParameterHandler &prm) const;
    IO::TxtDataReader<dim> bathymetry_data_reader;
    const Functions::InterpolatedUniformGridData<dim> bathymetry_data;

    double cf;
  };

  template <int dim>
  ProblemData<dim>::ProblemData(IO::ParameterHandler &prm)
    : Function<dim>(dim+3)
    , bathymetry_data_reader(bathymetry_filename(prm))
    , bathymetry_data(
        bathymetry_data_reader.endpoints,
        bathymetry_data_reader.n_intervals,
        Table<dim, double>(bathymetry_data_reader.n_intervals.front()+1,
                           bathymetry_data_reader.n_intervals.back()+1,
                           bathymetry_data_reader.get_data(bathymetry_data_reader.filename).begin()))
  {
    prm.enter_subsection("Physical constants");
    cf = prm.get_double("bottom_friction");
    prm.leave_subsection();
  }



  template <int dim>
  std::string ProblemData<dim>::bathymetry_filename(IO::ParameterHandler &prm) const
  {
    prm.enter_subsection("Input data files");
    std::string filename = prm.get("Bathymetry_filename");
    prm.leave_subsection();

    return filename;
  }

  template <int dim>
  double ProblemData<dim>::value(const Point<dim>  &x,
                                 const unsigned int component) const
  {
    if (component == 0)
      return bathymetry_data.value(x);
    else if (component == 1)
      return cf;
    else
      return 0.0;
  }



  // Apart from surface data we have the boundary conditions data.
  // Boundary data has `n_var` components at maximum. For subcritical
  // boundaries we need less conditions, thus less external data, but
  // we keep the function general so we have hard-coded `n_var`
  // components.
  // In coastal simulations of small semi-enclosed basin, the boundary
  // data can be considered constant over the boundary and only
  // time-varying.
  template <int dim, int n_vars>  
  class BoundaryData : public Function<dim>
  {
  public:
    BoundaryData(const double time,
                 std::string boundary_filename,
                 IO::ParameterHandler &prm);
    ~BoundaryData(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    IO::TxtDataReader<1> boundary_data_reader; //lrp: pay attention to this multiple dimension: dim, 1 ... put order here!
    const Functions::InterpolatedUniformGridData<1> boundary_data;
  };

  template <int dim, int n_vars>
  BoundaryData<dim, n_vars>::BoundaryData(const double          time,
                                          std::string           boundary_filename,
                                          IO::ParameterHandler &/*prm*/)
    : Function<dim>(n_vars, time)
    , boundary_data_reader(boundary_filename)
    , boundary_data(
        boundary_data_reader.endpoints,
        boundary_data_reader.n_intervals,
        Table<1, double>(boundary_data_reader.n_intervals.front()+1,
               //            boundary_data_reader.n_intervals.back()+1,
                           boundary_data_reader.get_data(boundary_filename).begin()))
  {}

  template <int dim, int n_vars>
  double BoundaryData<dim, n_vars>::value(const Point<dim>  &/*x*/,
                                          const unsigned int component) const
  {
    //const double t = this->get_time();

    Assert(dim == 2, ExcNotImplemented());
        
    Point<1> t;
    t[0] = this->get_time();

    if (component == 0)
    {
      return boundary_data.value(t);
    }
    else if (component == 1)
      return boundary_data.value(t);
    else
      return 0.;
  }



  // The class `ExactSolution` defines a reference functions that can be used
  // for diagnostic, for exmaple to measure the errors with respect ot another
  // model. Apart for the template for the
  // dimension which is in common with the base `Function` class, we have added
  // the number of variables. As seen in the introduction the free-surface is
  // available in a semi-analytical form and must be read from a file. We have
  // thus modified the constructor with two classes: a data reader class and a
  // the data class itself. Thanks to them we can read and compute the reference
  // solution with a bilinear interpolation. We do not talk more about these two
  // classes because they are discussed in detail when for the bathymetry.
  //
  // In the Oceano variables the exact solution must be given in the free-surface
  // and discharge variables. We check that the test runs in two-dimensions
  // which is consistent with the dimension of the file (otherwise it would raise
  // an error difficult to detect). The free-surface is computed by a bilinear
  // interpolation of the data read from file.
  template <int dim, int n_vars>  
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double          time,
                  IO::ParameterHandler &/*prm*/)
      : Function<dim>(n_vars, time)
/*      , data_reader(filename(prm))
      , data(
          data_reader.endpoints,
          data_reader.n_intervals,
          Table<dim, double>(data_reader.n_intervals.front()+1,
                             data_reader.n_intervals.back()+1,
                             data_reader.get_data(data_reader.filename).begin()))*/
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

/*  private:
    std::string filename(IO::ParameterHandler &prm) const;
    IO::TxtDataReader<dim> data_reader;
    const Functions::InterpolatedUniformGridData<dim> data;*/
  };  

/*  template <int dim, int n_vars>
  std::string ExactSolution<dim, n_vars>::filename(IO::ParameterHandler &prm) const
  {
    prm.enter_subsection("Input data files");
    std::string filename = prm.get("Exact_solution_filename");
    prm.leave_subsection();

    return filename;
  }*/

  template <int dim, int n_vars>
  double ExactSolution<dim, n_vars>::value(const Point<dim> & /*x*/,
                                           const unsigned int component) const
  {
//    //const double t = this->get_time();

    Assert(dim == 2, ExcNotImplemented());
    if (component == 0)
//      return data.value(x);
      return z0;
    else if (component == 1)
      return 0.;
    else
      return 0.;
  }



  // The `Ic` and `Bc` classes define the initial/boundary condition for the
  // test-case. They are very similar in the templates and the constructor.
  // They both take as argument the parameter class and they stored it
  // internally. This means that we can read the Parameter file from
  // anywhere when we are implementing ic/bc and we can access constants or
  // filenames from which the initial/boundary data depends.
  // The initial conditions are a $\zeta(0,x) = 0\,m$ and $hu(0,x) = q_0\, m^2 /s$.
  // We return either the water depth or the momentum depending on which component is
  // requested. Two sanity checks have been added. One is to
  // control that the space dimension is two (you cannot run this test in one dimension) and
  // another one on the number of variables, that for two-dimensional shallow water equation
  // is three or more (if you have tracers).
  //
  // We start with a wet channel in equilibrium with a sloping topography without any bump or
  // hill, that is with $\partial_x h=0$ in the above equation. The slope is thus controlled
  // by the friction. In case the bathyemtry coincide with such a sloping channel then we
  // should have an exact preservation of the flow.
  // If tracers are presents they are simply set to zero.
  // A supercritical inflow/outflow boundary condition is specified on the left and
  // right boundary of the domain. Top and bottom boundaries are walls.
  template <int dim, int n_vars>
  class Ic : public Function<dim>
  {
  public:
    Ic(IO::ParameterHandler &prm)
      : Function<dim>(n_vars, 0.)
    {
      prm.enter_subsection("Physical constants");
      g = prm.get_double("g");
      prm.leave_subsection();
    }
    ~Ic(){};

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  private:
    double g;
  };

  template <int dim, int n_vars>
  double Ic<dim, n_vars>::value(const Point<dim>  &/*x*/,
                                const unsigned int component) const
  {
    Assert(dim == 2, ExcNotImplemented());
    Assert(n_vars < 4, ExcNotImplemented());

    if (component == 0)
        return z0;
    else if (component == 1)
        return 0.;
    else
        return 0.;
  }



  template <int dim, int n_vars>  
  class BcRealistic : public BcBase<dim, n_vars>
  {
  public:
  
    BcRealistic(IO::ParameterHandler &prm);
    ~BcRealistic(){};

    void set_boundary_conditions() override;

  private:
    ParameterHandler &prm;
  };
  
  template <int dim, int n_vars>
  BcRealistic<dim, n_vars>::BcRealistic(IO::ParameterHandler &prm)
    : prm(prm)
  {}

  template <int dim, int n_vars>
  void BcRealistic<dim, n_vars>::set_boundary_conditions()
  {
    std::map<types::boundary_id, std::pair<std::string,std::string>> boundaryId;
    std::vector<std::string> info;
    bool found_boundary = false;

    prm.enter_subsection("Boundary conditions");
    for (unsigned int num = 1; num < 20; ++num)
      {
         std::string boundary_info = prm.get("Boundary_"+std::to_string(num));
         if (boundary_info == "There is no entry Boundary in the parameter file")
           {
             continue;
           }
         else
           {
             found_boundary = true;
           }
         boost::split(info, boundary_info, boost::is_any_of(":"));
         if (info.size() == 2)
           boundaryId[std::stoi(info[0])] = {info[1],"no_file_needed"};
         else if (info.size() == 3)
           boundaryId[std::stoi(info[0])] = {info[1],info[2]};
      }
    prm.leave_subsection();
    AssertThrow(found_boundary == true,
      ExcMessage("In the parameter file you have not specified the entry:\n"
                 "set Boundary_x = ...\n"
                 "for any boundary id.\n"
                 "Fill this entry for each boundary id"));

    for (auto &i : boundaryId)
      { 
        std::pair<std::string,std::string> type_and_filename;
        type_and_filename = i.second;
        if ( type_and_filename.first == "wall")
          this->set_wall_boundary(i.first);
        else if (type_and_filename.first == "height_inflow")
          this->set_height_inflow_boundary(
            i.first, std::make_unique<BoundaryData<dim, n_vars>>(0, type_and_filename.second, prm));
        else if (type_and_filename.first == "discharge_inflow")
          this->set_discharge_inflow_boundary(
            i.first, std::make_unique<BoundaryData<dim, n_vars>>(0, type_and_filename.second, prm));
        else if (type_and_filename.first == "absorbing_outflow")
          this->set_absorbing_outflow_boundary(
            i.first, std::make_unique<ExactSolution<dim, n_vars>>(0, prm));
      }
  }
} // namespace ICBC

#endif //ICBC_REALISTIC_HPP
