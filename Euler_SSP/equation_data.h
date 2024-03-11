/*--- Author: Giuseppe Orlando, 2024. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <cmath>

// @sect{Equation data}

// In this namespace, we declare the initial conditions,
// the velocity field
//
namespace EquationData {
  using namespace dealii;

  /*--- Polynomial degrees. We typically consider the same polynomial degree for all the variables ---*/
  static const unsigned int degree_T   = 2;
  static const unsigned int degree_rho = 2;
  static const unsigned int degree_u   = 2;

  static const double Cp_Cv = 1.4; /*--- Specific heats ratio ---*/


  // With this class defined, we declare class that describes the initial (and boundary)
  // condition for the momentum:
  //
  template<int dim>
  class Momentum: public Function<dim> {
  public:
    Momentum(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value for a single component ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Point value for the whole vector ---*/
  };

  // Constructor which simply relies on the 'Function' constructor.
  //
  template<int dim>
  Momentum<dim>::Momentum(const double initial_time): Function<dim>(dim, initial_time) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<int dim>
  double Momentum<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    /*--- Current time ---*/
    const double t           = this->get_time();

    /*--- Vortex parameters ---*/
    const double beta        = 5.0;
    const double u_infty     = 1.0;
    const double v_infty     = 1.0;
    const double x0          = 0.0;
    const double y0          = 0.0;

    /*--- Isentropic vortex description for the momentum ---*/
    const double r2          = (p[0] - u_infty*t - x0)*(p[0] - u_infty*t - x0) +
                               (p[1] - v_infty*t - y0)*(p[1] - v_infty*t - y0);
    const double factor      = beta/(2.0*numbers::PI)*std::exp(0.5*(1.0 - r2));
    const double density_log = std::log2(std::abs(1.0 - (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv*0.5*factor*factor));
    const double density     = std::exp2(density_log*(1.0/(EquationData::Cp_Cv - 1.0)));

    if(component == 0) {
      return density*(u_infty - factor*(p[1] - v_infty*t - y0));
    }
    else {
      return density*(v_infty + factor*(p[0] - u_infty*t - x0));
    }
  }

  // Put together for a vector evalutation of the momentum.
  //
  template<int dim>
  void Momentum<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < dim; ++i) {
      values[i] = value(p, i);
    }
  }


  // We do the same for the energy (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function
  //
  template<int dim>
  class Energy: public Function<dim> {
  public:
    Energy(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point evaluation ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Energy<dim>::Energy(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Energy<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    /*--- Current time ---*/
    const double t           = this->get_time();

    /*--- Vortex parameters ---*/
    const double beta        = 5.0;
    const double u_infty     = 1.0;
    const double v_infty     = 1.0;
    const double x0          = 0.0;
    const double y0          = 0.0;

    /*--- Vortex description to compute pressure and velocity along the two directions ---*/
    const double r2          = (p[0] - u_infty*t - x0)*(p[0] - u_infty*t - x0) +
                               (p[1] - v_infty*t - y0)*(p[1] - v_infty*t - y0);
    const double factor      = beta/(2.0*numbers::PI)*std::exp(0.5*(1.0 - r2));
    const double density_log = std::log2(std::abs(1.0 - (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv*0.5*factor*factor));
    const double density     = std::exp2(density_log*(1.0/(EquationData::Cp_Cv - 1.0)));
    const double pressure    = std::exp2(density_log*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)));
    const double u           = u_infty - factor*(p[1] - v_infty*t - y0);
    const double v           = v_infty + factor*(p[0] - u_infty*t - x0);

    return 1.0/(EquationData::Cp_Cv - 1.0)*pressure + 0.5*density*(u*u + v*v);
  }


  // We do the same for the density (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function
  //
  template<int dim>
  class Density: public Function<dim> {
  public:
    Density(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point evaluation ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Density<dim>::Density(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Density<dim>::value(const Point<dim>&  p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    /*--- Current time ---*/
    const double t           = this->get_time();

    /*--- Vortex parameters ---*/
    const double beta        = 5.0;
    const double u_infty     = 1.0;
    const double v_infty     = 1.0;
    const double x0          = 0.0;
    const double y0          = 0.0;

    /*--- Vortex description to compute the density ---*/
    const double r2          = (p[0] - u_infty*t - x0)*(p[0] - u_infty*t - x0) +
                                 (p[1] - v_infty*t - y0)*(p[1] - v_infty*t - y0);
    const double factor      = beta/(2.0*numbers::PI)*std::exp(0.5*(1.0 - r2));
    const double density_log = std::log2(std::abs(1.0 - (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv*0.5*factor*factor));

    return std::exp2(density_log*(1.0/(EquationData::Cp_Cv - 1.0)));
  }

} // namespace EquationData
