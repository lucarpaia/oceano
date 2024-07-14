/*--- Author: Giuseppe Orlando, 2024. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <cmath>

constexpr int my_ceil(double num) {
  return (static_cast<float>(static_cast<int>(num)) == num) ?
          static_cast<int>(num) :
          static_cast<int>(num) + ((num > 0) ? 1 : 0);
}

// @sect{Equation data}

// In this namespace, we declare initial and boundary conditions.
//
namespace EquationData {
  using namespace dealii;

  /*--- Polynomial degrees. We typically consider the same polynomial degree for all the variables ---*/
  static const unsigned int degree_h  = 1;
  static const unsigned int degree_hu = 1;
  static const unsigned int degree_hc = 1;

  static const double g = 9.81; /*--- Acceleration of gravity ---*/

  /*--- Mapping for the sake of generality (in case a curved domain or boundary appears) ---*/
  static const unsigned int degree_mapping          = 1;                                                             /*--- Mapping degree ---*/
  static const unsigned int extra_quadrature_degree = (degree_mapping == 1) ? 0 : my_ceil(0.5*(degree_mapping - 2)); /*--- Extra accuracy
                                                                                                                           for quadratures ---*/

  /*--- Number of stages of the IMEX scheme ---*/
  static const unsigned int n_stages = 3;


  // We declare now the class that describes the initial condition for the water depth.
  //
  template<int dim>
  class Depth: public Function<dim> {
  public:
    Depth(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Function evaluation ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Depth<dim>::Depth(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Depth<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double zb   = -5.0*std::exp(-0.4*(p[0] - 5.0)*(p[0] - 5.0));

    const double zeta = 10.0;

    return zeta + zb;
  }


  // We declare now the class that describes the initial condition for the discharge.
  //
  template<int dim>
  class Discharge: public Function<dim> {
  public:
    Discharge(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evaluation of the discharge for each component ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Vector evaluation of the velocity ---*/
  };

  // Constructor which relies on the 'Function' constructor.
  //
  template<int dim>
  Discharge<dim>::Discharge(const double initial_time): Function<dim>(dim, initial_time) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<int dim>
  double Discharge<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    return 0.0;
  }

  // Put together for a vector evalutation of the discharge.
  //
  template<int dim>
  void Discharge<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < dim; ++i) {
      values[i] = value(p, i);
    }
  }


  // We do the same for the tracer.
  //
  template<int dim>
  class Tracer: public Function<dim> {
  public:
    Tracer(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Function evaluation ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Tracer<dim>::Tracer(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Tracer<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    return 1.0;
  }


  // Finally, we do the same for the bathymetry.
  //
  template<int dim, typename T>
  class Bathymetry: public Function<dim, T> {
  public:
    Bathymetry(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual T value(const Point<dim>&  p,
                    const unsigned int component = 0) const override; /*--- Function evaluation ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim, typename T>
  Bathymetry<dim, T>::Bathymetry(const double initial_time): Function<dim, T>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim, typename T>
  T Bathymetry<dim, T>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const T zb = -5.0*std::exp(-0.4*(p[0] - 5.0)*(p[0] - 5.0));

    return zb;
  }

} // namespace EquationData
