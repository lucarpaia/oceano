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
 
#ifndef TIMESTEPPINGOCEANO_HPP
#define TIMESTEPPINGOCEANO_HPP

/**
 * Namespace containing the time stepping methods
 * not coded in Deal.II base/time_stepping.h
 */

namespace TimeSteppingOceano
{

  using namespace dealii;

  // In this file we have coded the timestepping method that are not available
  // in deal.II. One example are the Strong-Stability-Preserving Runge-Kutta
  // schemes. We use the similar structure of the original Deal.II timestepping
  // classes. In the following a list of the timestepping scheme that we
  // have added for Oceano:
  enum runge_kutta_method_oceano
  {
    /**
     * Forward Euler method, first order.
     */
    FORWARD_EULER,
    /**
     * Second order Runge-Kutta method coded in the Strong Stability Preserving
     * form of @cite gottlieb2001strong.
     */
    SSP_SECOND_ORDER,
    /**
     * Third order Strong Stability Preserving (SSP) Runge-Kutta method
     * (SSP time discretizations are also called Total Variation Diminishing
     * (TVD) methods in the literature, see @cite gottlieb2001strong).
     */
    SSP_THIRD_ORDER,
    /**
     * Classical fourth order Runge-Kutta method.
     */
    RK_CLASSIC_FOURTH_ORDER,
    /**
     * Second order three stage scheme of (Giraldo et al. 2012).
     */
    THREE_STAGE_SECOND_ORDER,
    /**
     * Trapezoidal-BDF2
     */
    TRAPEZOIDAL_BDF2,

    /**
     * Invalid.
     */
    invalid
  };



  // The ExplicitRungeKutta class could have been derived from RungeKutta
  // implementing a specific class of explicit methods. We did something even simpler
  // just cherry-picking what needed from the ExplicitRungeKutta class.
  // This is a constructor that initialize the Runge-Kutta tables with the
  // the coefficients and a function that retrieve such coefficients that are protected.
  // For the coefficient variable we have left the Deal.II names taken from the
  // Runge-Kutta literature `a`, `b` and `c`.
  class ExplicitRungeKutta
  {
  public:
    /**
     * Constructor. This function calls initialize(runge_kutta_method).
     */
    ExplicitRungeKutta(const runge_kutta_method_oceano method);
    /**
     * Initialize the explicit Runge-Kutta method.
     */
    void
    initialize(const runge_kutta_method_oceano method);

    /**
     * Get the coefficients of the scheme.
     * Note that here vector `a` and `b` are the conventional definition in terms of a
     * Butcher tableau.
     */
    void
    get_coefficients(std::vector<std::vector<double>> &a,
                     std::vector<double>              &b,
                     std::vector<double>              &c) const;

    protected:
    /**
     * Number of stages of the Runge-Kutta method.
     */
    unsigned int n_stages;

    /**
     * SSP tableau coefficients.
     */
      std::vector<std::vector<double>> a;
    /**
     * SSP tableau coefficients.
     */
      std::vector<double> b;
    /**
     * Butcher tableau coefficients.
     */
      std::vector<double> c;
  };



  ExplicitRungeKutta::ExplicitRungeKutta(
    const runge_kutta_method_oceano method)
  {
    ExplicitRungeKutta::initialize(method);
  }

  void
  ExplicitRungeKutta::initialize(const runge_kutta_method_oceano method)
  {

    switch (method)
      {
        case (FORWARD_EULER):
          {
            this->n_stages = 1;
            std::vector<double> tmp;
            tmp.resize(1);
            tmp[0] = 1.0;
            this->a.push_back(tmp);
            this->b.push_back(1.0);
            this->c.push_back(0.0);

            break;
          }
        case (SSP_SECOND_ORDER):
          {
            this->n_stages = 2;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            std::vector<double> tmp;
            tmp.resize(1);
            tmp[0] = 1.0;
            this->a.push_back(tmp);
            this->b.push_back(1.0 / 2.0);
            this->b.push_back(1.0 / 2.0);
            this->c.push_back(0.0);
            this->c.push_back(1.0);

            break;
          }
        case (SSP_THIRD_ORDER):
          {
            this->n_stages = 3;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            this->b.push_back(1.0 / 6.0);
            this->b.push_back(1.0 / 6.0);
            this->b.push_back(2.0 / 3.0);
            this->c.push_back(0.0);
            this->c.push_back(1.0);
            this->c.push_back(0.5);
            std::vector<double> tmp;
            //this->a.push_back(tmp);
            tmp.resize(1);
            tmp[0] = 1.0;
            this->a.push_back(tmp);
            tmp.resize(2);
            tmp[0] = 1.0 / 4.0;
            tmp[1] = 1.0 / 4.0;
            this->a.push_back(tmp);

            break;
          }
        case (RK_CLASSIC_FOURTH_ORDER):
          {
            this->n_stages = 4;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            std::vector<double> tmp;
            //this->a.push_back(tmp);
            tmp.resize(1);
            tmp[0] = 0.5;
            this->a.push_back(tmp);
            tmp.resize(2);
            tmp[0] = 0.0;
            tmp[1] = 0.5;
            this->a.push_back(tmp);
            tmp.resize(3);
            tmp[1] = 0.0;
            tmp[2] = 1.0;
            this->a.push_back(tmp);
            this->b.push_back(1.0 / 6.0);
            this->b.push_back(1.0 / 3.0);
            this->b.push_back(1.0 / 3.0);
            this->b.push_back(1.0 / 6.0);
            this->c.push_back(0.0);
            this->c.push_back(0.5);
            this->c.push_back(0.5);
            this->c.push_back(1.0);

            break;
          }
        case (THREE_STAGE_SECOND_ORDER):
          {
            const double chi = 2.0 - std::sqrt(2.0);
            const double a32 = 1.0 / 2.0;
            this->n_stages = 3;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            this->b.push_back(1.0 / 2.0 - chi / 4.0);
            this->b.push_back(1.0 / 2.0 - chi / 4.0);
            this->b.push_back(chi / 2.0);
            this->c.push_back(0.0);
            this->c.push_back(chi);
            this->c.push_back(1.0);
            std::vector<double> tmp;
            //this->a.push_back(tmp);
            tmp.resize(1);
            tmp[0] = chi;
            this->a.push_back(tmp);
            tmp.resize(2);
            tmp[0] = 1.0 - a32;
            tmp[1] = a32;
            this->a.push_back(tmp);

            break;
          }

        default:
          {
            AssertThrow(
              false, ExcMessage("Unimplemented explicit Runge-Kutta method."));
          }
      }
  }

  void
  ExplicitRungeKutta::get_coefficients(
    std::vector<std::vector<double>> &a,
    std::vector<double>              &b,
    std::vector<double>              &c) const
  {
    a = this->a;

    b.resize(this->b.size());
    b = this->b;

    c.resize(this->c.size());
    c = this->c;
  }


  
  // The ImplicitRungeKutta class could have been derived from RungeKutta
  // implementing a specific class of diagonally implicit methods (DIRK).
  // We did something even simpler
  // just cherry-picking what needed from the ExplicitRungeKutta class.
  // This is a constructor that initialize the Runge-Kutta tables with the
  // the coefficients and a function that retrieve such coefficients that are protected.
  // For the coefficient variable we have left the Deal.II names taken from the
  // Runge-Kutta literature `a`, `b` and `c`.  The lower triangular and the
  // diagonal part are stored separetaly with the diagonal coefficients that are
  // in are in `d`. This is because the latter does not go into the resiudal but
  // goes to modify the mass matrix.
  class ImplicitRungeKutta
  {
  public:
    /**
     * Constructor. This function calls initialize(runge_kutta_method).
     */
    ImplicitRungeKutta(const runge_kutta_method_oceano method);
    /**
     * Initialize the explicit Runge-Kutta method.
     */
    void
    initialize(const runge_kutta_method_oceano method);

    /**
     * Get the coefficients of the scheme
     */
    void
    get_coefficients(std::vector<std::vector<double>> &a,
                     std::vector<double>              &b,
                     std::vector<double>              &c,
                     std::vector<double>              &d) const;

    protected:
    /**
     * Number of stages of the Runge-Kutta method.
     */
    unsigned int n_stages;

    /**
     * SSP tableau coefficients.
     */
      std::vector<std::vector<double>> a;
    /**
     * SSP tableau coefficients.
     */
      std::vector<double> d;
    /**
     * SSP tableau coefficients.
     */
      std::vector<double> b;
    /**
     * Butcher tableau coefficients.
     */
      std::vector<double> c;
  };



  ImplicitRungeKutta::ImplicitRungeKutta(
    const runge_kutta_method_oceano method)
  {
    ImplicitRungeKutta::initialize(method);
  }

  void
  ImplicitRungeKutta::initialize(const runge_kutta_method_oceano method)
  {

    switch (method)
      {
        case (TRAPEZOIDAL_BDF2):
          {
            const double chi = 2.0 - std::sqrt(2.0);
            this->n_stages = 3;
            this->b.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            this->d.reserve(this->n_stages);
            this->b.push_back(1.0 / 2.0 - chi / 4.0);
            this->b.push_back(1.0 / 2.0 - chi / 4.0);
            this->b.push_back(chi / 2.0);
            this->c.push_back(0.0);
            this->c.push_back(chi);
            this->c.push_back(1.0);
            std::vector<double> tmp;
            //this->a.push_back(tmp);
            tmp.resize(1);
            tmp[0] = chi / 2.0;
            this->a.push_back(tmp);
            tmp.resize(2);
            tmp[0] = 1.0 / (2.0 * sqrt(2.0));
            tmp[1] = 1.0 / (2.0 * sqrt(2.0));
            this->a.push_back(tmp);
            this->d.push_back(chi / 2.0);
            this->d.push_back(1.0 - 1.0 / sqrt(2.0));

            break;
          }

        default:
          {
            AssertThrow(
              false, ExcMessage("Unimplemented implicit Runge-Kutta method."));
          }
      }
  }

  void
  ImplicitRungeKutta::get_coefficients(
    std::vector<std::vector<double>> &a,
    std::vector<double>              &b,
    std::vector<double>              &c,
    std::vector<double>              &d) const
  {
    a = this->a;

    b.resize(this->b.size());
    b = this->b;

    c.resize(this->c.size());
    c = this->c;

    d.resize(this->d.size());
    d = this->d;
  }  
} // namespace TimeSteppingOceano

#endif
