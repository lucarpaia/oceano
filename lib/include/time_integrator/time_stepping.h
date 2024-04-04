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
     * Invalid.
     */
    invalid
  };



  // The StrongStabilityRungeKutta class could have been derived from RungeKutta
  // implementing a specific class of explicit methods. We did something even simpler
  // just cherry-picking what needed from the ExplicitRungeKutta class.
  // This is a constructor that initialize the Runge-Kutta tables with the
  // the coefficients and a function that retrieve such coefficients that are protected.
  // For the coefficient variable we have left the Deal.II names taken from the
  // Runge-Kutta literature `a`, `b` and `c`. Note however that `a`  and `b`
  // correspond to the $\alpha$ and $\beta$ tables of the Shu form presented in
  // (Gottlieb and Shu). The table `a` is represented as a vector of vectors.
  // Since table `b` is diagonal for SSP22 and SSPP3 we
  // defined it as vector with the diagonal entries.
  // The main advantages of Strong-Stability-Preserving scheme is the enhanced
  // non-linear stability.
  class StrongStabilityRungeKutta
  {
  public:
    /**
     * Constructor. This function calls initialize(runge_kutta_method).
     */
    StrongStabilityRungeKutta(const runge_kutta_method_oceano method);
    /**
     * Initialize the explicit Runge-Kutta method.
     */
    void
    initialize(const runge_kutta_method_oceano method);

    /**
     * Get the coefficients of the scheme.
     * Note that here vector `a` and `b` are not the conventional definition in terms of a
     * Butcher tableau but corresponds respectively to the coefficients $\alpha_{ij}$ 
     * and $\beta_{ij}$ of the Shu form. More details can be
     * found in (Gottlieb and Shu).
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
      std::vector<std::vector<double>> alpha;
    /**
     * SSP tableau coefficients.
     */
      std::vector<double> beta;
    /**
     * Butcher tableau coefficients.
     */
      std::vector<double> c;
  };



  StrongStabilityRungeKutta::StrongStabilityRungeKutta(
    const runge_kutta_method_oceano method)
  {
    StrongStabilityRungeKutta::initialize(method);
  }

  void
  StrongStabilityRungeKutta::initialize(const runge_kutta_method_oceano method)
  {

    switch (method)
      {
        case (FORWARD_EULER):
          {
            this->n_stages = 1;
            std::vector<double> tmp; 
            tmp.resize(1);
            tmp[0] = 1.0;
            this->alpha.push_back(tmp);
            this->beta.push_back(1.0);
            this->c.push_back(0.0);

            break;
          }
        case (SSP_SECOND_ORDER):
          {
            this->n_stages = 2;
            this->beta.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            std::vector<double> tmp;
            tmp.resize(1);
            tmp[0] = 1.0;
            this->alpha.push_back(tmp);
            tmp.resize(2);
            tmp[0] = 1.0 / 2.0;
            tmp[1] = 1.0 / 2.0;
            this->alpha.push_back(tmp);
            this->beta.push_back(1.0);
            this->beta.push_back(1.0 / 2.0);
            this->c.push_back(0.0);
            this->c.push_back(1.0);

            break;
          }
        case (SSP_THIRD_ORDER):
          {
            this->n_stages = 3;
            this->beta.reserve(this->n_stages);
            this->c.reserve(this->n_stages);
            std::vector<double> tmp;
            tmp.resize(1);
            tmp[0] = 1.0;
            this->alpha.push_back(tmp);
            tmp.resize(2);
            tmp[0] = 3.0 / 4.0;
            tmp[1] = 1.0 / 4.0;
            this->alpha.push_back(tmp);
            tmp.resize(3);
            tmp[0] = 1.0 / 3.0;
            tmp[1] = 0.0;
            tmp[2] = 2.0 / 3.0;
            this->alpha.push_back(tmp);     
            this->beta.push_back(1.0);
            this->beta.push_back(1.0 / 4.0);
            this->beta.push_back(2.0 / 3.0);
            this->c.push_back(0.0);
            this->c.push_back(1.0);
            this->c.push_back(0.5);

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
  StrongStabilityRungeKutta::get_coefficients(
    std::vector<std::vector<double>> &a,
    std::vector<double>              &b,
    std::vector<double>              &c) const
  {
    a = this->alpha;

    b.resize(this->beta.size());
    b = this->beta;

    c.resize(this->c.size());
    c = this->c;
  }
} // namespace TimeSteppingOceano

#endif
