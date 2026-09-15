/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2026 by CNR-ISMAR
 *
 * This code, as the deal.II library is free software; you can use it,
 * redistribute it, and/or modify it under the terms of the GNU Lesser
 * General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later
 * version.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Luca Arpaia, 2026
 */

#ifndef GROWTHDEATHRATEROSENZWEIGMACARTHUR_H
#define GROWTHDEATHRATEROSENZWEIGMACARTHUR_H

/**
 * Namespace containing biology physics parametrizations.
 */
namespace Physics
{

  using namespace dealii;

  // @sect3{Implementation of the Rosenzweig-MacArthur growth-death rate}

  // The Rosenzweig-MacArthur model describes the interaction between
  // a prey population n1 and a predator population n2:
  //
  // $$f(n1,n2) = r n1 (1 - n1/K)
  //                - a n1 n2/(1 + a tau n1),$$
  //
  // $$g(n1,n2) = e a n1 n2/(1 + a tau n1) - d n2.$$
  //
  // The prey follows logistic growth with intrinsic growth rate r and
  // carrying capacity K. Predation follows a Holling type II functional
  // response with attack rate a and handling time tau. The predator growth
  // is proportional to predation through the conversion rate e and predators
  // die at the constant rate d.
  //
  // All the parameters specific to the model are declared as private values
  // of type double. We choose double values because the differences
  // between growth and predation terms can suffer noticeably more
  // rounding error with float, particularly near an equilibrium where the
  // net source should approach zero.
  class GrowthDeathRateRosenzweigMacArthur
  {
  public:
    GrowthDeathRateRosenzweigMacArthur(IO::ParameterHandler &prm);
    ~GrowthDeathRateRosenzweigMacArthur() = default;

    template <int n_tra, typename Number>
    inline DEAL_II_ALWAYS_INLINE //
      Tensor<1, n_tra, Number>
      value(const Tensor<1, n_tra, Number> &concentration) const;

  private:
    double r;    /*!<prey intrinsic growth rate */
    double K;    /*!<prey carrying capacity */
    double e;    /*!<predator conversion efficiency */
    double a;    /*!<predator attack rate */
    double tau;  /*!<handling time */
    double d;    /*!<predator mortality rate */
  };



  // The class constructor reads the Rosenzweig-MacArthur physical
  // constants from the parameter file.
  GrowthDeathRateRosenzweigMacArthur::GrowthDeathRateRosenzweigMacArthur(
    IO::ParameterHandler &prm)
  {
    prm.enter_subsection("Biological constants");
    r = prm.get_double("prey_growth_rate");
    K = prm.get_double("prey_capacity");
    e = prm.get_double("predator_efficiency");
    a = prm.get_double("predator_attack_rate");
    tau = prm.get_double("predator_handling_time");
    d = prm.get_double("predator_death_rate");
    prm.leave_subsection();
  }



  template <int n_tra, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_tra, Number>
    GrowthDeathRateRosenzweigMacArthur::value(
      const Tensor<1, n_tra, Number> &concentration) const
  {
    const Number n1 = concentration[0];
    const Number n2 = concentration[1];

    const Number predation =
      Number(a) * n1 * n2 /
      (Number(1.0) + Number(a * tau) * n1);

    Tensor<1, n_tra, Number> source;
    source[0] =
      Number(r) * n1 * (Number(1.0) - n1 / Number(K)) - predation;
    source[1] =
      Number(e) * predation - Number(d) * n2;

    return source;
  }
} // namespace Physics
#endif // GROWTHDEATHRATEROSENZWEIGMACARTHUR_H
