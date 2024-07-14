/* Author: Giuseppe Orlando, 2024. */

// @sect{Include files}

// We start by including all the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/meshworker/mesh_loop.h>

#include "runtime_parameters.h"
#include "equation_data.h"

// This is the class that implements the discretization
//
namespace SW {
  using namespace dealii;

  // @sect{ <code>SWOperator::SWOperator</code> }
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  class SWOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    SWOperator(); /*--- Default constructor ---*/

    SWOperator(RunTimeParameters::Data_Storage& data); /*--- Constructor with some input related data ---*/

    void set_dt(const double time_step); /*--- Setter of the time-step. This is useful both for multigrid purposes and also
                                               in case of modifications of the time step. ---*/

    void set_IMEX_stage(const unsigned int stage); /*--- Setter of the IMEX stage. ---*/

    void set_SW_stage(const unsigned int stage); /*--- Setter of the equation currently under solution. ---*/

    void set_h_curr(const Vec& src); /*--- Setter of the current depth. This is for the assembling of the bilinear forms
                                           where only one source vector can be passed in input. ---*/

    void set_hu_curr(const Vec& src); /*--- Setter of the current discharge. This is for the assembling of the bilinear forms
                                            where only one source vector can be passed in input. ---*/

    void vmult_rhs_h(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs
                                                                         for the depth. ---*/

    void vmult_rhs_hu(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                           for the discharge. ---*/

    void vmult_rhs_hc(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                           for the tracer. ---*/

    virtual void compute_diagonal() override; /*--- Compute the diagonal for several preconditioners ---*/

  protected:
    double dt; /*--- Time step. ---*/

    const double gamma; /*--- TR-BDF2 (i.e. implicit part) parameter. ---*/
    /*--- The following variables follow the classical Butcher tableaux notation ---*/
    const double a21;
    const double a31;
    const double a32;
    std::vector<std::vector<double>> a;

    const double a21_tilde;
    const double a22_tilde;
    const double a31_tilde;
    const double a32_tilde;
    const double a33_tilde;
    std::vector<std::vector<double>> a_tilde;

    const double b1;
    const double b2;
    const double b3;
    std::vector<double> b;
    std::vector<double> b_tilde;

    unsigned int IMEX_stage; /*--- Flag for the IMEX stage ---*/
    unsigned int SW_stage;   /*--- Flag for the equation actually solved ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
    Vec h_curr,
        hu_curr;

    /*--- Assembler functions for the rhs related to the depth equation. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_h(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_h(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_h(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler function related to the bilinear form of the depth equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_h(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the rhs related to the discharge equation. ---*/
    void assemble_rhs_cell_term_hu(const MatrixFree<dim, Number>&               data,
                                   Vec&                                         dst,
                                   const std::vector<Vec>&                      src,
                                   const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_hu(const MatrixFree<dim, Number>&               data,
                                   Vec&                                         dst,
                                   const std::vector<Vec>&                      src,
                                   const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_hu(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler function for the matrix associated to the discharge equation. ---*/
    void assemble_cell_term_hu(const MatrixFree<dim, Number>&               data,
                               Vec&                                         dst,
                               const Vec&                                   src,
                               const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the rhs of the tracer equation. ---*/
    void assemble_rhs_cell_term_hc(const MatrixFree<dim, Number>&               data,
                                   Vec&                                         dst,
                                   const std::vector<Vec>&                      src,
                                   const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_hc(const MatrixFree<dim, Number>&               data,
                                   Vec&                                         dst,
                                   const std::vector<Vec>&                      src,
                                   const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_hc(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler function for the matrix associated to the tracer equation. ---*/
    void assemble_cell_term_hc(const MatrixFree<dim, Number>&               data,
                               Vec&                                         dst,
                               const Vec&                                   src,
                               const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the diagonal part of the matrix for the depth equation. ---*/
    void assemble_diagonal_cell_term_h(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the diagonal part of the matrix for the discharge equation. ---*/
    void assemble_diagonal_cell_term_hu(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const unsigned int&                          src,
                                        const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the diagonal part of the matrix for the tracer equation. ---*/
    void assemble_diagonal_cell_term_hc(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const unsigned int&                          src,
                                        const std::pair<unsigned int, unsigned int>& cell_range) const;
  };


  // Default constructor
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  SWOperator<dim, n_stages,
             fe_degree_h, fe_degree_hu, fe_degree_hc,
             n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  SWOperator(): MatrixFreeOperators::Base<dim, Vec>(), dt(),
                gamma(2.0 - std::sqrt(2.0)), a21(gamma),
                a31(0.5), a32(0.5), a(n_stages, std::vector<double>(n_stages)),
                a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0), a33_tilde(1.0 - std::sqrt(2)/2.0),
                a_tilde(n_stages, std::vector<double>(n_stages)),
                b1(0.5 - 0.25*gamma), b2(0.5 - 0.25*gamma), b3(0.5*gamma), b(n_stages), b_tilde(b),
                IMEX_stage(1), SW_stage(1) {
    /*--- Butcher tableux of the explicit part ---*/
    std::fill(a.begin(), a.end(), std::vector<double>(n_stages, 0.0));
    a[1][0] = a21;
    a[2][0] = a31;
    a[2][1] = a32;

    /*--- Butcher tableux of the implicit part ---*/
    std::fill(a_tilde.begin(), a_tilde.end(), std::vector<double>(n_stages, 0.0));
    a_tilde[1][0] = a21_tilde;
    a_tilde[1][1] = a22_tilde;
    a_tilde[2][0] = a31_tilde;
    a_tilde[2][1] = a32_tilde;
    a_tilde[2][2] = a33_tilde;

    /*--- Auxiliary vectors for the weigths ---*/
    b[0] = b1;
    b[1] = b2;
    b[2] = b3;
    b_tilde = b;
  }

  // Constructor with runtime parameters storage
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  SWOperator<dim, n_stages,
             fe_degree_h, fe_degree_hu, fe_degree_hc,
             n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  SWOperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(), dt(data.dt),
                                                     gamma(2.0 - std::sqrt(2.0)), a21(gamma),
                                                     a31(0.5), a32(0.5), a(n_stages, std::vector<double>(n_stages)),
                                                     a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                                                     a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0),
                                                     a33_tilde(1.0 - std::sqrt(2)/2.0), a_tilde(n_stages, std::vector<double>(n_stages)),
                                                     b1(0.5 - 0.25*gamma), b2(0.5 - 0.25*gamma), b3(0.5*gamma), b(n_stages), b_tilde(b),
                                                     IMEX_stage(1), SW_stage(1) {
    /*--- Butcher tableux of the explicit part ---*/
    std::fill(a.begin(), a.end(), std::vector<double>(n_stages, 0.0));
    a[1][0] = a21;
    a[2][0] = a31;
    a[2][1] = a32;

    /*--- Butcher tableux of the implicit part ---*/
    std::fill(a_tilde.begin(), a_tilde.end(), std::vector<double>(n_stages, 0.0));
    a_tilde[1][0] = a21_tilde;
    a_tilde[1][1] = a22_tilde;
    a_tilde[2][0] = a31_tilde;
    a_tilde[2][1] = a32_tilde;
    a_tilde[2][2] = a33_tilde;

    /*--- Auxiliary vectors for the weigths ---*/
    b[0] = b1;
    b[1] = b2;
    b[2] = b3;
    b_tilde = b;
  }


  // Setter of time-step
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  set_dt(const double time_step) {
    dt = time_step;
  }

  // Setter of IMEX stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  set_IMEX_stage(const unsigned int stage) {
    AssertIndexRange(stage, n_stages + 2);
    Assert(stage > 0, ExcInternalError());

    IMEX_stage = stage;
  }

  // Setter of SW stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  set_SW_stage(const unsigned int stage) {
    AssertIndexRange(stage, 7);
    Assert(stage > 0, ExcInternalError());

    SW_stage = stage;
  }

  // Setter of current depth
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  set_h_curr(const Vec& src) {
    h_curr = src;
    h_curr.update_ghost_values();
  }

  // Setter of current discharge
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  set_hu_curr(const Vec& src) {
    hu_curr = src;
    hu_curr.update_ghost_values();
  }


  // Assemble rhs cell term for the depth equation
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_rhs_cell_term_h(const MatrixFree<dim, Number>&               data,
                           Vec&                                         dst,
                           const std::vector<Vec>&                      src,
                           const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEEvaluation_h  = FEEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>;
    using FEEvaluation_hu = FEEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the density and
      the velocity and previous stages. 'phi' will be used only to 'submit' the result.
      The second argument specifies which dof handler has to be used (in this implementation 0 stands for
      depth, 1 for discharge (aka mass flux) and 2 for tracer). ---*/
      FEEvaluation_h               phi(data, 0),
                                   phi_h_old(data, 0);
      std::vector<FEEvaluation_hu> phi_hu(IMEX_stage - 1, FEEvaluation_hu(data, 1));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        /*--- Now we need to assign the current cell to each FEEvaluation object and then to specify which src vector
        it has to read (the proper order is clearly delegated to the user, which has to pay attention in the function
        call to be coherent). All these considerations are valid also for the other assembler functions ---*/
        phi_h_old.reinit(cell);
        phi_h_old.gather_evaluate(src[0], EvaluationFlags::values);
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_hu[s - 1].reinit(cell);
          phi_hu[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the depth at the previous step (always needed). ---*/
          const auto& h_old = phi_h_old.get_value(q);

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& hu_s = phi_hu[s - 1].get_value(q);

            flux += a[IMEX_stage - 1][s - 1]*dt*hu_s;
          }

          phi.submit_value(h_old, q);
          /*--- submit_value is used for quantities to be tested against test functions ---*/
          phi.submit_gradient(flux, q);
          /*--- submit_gradient is used for quantities to be tested against gradient of test functions ---*/
        }

        /*--- 'integrate_scatter' is the responsible of distributing into dst.
              The flag parameter specifies if we are testing against the test function and/or its gradient ---*/
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_h               phi(data, 0),
                                   phi_h_old(data, 0);
      std::vector<FEEvaluation_hu> phi_hu(IMEX_stage - 1, FEEvaluation_hu(data, 1));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_h_old.reinit(cell);
        phi_h_old.gather_evaluate(src[0], EvaluationFlags::values);
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_hu[s - 1].reinit(cell);
          phi_hu[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the density at the previous step (always needed) ---*/
          const auto& h_old = phi_h_old.get_value(q);

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& hu_s = phi_hu[s - 1].get_value(q);

            flux += b[s - 1]*dt*hu_s;
          }

          phi.submit_value(h_old, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term for the depth
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_rhs_face_term_h(const MatrixFree<dim, Number>&               data,
                           Vec&                                         dst,
                           const std::vector<Vec>&                      src,
                           const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEFaceEvaluation_h  = FEFaceEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>;
    using FEFaceEvaluation_hu = FEFaceEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities.
            'true' means that we are reading the information from 'inside', whereas 'false' from 'outside' ---*/
      FEFaceEvaluation_h  phi_m(data, true, 0),
                          phi_p(data, false, 0),
                          phi_h_m(data, true, 0),
                          phi_h_p(data, false, 0);
      FEFaceEvaluation_hu phi_hu_m(data, true, 1),
                          phi_hu_p(data, false, 1);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_hu_m.reinit(face);
        phi_hu_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        phi_h_m.reinit(face);
        phi_h_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_plus = phi_m.get_normal_vector(q); /*--- Notice that the unit normal vector is the same from
                                                                 'both sides'. ---*/

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_hu_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
            phi_hu_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);

            const auto& hu_s_m     = phi_hu_m.get_value(q);
            const auto& hu_s_p     = phi_hu_p.get_value(q);

            const auto& avg_flux_s = 0.5*(hu_s_m + hu_s_p);
            phi_h_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_h_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            const auto& h_s_m      = phi_h_m.get_value(q);
            const auto& h_s_p      = phi_h_p.get_value(q);
            const auto& lambda_s   = std::max(std::abs(scalar_product(hu_s_m/h_s_m, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_m),
                                              std::abs(scalar_product(hu_s_p/h_s_p, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_p));
            const auto& jump_h_s   = h_s_m - h_s_p;

            flux += a[IMEX_stage - 1][s - 1]*dt*(scalar_product(avg_flux_s, n_plus) + 0.5*lambda_s*jump_h_s);
          }

          phi_m.submit_value(-flux, q);
          phi_p.submit_value(flux, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_h  phi_m(data, true, 0),
                          phi_p(data, false, 0),
                          phi_h_m(data, true, 0),
                          phi_h_p(data, false, 0);
      FEFaceEvaluation_hu phi_hu_m(data, true, 1),
                          phi_hu_p(data, false, 1);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_hu_m.reinit(face);
        phi_hu_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        phi_h_m.reinit(face);
        phi_h_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_plus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_hu_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
            phi_hu_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);

            const auto& hu_s_m     = phi_hu_m.get_value(q);
            const auto& hu_s_p     = phi_hu_p.get_value(q);

            const auto& avg_flux_s = 0.5*(hu_s_m + hu_s_p);
            phi_h_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_h_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            const auto& h_s_m      = phi_h_m.get_value(q);
            const auto& h_s_p      = phi_h_p.get_value(q);
            const auto& lambda_s   = std::max(std::abs(scalar_product(hu_s_m/h_s_m, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_m),
                                              std::abs(scalar_product(hu_s_p/h_s_p, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_p));
            const auto& jump_h_s  = h_s_m - h_s_p;

            flux += b[s - 1]*dt*(scalar_product(avg_flux_s, n_plus) + 0.5*lambda_s*jump_h_s);
          }

          phi_m.submit_value(-flux, q);
          phi_p.submit_value(flux, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for depth
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  vmult_rhs_h(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&SWOperator::assemble_rhs_cell_term_h,
                     &SWOperator::assemble_rhs_face_term_h,
                     &SWOperator::assemble_rhs_boundary_term_h,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }

  // Assemble cell term for the depth
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_cell_term_h(const MatrixFree<dim, Number>&               data,
                       Vec&                                         dst,
                       const Vec&                                   src,
                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number> phi(data, 0);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble rhs cell term for the discharge
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_rhs_cell_term_hu(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const std::vector<Vec>&                      src,
                            const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEEvaluation_hu   = FEEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number>;
    using FEEvaluation_h    = FEEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>;
    using FEEvaluation_zeta = FEEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_hu                phi(data, 1);
      std::vector<FEEvaluation_hu>   phi_hu(IMEX_stage - 1, FEEvaluation_hu(data, 1));
      std::vector<FEEvaluation_h>    phi_h(IMEX_stage - 1, FEEvaluation_h(data, 0));
      std::vector<FEEvaluation_zeta> phi_zeta(IMEX_stage - 1, FEEvaluation_zeta(data, 0));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_h[s - 1].reinit(cell);
          phi_h[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_hu[s - 1].reinit(cell);
          phi_hu[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_zeta[s - 1].reinit(cell);
          phi_zeta[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::gradients);
        }

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the discharge at the previous step (always necessary).
                Notice that this is ok because of explicit method. ---*/
          const auto& hu_old = phi_hu[0].get_value(q);

          /*--- Compute the quantites at the previous stages ---*/
          Tensor<2, dim, VectorizedArray<Number>> flux;
          Tensor<1, dim, VectorizedArray<Number>> non_cons_flux;
          Tensor<1, dim, VectorizedArray<Number>> friction;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& h_s         = phi_h[s - 1].get_value(q);
            const auto& hu_s        = phi_hu[s - 1].get_value(q);

            const auto& grad_zeta_s = phi_zeta[s - 1].get_gradient(q);

            flux += a[IMEX_stage - 1][s - 1]*dt*outer_product(hu_s, hu_s/h_s);
            non_cons_flux += a[IMEX_stage - 1][s - 1]*dt*EquationData::g*h_s*grad_zeta_s;

            const auto& gamma_s = 0.0; // TODO: Add proper expression of the friction
            friction += a_tilde[IMEX_stage - 1][s - 1]*dt*(gamma_s*(hu_s/h_s));
          }

          phi.submit_value(hu_old - non_cons_flux - friction, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_hu                phi(data, 1);
      std::vector<FEEvaluation_hu>   phi_hu(IMEX_stage - 1, FEEvaluation_hu(data, 1));
      std::vector<FEEvaluation_h>    phi_h(IMEX_stage - 1, FEEvaluation_h(data, 0));
      std::vector<FEEvaluation_zeta> phi_zeta(IMEX_stage - 1, FEEvaluation_zeta(data, 0));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_h[s - 1].reinit(cell);
          phi_h[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_hu[s - 1].reinit(cell);
          phi_hu[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_zeta[s - 1].reinit(cell);
          phi_zeta[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::gradients);
        }

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the discharge at the previous step (always necessary).
                Notice that this is ok because of explicit method.---*/
          const auto& hu_old = phi_hu[0].get_value(q);

          /*--- Compute the quantites at the previous stages ---*/
          Tensor<2, dim, VectorizedArray<Number>> flux;
          Tensor<1, dim, VectorizedArray<Number>> non_cons_flux;
          Tensor<1, dim, VectorizedArray<Number>> friction;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& h_s         = phi_h[s - 1].get_value(q);
            const auto& hu_s        = phi_hu[s - 1].get_value(q);

            const auto& grad_zeta_s = phi_zeta[s - 1].get_gradient(q);

            flux += b[s - 1]*dt*outer_product(hu_s, hu_s/h_s);
            non_cons_flux += b[s - 1]*dt*EquationData::g*h_s*grad_zeta_s;

            const auto& gamma_s = 0.0; //TODO: Add proper expression of the friction
            friction += b_tilde[s-1]*dt*(gamma_s*(hu_s/h_s));
          }

          phi.submit_value(hu_old - non_cons_flux - friction, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term for the discharge
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_rhs_face_term_hu(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const std::vector<Vec>&                      src,
                            const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEFaceEvaluation_hu   = FEFaceEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number>;
    using FEFaceEvaluation_h    = FEFaceEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>;
    using FEFaceEvaluation_zeta = FEFaceEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_hu   phi_m(data, true, 1),
                            phi_p(data, false, 1),
                            phi_hu_m(data, true, 1),
                            phi_hu_p(data, false, 1);
      FEFaceEvaluation_h    phi_h_m(data, true, 0),
                            phi_h_p(data, false, 0);
      FEFaceEvaluation_zeta phi_zeta_m(data, true, 0),
                            phi_zeta_p(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_h_m.reinit(face);
        phi_h_p.reinit(face);
        phi_hu_m.reinit(face);
        phi_hu_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        phi_zeta_m.reinit(face);
        phi_zeta_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_plus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          VectorizedArray<Number> non_cons_flux_p = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> non_cons_flux_m = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_h_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_h_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_hu_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_hu_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);

            const auto& h_s_m                   = phi_h_m.get_value(q);
            const auto& h_s_p                   = phi_h_p.get_value(q);
            const auto& hu_s_m                  = phi_hu_m.get_value(q);
            const auto& hu_s_p                  = phi_hu_p.get_value(q);

            const auto& avg_tensor_product_flux = 0.5*(outer_product(hu_s_m, hu_s_m/h_s_m) +
                                                       outer_product(hu_s_p, hu_s_p/h_s_p));
            const auto& lambda_s                = std::max(std::abs(scalar_product(hu_s_m/h_s_m, n_plus)) +
                                                           std::sqrt(EquationData::g*h_s_m),
                                                           std::abs(scalar_product(hu_s_p/h_s_p, n_plus)) +
                                                           std::sqrt(EquationData::g*h_s_p));
            const auto& jump_hu_s               = hu_s_m - hu_s_p;

            /*--- Add contribution due to non-conservative term ---*/
            phi_zeta_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_zeta_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            const auto& zeta_s_m                = phi_zeta_m.get_value(q);
            const auto& zeta_s_p                = phi_zeta_p.get_value(q);
            /*const auto& avg_flux_part_1_zb      = 0.5*EquationData::g*(h_s_m*zeta_s_m + h_s_p*zeta_s_p);
            const auto& avg_flux_part_2_zb      = 0.5*EquationData::g*(h_s_m + h_s_p);*/ //NOTE: No exact cancellation
            const auto& avg_flux_part_1_zb      = 0.5*EquationData::g*h_s_m*zeta_s_m
                                                + 0.5*EquationData::g*h_s_p*zeta_s_p;
            const auto& avg_flux_part_2_zb      = 0.5*EquationData::g*h_s_m
                                                + 0.5*EquationData::g*h_s_p;

            flux += a[IMEX_stage - 1][s - 1]*dt*(avg_tensor_product_flux*n_plus +
                                                 0.5*lambda_s*jump_hu_s);

            non_cons_flux_m += a[IMEX_stage - 1][s - 1]*dt*(avg_flux_part_1_zb - avg_flux_part_2_zb*zeta_s_m);
            non_cons_flux_p += a[IMEX_stage - 1][s - 1]*dt*(avg_flux_part_1_zb - avg_flux_part_2_zb*zeta_s_p);
          }

          phi_m.submit_value(-flux - non_cons_flux_m*n_plus, q);
          phi_p.submit_value(flux + non_cons_flux_p*n_plus, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_hu   phi_m(data, true, 1),
                            phi_p(data, false, 1),
                            phi_hu_m(data, true, 1),
                            phi_hu_p(data, false, 1);
      FEFaceEvaluation_h    phi_h_m(data, true, 0),
                            phi_h_p(data, false, 0);                      
      FEFaceEvaluation_zeta phi_zeta_m(data, true, 0),
                            phi_zeta_p(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_h_m.reinit(face);
        phi_h_p.reinit(face);
        phi_hu_m.reinit(face);
        phi_hu_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        phi_zeta_m.reinit(face);
        phi_zeta_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_plus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          VectorizedArray<Number> non_cons_flux_p = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> non_cons_flux_m = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_h_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_h_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_hu_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_hu_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);

            const auto& h_s_m                   = phi_h_m.get_value(q);
            const auto& h_s_p                   = phi_h_p.get_value(q);
            const auto& hu_s_m                  = phi_hu_m.get_value(q);
            const auto& hu_s_p                  = phi_hu_p.get_value(q);

            const auto& avg_tensor_product_flux = 0.5*(outer_product(hu_s_m, hu_s_m/h_s_m) +
                                                       outer_product(hu_s_p, hu_s_p/h_s_p));
            const auto& lambda_s                = std::max(std::abs(scalar_product(hu_s_m/h_s_m, n_plus)) +
                                                           std::sqrt(EquationData::g*h_s_m),
                                                           std::abs(scalar_product(hu_s_p/h_s_p, n_plus)) +
                                                           std::sqrt(EquationData::g*h_s_p));
            const auto& jump_hu_s               = hu_s_m - hu_s_p;

            /*--- Add contribution due to non-conservative term ---*/
            phi_zeta_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_zeta_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            const auto& zeta_s_m                = phi_zeta_m.get_value(q);
            const auto& zeta_s_p                = phi_zeta_p.get_value(q);
            /*const auto& avg_flux_part_1_zb      = 0.5*EquationData::g*(h_s_m*zeta_s_m + h_s_p*zeta_s_p);
            const auto& avg_flux_part_2_zb      = 0.5*EquationData::g*(h_s_m + h_s_p);*/ //NOTE: No exact cancellation
            const auto& avg_flux_part_1_zb      = 0.5*EquationData::g*h_s_m*zeta_s_m
                                                + 0.5*EquationData::g*h_s_p*zeta_s_p;
            const auto& avg_flux_part_2_zb      = 0.5*EquationData::g*h_s_m
                                                + 0.5*EquationData::g*h_s_p;

            flux += b[s - 1]*dt*(avg_tensor_product_flux*n_plus +
                                 0.5*lambda_s*jump_hu_s);

            non_cons_flux_m += b[s - 1]*dt*(avg_flux_part_1_zb - avg_flux_part_2_zb*zeta_s_m);
            non_cons_flux_p += b[s - 1]*dt*(avg_flux_part_1_zb - avg_flux_part_2_zb*zeta_s_p);
          }

          phi_m.submit_value(-flux - non_cons_flux_m*n_plus, q);
          phi_p.submit_value(flux + non_cons_flux_p*n_plus, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the discharge
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  vmult_rhs_hu(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&SWOperator::assemble_rhs_cell_term_hu,
                     &SWOperator::assemble_rhs_face_term_hu,
                     &SWOperator::assemble_rhs_boundary_term_hu,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }

  // Assemble cell term for the discharge
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_cell_term_hu(const MatrixFree<dim, Number>&               data,
                        Vec&                                         dst,
                        const Vec&                                   src,
                        const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number> phi(data, 1);

    if(IMEX_stage <= n_stages) {
      /*--- Since here we have just one 'src' vector, but we also need to deal with the current depth and discharge,
            we employ the auxiliary vectors where we set this information ---*/
      FEEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>    phi_h_curr(data, 0);
      FEEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number> phi_hu_curr(data, 1);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_h_curr.reinit(cell);
        phi_h_curr.gather_evaluate(h_curr, EvaluationFlags::values);

        phi_hu_curr.reinit(cell);
        phi_hu_curr.gather_evaluate(hu_curr, EvaluationFlags::values);

        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& h_s      = phi_h_curr.get_value(q);

          const auto& hu_s     = phi_hu_curr.get_value(q);
          const auto& mod_hu_s = std::sqrt(scalar_product(hu_s, hu_s));

          const auto& gamma    = 0.0; // TODO: Add friction as a function of h_s and hu_s

          phi.submit_value((1.0 + a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*(gamma/h_s))*phi.get_value(q), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi.get_value(q), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // Assemble rhs cell term for the tracer equation
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_rhs_cell_term_hc(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const std::vector<Vec>&                      src,
                            const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEEvaluation_hc = FEEvaluation<dim, fe_degree_hc, n_q_points_1d_hu, 1, Number>;
    using FEEvaluation_hu = FEEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number>;
    using FEEvaluation_h  = FEEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_hc              phi(data, 2);
      std::vector<FEEvaluation_hc> phi_hc(IMEX_stage - 1, FEEvaluation_hc(data, 2));
      std::vector<FEEvaluation_hu> phi_hu(IMEX_stage - 1, FEEvaluation_hu(data, 1));
      std::vector<FEEvaluation_h>  phi_h(IMEX_stage - 1, FEEvaluation_h(data, 0));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_h[s - 1].reinit(cell);
          phi_h[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_hu[s - 1].reinit(cell);
          phi_hu[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_hc[s - 1].reinit(cell);
          phi_hc[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the tracer at the previous step (always needed).
                Notice that this is ok because of explicit method. ---*/
          const auto& hc_old = phi_hc[0].get_value(q);

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& hc_s = phi_hc[s - 1].get_value(q);

            const auto& hu_s = phi_hu[s - 1].get_value(q);
            const auto& h_s  = phi_h[s - 1].get_value(q);

            flux += a[IMEX_stage - 1][s - 1]*dt*hu_s*(hc_s/h_s);
          }

          phi.submit_value(hc_old, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_hc              phi(data, 2);
      std::vector<FEEvaluation_hc> phi_hc(IMEX_stage - 1, FEEvaluation_hc(data, 2));
      std::vector<FEEvaluation_hu> phi_hu(IMEX_stage - 1, FEEvaluation_hu(data, 1));
      std::vector<FEEvaluation_h>  phi_h(IMEX_stage - 1, FEEvaluation_h(data, 0));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_h[s - 1].reinit(cell);
          phi_h[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_hu[s - 1].reinit(cell);
          phi_hu[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_hc[s - 1].reinit(cell);
          phi_hc[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the tracer at the previous step (always needed).
                Notice that this is ok because of explicit method. ---*/
          const auto& hc_old = phi_hc[0].get_value(q);

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& hc_s = phi_hc[s - 1].get_value(q);

            const auto& hu_s = phi_hu[s - 1].get_value(q);
            const auto& h_s  = phi_h[s - 1].get_value(q);

            flux += b[s - 1]*dt*hu_s*(hc_s/h_s);
          }

          phi.submit_value(hc_old, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term for the tracer
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_rhs_face_term_hc(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const std::vector<Vec>&                      src,
                            const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEFaceEvaluation_hc = FEFaceEvaluation<dim, fe_degree_hc, n_q_points_1d_hu, 1, Number>;
    using FEFaceEvaluation_hu = FEFaceEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number>;
    using FEFaceEvaluation_h  = FEFaceEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_hc phi_m(data, true, 2),
                          phi_p(data, false, 2),
                          phi_hc_m(data, true, 2),
                          phi_hc_p(data, false, 2);
      FEFaceEvaluation_hu phi_hu_m(data, true, 1),
                          phi_hu_p(data, false, 1);
      FEFaceEvaluation_h  phi_h_m(data, true, 0),
                          phi_h_p(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_hc_m.reinit(face);
        phi_hc_p.reinit(face);
        phi_hu_m.reinit(face);
        phi_hu_p.reinit(face);
        phi_h_m.reinit(face);
        phi_h_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_plus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_h_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_h_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_hu_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_hu_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_hc_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_hc_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& hc_s_m     = phi_hc_m.get_value(q);
            const auto& hc_s_p     = phi_hc_p.get_value(q);
            const auto& hu_s_m     = phi_hu_m.get_value(q);
            const auto& hu_s_p     = phi_hu_p.get_value(q);
            const auto& h_s_m      = phi_h_m.get_value(q);
            const auto& h_s_p      = phi_h_p.get_value(q);

            const auto& avg_flux_s = 0.5*(hu_s_m*(hc_s_m/h_s_m) + hu_s_p*(hc_s_p/h_s_p));
            const auto& lambda_s   = std::max(std::abs(scalar_product(hu_s_m/h_s_m, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_m),
                                              std::abs(scalar_product(hu_s_p/h_s_p, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_p));
            const auto& jump_hc_s  = hc_s_m - hc_s_p;

            flux += a[IMEX_stage - 1][s - 1]*dt*(scalar_product(avg_flux_s, n_plus) + 0.5*lambda_s*jump_hc_s);
          }

          phi_m.submit_value(-flux, q);
          phi_p.submit_value(flux, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_hc phi_m(data, true, 2),
                          phi_p(data, false, 2),
                          phi_hc_m(data, true, 2),
                          phi_hc_p(data, false, 2);
      FEFaceEvaluation_hu phi_hu_m(data, true, 1),
                          phi_hu_p(data, false, 1);
      FEFaceEvaluation_h  phi_h_m(data, true, 0),
                          phi_h_p(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_hc_m.reinit(face);
        phi_hc_p.reinit(face);
        phi_hu_m.reinit(face);
        phi_hu_p.reinit(face);
        phi_h_m.reinit(face);
        phi_h_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_plus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_h_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_h_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_hu_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_hu_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_hc_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_hc_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& hc_s_m     = phi_hc_m.get_value(q);
            const auto& hc_s_p     = phi_hc_p.get_value(q);
            const auto& hu_s_m     = phi_hu_m.get_value(q);
            const auto& hu_s_p     = phi_hu_p.get_value(q);
            const auto& h_s_m      = phi_h_m.get_value(q);
            const auto& h_s_p      = phi_h_p.get_value(q);

            const auto& avg_flux_s = 0.5*(hu_s_m*(hc_s_m/h_s_m) + hu_s_p*(hc_s_p/h_s_p));
            const auto& lambda_s   = std::max(std::abs(scalar_product(hu_s_m/h_s_m, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_m),
                                              std::abs(scalar_product(hu_s_p/h_s_p, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_p));
            const auto& jump_hc_s  = hc_s_m - hc_s_p;

            flux += b[s - 1]*dt*(scalar_product(avg_flux_s, n_plus) + 0.5*lambda_s*jump_hc_s);
          }

          phi_m.submit_value(-flux, q);
          phi_p.submit_value(flux, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the tracer
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  vmult_rhs_hc(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&SWOperator::assemble_rhs_cell_term_hc,
                     &SWOperator::assemble_rhs_face_term_hc,
                     &SWOperator::assemble_rhs_boundary_term_hc,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }

  // Assemble cell term for the tracer
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_cell_term_hc(const MatrixFree<dim, Number>&               data,
                        Vec&                                         dst,
                        const Vec&                                   src,
                        const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_hc, n_q_points_1d_hu, 1, Number> phi(data, 2);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(SW_stage, 7);
    Assert(SW_stage > 0, ExcInternalError());

    if(SW_stage == 1 || SW_stage == 4) {
      this->data->cell_loop(&SWOperator::assemble_cell_term_h,
                            this, dst, src, false);
    }
    else if(SW_stage == 2 || SW_stage == 5) {
      this->data->cell_loop(&SWOperator::assemble_cell_term_hu,
                            this, dst, src, false);
    }
    else if(SW_stage == 3 || SW_stage == 6) {
      this->data->cell_loop(&SWOperator::assemble_cell_term_hc,
                            this, dst, src, false);
    }
    else {
      Assert(false, ExcInternalError());
    }
  }


  // Assemble diagonal cell term for the depth
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_diagonal_cell_term_h(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const unsigned int&                          ,
                                const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number> phi(data, 0);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the discharge
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                      fe_degree_h, fe_degree_hu, fe_degree_hc,
                      n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_diagonal_cell_term_hu(const MatrixFree<dim, Number>&               data,
                                 Vec&                                         dst,
                                 const unsigned int&                          ,
                                 const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
          a vector which is 1 for the node of interest and 0 elsewhere. This is what 'tmp' does. ---*/
    FEEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number> phi(data, 1);
    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
    Tensor<1, dim, VectorizedArray<Number>> tmp;
    for(unsigned int d = 0; d < dim; ++d) {
      tmp[d] = make_vectorized_array<Number>(1.0);
    }

    if(IMEX_stage <= n_stages) {
      /*--- Since here we have just one 'src' vector, but we also need to deal with the current depth and discharge,
            we employ the auxiliary vectors where we set this information ---*/
      FEEvaluation<dim, fe_degree_h, n_q_points_1d_hu, 1, Number>    phi_h_curr(data, 0);
      FEEvaluation<dim, fe_degree_hu, n_q_points_1d_hu, dim, Number> phi_hu_curr(data, 1);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_h_curr.reinit(cell);
        phi_h_curr.gather_evaluate(h_curr, EvaluationFlags::values);

        phi_hu_curr.reinit(cell);
        phi_hu_curr.gather_evaluate(hu_curr, EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all dofs ---*/
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          }
          phi.submit_dof_value(tmp, i);
          phi.evaluate(EvaluationFlags::values);

          /*--- Loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& h_s      = phi_h_curr.get_value(q);

            const auto& hu_s     = phi_hu_curr.get_value(q);
            const auto& mod_hu_s = std::sqrt(scalar_product(hu_s, hu_s));

            const auto& gamma    = 0.0; // TODO: Add friction as a function of h_s and hu_s

            phi.submit_value((1.0 + a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*(gamma/h_s))*phi.get_value(q), q);
          }

          phi.integrate(EvaluationFlags::values);
          diagonal[i] = phi.get_dof_value(i);
        }

        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          phi.submit_dof_value(diagonal[i], i);
        }
        phi.distribute_local_to_global(dst);
      }
    }
    else {
      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);

        /*--- Loop over all dofs ---*/
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          }
          phi.submit_dof_value(tmp, i);
          phi.evaluate(EvaluationFlags::values);

          /*--- Loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            phi.submit_value(phi.get_value(q), q);
          }

          phi.integrate(EvaluationFlags::values);
          diagonal[i] = phi.get_dof_value(i);
        }

        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          phi.submit_dof_value(diagonal[i], i);
        }
        phi.distribute_local_to_global(dst);
      }
    }
  }


  // Assemble diagonal cell term for the tracer
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_h, fe_degree_hu, fe_degree_hc,
                  n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  assemble_diagonal_cell_term_hc(const MatrixFree<dim, Number>&               data,
                                 Vec&                                         dst,
                                 const unsigned int&                          ,
                                 const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_hc, n_q_points_1d_hu, 1, Number> phi(data, 2);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Compute diagonal of various steps
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_h, int fe_degree_hu, int fe_degree_hc,
           int n_q_points_1d_h, int n_q_points_1d_hu, int n_q_points_1d_hc, typename Vec>
  void SWOperator<dim, n_stages,
                      fe_degree_h, fe_degree_hu, fe_degree_hc,
                      n_q_points_1d_h, n_q_points_1d_hu, n_q_points_1d_hc, Vec>::
  compute_diagonal() {
    AssertIndexRange(SW_stage, 7);
    Assert(SW_stage > 0, ExcInternalError());

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    const unsigned int dummy = 0;

    if(SW_stage == 1 || SW_stage == 4) {
      this->data->initialize_dof_vector(inverse_diagonal, 0);

      this->data->cell_loop(&SWOperator::assemble_diagonal_cell_term_h,
                            this, inverse_diagonal, dummy, false);
    }
    else if(SW_stage == 2 || SW_stage == 5) {
      this->data->initialize_dof_vector(inverse_diagonal, 1);

      this->data->cell_loop(&SWOperator::assemble_diagonal_cell_term_hu,
                            this, inverse_diagonal, dummy, false);
    }
    else if(SW_stage == 3 || SW_stage == 6) {
      this->data->initialize_dof_vector(inverse_diagonal, 2);

      this->data->cell_loop(&SWOperator::assemble_diagonal_cell_term_hc,
                            this, inverse_diagonal, dummy, false);
    }
    else {
      Assert(false, ExcInternalError());
    }

    /*--- For the preconditioner, we actually need the inverse of the diagonal ---*/
    for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != 0.0,
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
    }
  }

} // End of namespace
