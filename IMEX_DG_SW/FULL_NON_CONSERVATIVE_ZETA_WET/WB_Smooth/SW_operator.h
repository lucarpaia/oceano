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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  class SWOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    SWOperator(); /*--- Default constructor ---*/

    SWOperator(RunTimeParameters::Data_Storage& data); /*--- Constructor with some input related data ---*/

    void set_dt(const double time_step); /*--- Setter of the time-step. This is useful both for multigrid purposes and also
                                               in case of modifications of the time step. ---*/

    void set_IMEX_stage(const unsigned int stage); /*--- Setter of the IMEX stage. ---*/

    void set_SW_stage(const unsigned int stage); /*--- Setter of the equation currently under solution. ---*/

    void set_zeta_curr(const Vec& src); /*--- Setter of the current height. This is for the assembling of the bilinear forms
                                              where only one source vector can be passed in input. ---*/

    void set_u_curr(const Vec& src); /*--- Setter of the current velocity. This is for the assembling of the bilinear forms
                                           where only one source vector can be passed in input. ---*/

    void vmult_rhs_zeta(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs
                                                                            for the height. ---*/

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
    Vec zeta_curr,
        u_curr;

    /*-- Auxiliary function for the bathymetry ---*/
    EquationData::Bathymetry<dim, Number> zb;

    /*--- Assembler functions for the rhs related to the height equation. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_zeta(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const std::vector<Vec>&                      src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_zeta(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const std::vector<Vec>&                      src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_zeta(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler function related to the bilinear form of the height equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_zeta(const MatrixFree<dim, Number>&               data,
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

    /*--- Assembler functions for the diagonal part of the matrix for the height equation. ---*/
    void assemble_diagonal_cell_term_zeta(const MatrixFree<dim, Number>&               data,
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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  SWOperator<dim, n_stages,
             fe_degree_zeta, fe_degree_u, fe_degree_c,
             n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  SWOperator(): MatrixFreeOperators::Base<dim, Vec>(), dt(),
                gamma(2.0 - std::sqrt(2.0)), a21(gamma),
                a31(0.5), a32(0.5), a(n_stages, std::vector<double>(n_stages)),
                a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0), a33_tilde(1.0 - std::sqrt(2)/2.0),
                a_tilde(n_stages, std::vector<double>(n_stages)),
                b1(0.5 - 0.25*gamma), b2(0.5 - 0.25*gamma), b3(0.5*gamma), b(n_stages), b_tilde(b),
                IMEX_stage(1), SW_stage(1), zb() {
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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  SWOperator<dim, n_stages,
             fe_degree_zeta, fe_degree_u, fe_degree_c,
             n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  SWOperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(), dt(data.dt),
                                                     gamma(2.0 - std::sqrt(2.0)), a21(gamma),
                                                     a31(0.5), a32(0.5), a(n_stages, std::vector<double>(n_stages)),
                                                     a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                                                     a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0),
                                                     a33_tilde(1.0 - std::sqrt(2)/2.0), a_tilde(n_stages, std::vector<double>(n_stages)),
                                                     b1(0.5 - 0.25*gamma), b2(0.5 - 0.25*gamma), b3(0.5*gamma), b(n_stages), b_tilde(b),
                                                     IMEX_stage(1), SW_stage(1), zb(data.initial_time) {
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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of IMEX stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  set_IMEX_stage(const unsigned int stage) {
    AssertIndexRange(stage, n_stages + 2);
    Assert(stage > 0, ExcInternalError());

    IMEX_stage = stage;
  }


  // Setter of SW stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  set_SW_stage(const unsigned int stage) {
    AssertIndexRange(stage, 7);
    Assert(stage > 0, ExcInternalError());

    SW_stage = stage;
  }


  // Setter of current height
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  set_zeta_curr(const Vec& src) {
    zeta_curr = src;
    zeta_curr.update_ghost_values();
  }


  // Setter of current velocity
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  set_u_curr(const Vec& src) {
    u_curr = src;
    u_curr.update_ghost_values();
  }


  // Assemble rhs cell term for the height equation
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_rhs_cell_term_zeta(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const std::vector<Vec>&                      src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEEvaluation_zeta = FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number>;
    using FEEvaluation_u    = FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the density and
      the velocity and previous stages. 'phi' will be used only to 'submit' the result.
      The second argument specifies which dof handler has to be used (in this implementation 0 stands for
      height, 1 for velocity and 2 for tracer). ---*/
      FEEvaluation_zeta              phi(data, 0);
      std::vector<FEEvaluation_zeta> phi_zeta(IMEX_stage - 1, FEEvaluation_zeta(data, 0));
      std::vector<FEEvaluation_u>    phi_u(IMEX_stage - 1, FEEvaluation_u(data, 1));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        /*--- Now we need to assign the current cell to each FEEvaluation object and then to specify which src vector
        it has to read (the proper order is clearly delegated to the user, which has to pay attention in the function
        call to be coherent). All these considerations are valid also for the other assembler functions ---*/
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_zeta[s - 1].reinit(cell);
          phi_zeta[s - 1].gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
          phi_u[s - 1].reinit(cell);
          phi_u[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the height at the previous step (always needed) ---*/
          const auto& zeta_old = phi_zeta[0].get_value(q);

          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& h_s = phi_zeta[s - 1].get_value(q) + zb_q;
            const auto& u_s = phi_u[s - 1].get_value(q);

            flux += a[IMEX_stage - 1][s - 1]*dt*h_s*u_s;
          }

          phi.submit_value(zeta_old, q);
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
      FEEvaluation_zeta              phi(data, 0);
      std::vector<FEEvaluation_zeta> phi_zeta(IMEX_stage - 1, FEEvaluation_zeta(data, 0));
      std::vector<FEEvaluation_u>    phi_u(IMEX_stage - 1, FEEvaluation_u(data, 1));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_zeta[s - 1].reinit(cell);
          phi_zeta[s - 1].gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
          phi_u[s - 1].reinit(cell);
          phi_u[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the height at the previous step (always needed) ---*/
          const auto& zeta_old = phi_zeta[0].get_value(q);

          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& h_s = phi_zeta[s - 1].get_value(q) + zb_q;
            const auto& u_s = phi_u[s - 1].get_value(q);

            flux += b[s - 1]*dt*h_s*u_s;
          }

          phi.submit_value(zeta_old, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Assemble rhs face term for the height
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_rhs_face_term_zeta(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const std::vector<Vec>&                      src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEFaceEvaluation_zeta = FEFaceEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number>;
    using FEFaceEvaluation_u    = FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities.
            'true' means that we are reading the information from 'inside', whereas 'false' from 'outside' ---*/
      FEFaceEvaluation_zeta phi_m(data, true, 0),
                            phi_p(data, false, 0),
                            phi_zeta_m(data, true, 0),
                            phi_zeta_p(data, false, 0);
      FEFaceEvaluation_u    phi_u_m(data, true, 1),
                            phi_u_p(data, false, 1);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_zeta_m.reinit(face);
        phi_zeta_p.reinit(face);
        phi_u_m.reinit(face);
        phi_u_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi_m.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          const auto& n_plus = phi_m.get_normal_vector(q); /*--- Notice that the unit normal vector is the same from
                                                                 'both sides'. ---*/

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_zeta_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_zeta_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_u_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
            phi_u_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);

            const auto& zeta_s_m    = phi_zeta_m.get_value(q);
            const auto& h_s_m       = zeta_s_m + zb_q;
            const auto& zeta_s_p    = phi_zeta_p.get_value(q);
            const auto& h_s_p       = zeta_s_p + zb_q;
            const auto& u_s_m       = phi_u_m.get_value(q);
            const auto& u_s_p       = phi_u_p.get_value(q);

            const auto& avg_flux_s  = 0.5*(h_s_m*u_s_m + h_s_p*u_s_p);
            const auto& lambda_s    = std::max(std::abs(scalar_product(u_s_m, n_plus)) +
                                               std::sqrt(EquationData::g*h_s_m),
                                               std::abs(scalar_product(u_s_p, n_plus)) +
                                               std::sqrt(EquationData::g*h_s_p));
            const auto& jump_zeta_s = zeta_s_m - zeta_s_p;

            flux += a[IMEX_stage - 1][s - 1]*dt*(scalar_product(avg_flux_s, n_plus) + 0.5*lambda_s*jump_zeta_s);
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
      FEFaceEvaluation_zeta phi_m(data, true, 0),
                            phi_p(data, false, 0),
                            phi_zeta_m(data, true, 0),
                            phi_zeta_p(data, false, 0);
      FEFaceEvaluation_u    phi_u_m(data, true, 1),
                            phi_u_p(data, false, 1);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_zeta_m.reinit(face);
        phi_zeta_p.reinit(face);
        phi_u_m.reinit(face);
        phi_u_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi_m.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          const auto& n_plus = phi_m.get_normal_vector(q); /*--- Notice that the unit normal vector is the same from
                                                                 'both sides'. ---*/

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_zeta_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_zeta_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_u_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
            phi_u_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);

            const auto& zeta_s_m    = phi_zeta_m.get_value(q);
            const auto& h_s_m       = zeta_s_m + zb_q;
            const auto& zeta_s_p    = phi_zeta_p.get_value(q);
            const auto& h_s_p       = zeta_s_p + zb_q;
            const auto& u_s_m       = phi_u_m.get_value(q);
            const auto& u_s_p       = phi_u_p.get_value(q);

            const auto& avg_flux_s  = 0.5*(h_s_m*u_s_m + h_s_p*u_s_p);
            const auto& lambda_s    = std::max(std::abs(scalar_product(u_s_m, n_plus)) +
                                               std::sqrt(EquationData::g*h_s_m),
                                               std::abs(scalar_product(u_s_p, n_plus)) +
                                               std::sqrt(EquationData::g*h_s_p));
            const auto& jump_zeta_s = zeta_s_m - zeta_s_p;

            flux += b[s - 1]*dt*(scalar_product(avg_flux_s, n_plus) + 0.5*lambda_s*jump_zeta_s);
          }

          phi_m.submit_value(-flux, q);
          phi_p.submit_value(flux, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // Put together all the previous steps for height
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  vmult_rhs_zeta(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&SWOperator::assemble_rhs_cell_term_zeta,
                     &SWOperator::assemble_rhs_face_term_zeta,
                     &SWOperator::assemble_rhs_boundary_term_zeta,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the height
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_cell_term_zeta(const MatrixFree<dim, Number>&               data,
                          Vec&                                         dst,
                          const Vec&                                   src,
                          const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number> phi(data, 0);

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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_rhs_cell_term_hu(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const std::vector<Vec>&                      src,
                            const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEEvaluation_zeta = FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number>;
    using FEEvaluation_u    = FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_u                 phi(data, 1);
      std::vector<FEEvaluation_u>    phi_u(IMEX_stage - 1, FEEvaluation_u(data, 1));
      std::vector<FEEvaluation_zeta> phi_zeta(IMEX_stage - 1, FEEvaluation_zeta(data, 0));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_zeta[s - 1].reinit(cell);
          phi_zeta[s - 1].gather_evaluate(src[2*(s-1)], EvaluationFlags::values | EvaluationFlags::gradients);
          phi_u[s - 1].reinit(cell);
          phi_u[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          /*--- Compute the discharge at the previous step (always necessary) ---*/
          const auto& h_old = phi_zeta[0].get_value(q) + zb_q;
          const auto& u_old = phi_u[0].get_value(q);

          /*--- Compute the quantites at the previous stages ---*/
          Tensor<2, dim, VectorizedArray<Number>> flux;
          Tensor<1, dim, VectorizedArray<Number>> non_cons_flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& h_s         = phi_zeta[s - 1].get_value(q) + zb_q;
            const auto& u_s         = phi_u[s - 1].get_value(q);

            const auto& grad_zeta_s = phi_zeta[s - 1].get_gradient(q);

            flux += a[IMEX_stage - 1][s - 1]*dt*outer_product(h_s*u_s, u_s);
            non_cons_flux += a[IMEX_stage - 1][s - 1]*dt*EquationData::g*h_s*grad_zeta_s;
          }

          phi.submit_value(h_old*u_old - non_cons_flux, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_u                 phi(data, 1);
      std::vector<FEEvaluation_u>    phi_u(IMEX_stage - 1, FEEvaluation_u(data, 1));
      std::vector<FEEvaluation_zeta> phi_zeta(IMEX_stage - 1, FEEvaluation_zeta(data, 0));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_zeta[s - 1].reinit(cell);
          phi_zeta[s - 1].gather_evaluate(src[2*(s-1)], EvaluationFlags::values | EvaluationFlags::gradients);
          phi_u[s - 1].reinit(cell);
          phi_u[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          /*--- Compute the discharge at the previous step (always necessary) ---*/
          const auto& h_old = phi_zeta[0].get_value(q) + zb_q;
          const auto& u_old = phi_u[0].get_value(q);

          /*--- Compute the quantites at the previous stages ---*/
          Tensor<2, dim, VectorizedArray<Number>> flux;
          Tensor<1, dim, VectorizedArray<Number>> non_cons_flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& h_s         = phi_zeta[s - 1].get_value(q) + zb_q;
            const auto& u_s         = phi_u[s - 1].get_value(q);

            const auto& grad_zeta_s = phi_zeta[s - 1].get_gradient(q);

            flux += b[s - 1]*dt*outer_product(h_s*u_s, u_s);
            non_cons_flux += b[s - 1]*dt*EquationData::g*h_s*grad_zeta_s;
          }

          phi.submit_value(h_old*u_old - non_cons_flux, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Assemble rhs face term for the discharge
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_rhs_face_term_hu(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const std::vector<Vec>&                      src,
                            const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEFaceEvaluation_u    = FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number>;
    using FEFaceEvaluation_zeta = FEFaceEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u    phi_m(data, true, 1),
                            phi_p(data, false, 1),
                            phi_u_m(data, true, 1),
                            phi_u_p(data, false, 1);
      FEFaceEvaluation_zeta phi_zeta_m(data, true, 0),
                            phi_zeta_p(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_zeta_m.reinit(face);
        phi_zeta_p.reinit(face);
        phi_u_m.reinit(face);
        phi_u_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi_m.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          const auto& n_plus = phi_p.get_normal_vector(q);

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          VectorizedArray<Number> non_cons_flux_p = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> non_cons_flux_m = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_zeta_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_zeta_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_u_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
            phi_u_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);

            const auto& zeta_s_m                = phi_zeta_m.get_value(q);
            const auto& h_s_m                   = zeta_s_m + zb_q;
            const auto& zeta_s_p                = phi_zeta_p.get_value(q);
            const auto& h_s_p                   = zeta_s_p + zb_q;
            const auto& u_s_m                   = phi_u_m.get_value(q);
            const auto& u_s_p                   = phi_u_p.get_value(q);

            const auto& avg_tensor_product_flux = 0.5*(outer_product(h_s_m*u_s_m, u_s_m) +
                                                       outer_product(h_s_p*u_s_p, u_s_p));
            const auto& lambda_s                = std::max(std::abs(scalar_product(u_s_m, n_plus)) +
                                                           std::sqrt(EquationData::g*h_s_m),
                                                           std::abs(scalar_product(u_s_p, n_plus)) +
                                                           std::sqrt(EquationData::g*h_s_p));
            const auto& jump_hu_s               = h_s_m*u_s_m - h_s_p*u_s_p;

            /*--- Add contribution due to non-conservative term ---*/
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
      FEFaceEvaluation_u    phi_m(data, true, 1),
                            phi_p(data, false, 1),
                            phi_u_m(data, true, 1),
                            phi_u_p(data, false, 1);
      FEFaceEvaluation_zeta phi_zeta_m(data, true, 0),
                            phi_zeta_p(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_zeta_m.reinit(face);
        phi_zeta_p.reinit(face);
        phi_u_m.reinit(face);
        phi_u_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi_m.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          const auto& n_plus = phi_p.get_normal_vector(q);

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          VectorizedArray<Number> non_cons_flux_p = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> non_cons_flux_m = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_zeta_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_zeta_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_u_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
            phi_u_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);

            const auto& zeta_s_m                = phi_zeta_m.get_value(q);
            const auto& h_s_m                   = zeta_s_m + zb_q;
            const auto& zeta_s_p                = phi_zeta_p.get_value(q);
            const auto& h_s_p                   = zeta_s_p + zb_q;
            const auto& u_s_m                   = phi_u_m.get_value(q);
            const auto& u_s_p                   = phi_u_p.get_value(q);

            const auto& avg_tensor_product_flux = 0.5*(outer_product(h_s_m*u_s_m, u_s_m) +
                                                       outer_product(h_s_p*u_s_p, u_s_p));
            const auto& lambda_s                = std::max(std::abs(scalar_product(u_s_m, n_plus)) +
                                                           std::sqrt(EquationData::g*h_s_m),
                                                           std::abs(scalar_product(u_s_p, n_plus)) +
                                                           std::sqrt(EquationData::g*h_s_p));
            const auto& jump_hu_s               = h_s_m*u_s_m - h_s_p*u_s_p;

            /*--- Add contribution due to non-conservative term ---*/
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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_cell_term_hu(const MatrixFree<dim, Number>&               data,
                        Vec&                                         dst,
                        const Vec&                                   src,
                        const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 1);

    if(IMEX_stage <= n_stages) {
      /*--- Since here we have just one 'src' vector, but we also need to deal with the current height and velocity,
            we employ the auxiliary vectors where we set this information ---*/
      FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number> phi_zeta_curr(data, 0);
      //FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_curr(data, 1);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_zeta_curr.reinit(cell);
        phi_zeta_curr.gather_evaluate(zeta_curr, EvaluationFlags::values);

        /*phi_u_curr.reinit(cell);
        phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::values);*/

        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          const auto& h_s      = phi_zeta_curr.get_value(q) + zb_q;

          /*const auto& u_s      = phi_u_curr.get_value(q);
          const auto& mod_hu_s = std::sqrt(scalar_product(h_s*u_s, h_s*u_s));

          const auto& gamma    = 0.0; TODO: Add friction as a function of h_s and hu_s

          phi.submit_value((1.0 + a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*gamma/h_s)*h_s*phi.get_value(q), q);*/

          phi.submit_value(h_s*phi.get_value(q), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- Since here we have just one 'src' vector, but we also need to deal with the current height,
            we employ the auxiliary vectors where we set this information ---*/
      FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number> phi_zeta_curr(data, 0);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_zeta_curr.reinit(cell);
        phi_zeta_curr.gather_evaluate(zeta_curr, EvaluationFlags::values);

        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& point_vectorized = phi.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          const auto& h_s = phi_zeta_curr.get_value(q) + zb_q;

          phi.submit_value(h_s*phi.get_value(q), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // Assemble rhs cell term for the tracer equation
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_rhs_cell_term_hc(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const std::vector<Vec>&                      src,
                            const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEEvaluation_zeta = FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number>;
    using FEEvaluation_u    = FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number>;
    using FEEvaluation_c    = FEEvaluation<dim, fe_degree_c, n_q_points_1d_u, 1, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_c                 phi(data, 2);
      std::vector<FEEvaluation_c>    phi_c(IMEX_stage - 1, FEEvaluation_c(data, 2));
      std::vector<FEEvaluation_u>    phi_u(IMEX_stage - 1, FEEvaluation_u(data, 1));
      std::vector<FEEvaluation_zeta> phi_zeta(IMEX_stage - 1, FEEvaluation_zeta(data, 0));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_zeta[s - 1].reinit(cell);
          phi_zeta[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_u[s - 1].reinit(cell);
          phi_u[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_c[s - 1].reinit(cell);
          phi_c[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          /*--- Compute the tracer at the previous step (always needed) ---*/
          const auto& h_old = phi_zeta[0].get_value(q) + zb_q;
          const auto& c_old = phi_c[0].get_value(q);

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& c_s = phi_c[s - 1].get_value(q);
            const auto& u_s = phi_u[s - 1].get_value(q);
            const auto& h_s = phi_zeta[s - 1].get_value(q) + zb_q;

            flux += a[IMEX_stage - 1][s - 1]*dt*(h_s*u_s*c_s);
          }

          phi.submit_value(h_old*c_old, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_c                 phi(data, 2);
      std::vector<FEEvaluation_c>    phi_c(IMEX_stage - 1, FEEvaluation_c(data, 2));
      std::vector<FEEvaluation_u>    phi_u(IMEX_stage - 1, FEEvaluation_u(data, 1));
      std::vector<FEEvaluation_zeta> phi_zeta(IMEX_stage - 1, FEEvaluation_zeta(data, 0));

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
          phi_zeta[s - 1].reinit(cell);
          phi_zeta[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_u[s - 1].reinit(cell);
          phi_u[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_c[s - 1].reinit(cell);
          phi_c[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
        }

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          /*--- Compute the tracer at the previous step (always needed) ---*/
          const auto& h_old = phi_zeta[0].get_value(q) + zb_q;
          const auto& c_old = phi_c[0].get_value(q);

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& c_s = phi_c[s - 1].get_value(q);
            const auto& u_s = phi_u[s - 1].get_value(q);
            const auto& h_s = phi_zeta[s - 1].get_value(q) + zb_q;

            flux += b[s - 1]*dt*(h_s*u_s*c_s);
          }

          phi.submit_value(h_old*c_old, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Assemble rhs face term for the tracer
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_rhs_face_term_hc(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const std::vector<Vec>&                      src,
                            const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- Define typedef for sake of readibility and convenience ----*/
    using FEFaceEvaluation_c    = FEFaceEvaluation<dim, fe_degree_c, n_q_points_1d_u, 1, Number>;
    using FEFaceEvaluation_u    = FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number>;
    using FEFaceEvaluation_zeta = FEFaceEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number>;

    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_c    phi_m(data, true, 2),
                            phi_p(data, false, 2),
                            phi_c_m(data, true, 2),
                            phi_c_p(data, false, 2);
      FEFaceEvaluation_u    phi_u_m(data, true, 1),
                            phi_u_p(data, false, 1);
      FEFaceEvaluation_zeta phi_zeta_m(data, true, 0),
                            phi_zeta_p(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_c_m.reinit(face);
        phi_c_p.reinit(face);
        phi_u_m.reinit(face);
        phi_u_p.reinit(face);
        phi_zeta_m.reinit(face);
        phi_zeta_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi_m.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          const auto& n_plus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_zeta_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_zeta_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_u_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_u_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_c_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_c_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& c_s_m = phi_c_m.get_value(q);
            const auto& c_s_p = phi_c_p.get_value(q);
            const auto& u_s_m = phi_u_m.get_value(q);
            const auto& u_s_p = phi_u_p.get_value(q);
            const auto& h_s_m = phi_zeta_m.get_value(q) + zb_q;
            const auto& h_s_p = phi_zeta_p.get_value(q) + zb_q;

            const auto& avg_flux_s = 0.5*(h_s_m*u_s_m*c_s_m + h_s_p*u_s_p*c_s_p);
            const auto& lambda_s   = std::max(std::abs(scalar_product(u_s_m, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_m),
                                              std::abs(scalar_product(u_s_p, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_p));
            const auto& jump_hc_s  = h_s_m*c_s_m - h_s_p*c_s_p;

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
      FEFaceEvaluation_c    phi_m(data, true, 2),
                            phi_p(data, false, 2),
                            phi_c_m(data, true, 2),
                            phi_c_p(data, false, 2);
      FEFaceEvaluation_u    phi_u_m(data, true, 1),
                            phi_u_p(data, false, 1);
      FEFaceEvaluation_zeta phi_zeta_m(data, true, 0),
                            phi_zeta_p(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_c_m.reinit(face);
        phi_c_p.reinit(face);
        phi_u_m.reinit(face);
        phi_u_p.reinit(face);
        phi_zeta_m.reinit(face);
        phi_zeta_p.reinit(face);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi_m.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          const auto& n_plus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux = make_vectorized_array<Number>(0.0);
          for(unsigned int s = 1; s <= IMEX_stage - 1; ++s) {
            phi_zeta_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_zeta_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_u_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_u_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_c_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_c_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& c_s_m = phi_c_m.get_value(q);
            const auto& c_s_p = phi_c_p.get_value(q);
            const auto& u_s_m = phi_u_m.get_value(q);
            const auto& u_s_p = phi_u_p.get_value(q);
            const auto& h_s_m = phi_zeta_m.get_value(q) + zb_q;
            const auto& h_s_p = phi_zeta_p.get_value(q) + zb_q;

            const auto& avg_flux_s = 0.5*(h_s_m*u_s_m*c_s_m + h_s_p*u_s_p*c_s_p);
            const auto& lambda_s   = std::max(std::abs(scalar_product(u_s_m, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_m),
                                              std::abs(scalar_product(u_s_p, n_plus)) +
                                              std::sqrt(EquationData::g*h_s_p));
            const auto& jump_hc_s  = h_s_m*c_s_m - h_s_p*c_s_p;

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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_cell_term_hc(const MatrixFree<dim, Number>&               data,
                        Vec&                                         dst,
                        const Vec&                                   src,
                        const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_c, n_q_points_1d_u, 1, Number>    phi(data, 2);
    FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number> phi_zeta_curr(data, 0);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_zeta_curr.reinit(cell);
      phi_zeta_curr.gather_evaluate(zeta_curr, EvaluationFlags::values);

      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        /*--- Evaluate the bathymetry ---*/
        const auto& point_vectorized = phi.quadrature_point(q);
        VectorizedArray<Number> zb_q;
        for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
          Point<dim> point;
          for(unsigned int d = 0; d < dim; ++d) {
            point[d] = point_vectorized[d][v];
          }
          zb_q[v] = zb.value(point);
        }

        const auto& h_s = phi_zeta_curr.get_value(q) + zb_q;

        phi.submit_value(h_s*phi.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(SW_stage, 7);
    Assert(SW_stage > 0, ExcInternalError());

    if(SW_stage == 1 || SW_stage == 4) {
      this->data->cell_loop(&SWOperator::assemble_cell_term_zeta,
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


  // Assemble diagonal cell term for the height
  //
  template<int dim, unsigned int n_stages,
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_diagonal_cell_term_zeta(const MatrixFree<dim, Number>&               data,
                                   Vec&                                         dst,
                                   const unsigned int&                          ,
                                   const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number> phi(data, 0);

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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                      fe_degree_zeta, fe_degree_u, fe_degree_c,
                      n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_diagonal_cell_term_hu(const MatrixFree<dim, Number>&               data,
                                 Vec&                                         dst,
                                 const unsigned int&                          ,
                                 const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
          a vector which is 1 for the node of interest and 0 elsewhere. This is what 'tmp' does. ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 1);
    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
    Tensor<1, dim, VectorizedArray<Number>> tmp;
    for(unsigned int d = 0; d < dim; ++d) {
      tmp[d] = make_vectorized_array<Number>(1.0);
    }

    if(IMEX_stage <= n_stages) {
      /*--- Since here we have just one 'src' vector, but we also need to deal with the current height and velocity,
            we employ the auxiliary vectors where we set this information ---*/
      FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number> phi_zeta_curr(data, 0);
      //FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_curr(data, 1);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_zeta_curr.reinit(cell);
        phi_zeta_curr.gather_evaluate(zeta_curr, EvaluationFlags::values);

        /*phi_u_curr.reinit(cell);
        phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::values);*/

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
            /*--- Evaluate the bathymetry ---*/
            const auto& point_vectorized = phi.quadrature_point(q);
            VectorizedArray<Number> zb_q;
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point;
              for(unsigned int d = 0; d < dim; ++d) {
                point[d] = point_vectorized[d][v];
              }
              zb_q[v] = zb.value(point);
            }

            const auto& h_s      = phi_zeta_curr.get_value(q) + zb_q;

            /*const auto& u_s     = phi_u_curr.get_value(q);
            const auto& mod_hu_s = std::sqrt(scalar_product(h_s*u_s, h_s*u_s));

            const auto& gamma    = 0.0; TODO: Add friction as a function of h_s and hu_s

            phi.submit_value((1.0 + a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*gamma/h_s)*h_s*phi.get_value(q), q);*/

            phi.submit_value(h_s*phi.get_value(q), q);
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
      /*--- Since here we have just one 'src' vector, but we also need to deal with the current height,
            we employ the auxiliary vectors where we set this information ---*/
      FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number> phi_zeta_curr(data, 0);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_zeta_curr.reinit(cell);
        phi_zeta_curr.gather_evaluate(zeta_curr, EvaluationFlags::values);

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
            const auto& point_vectorized = phi.quadrature_point(q);
            VectorizedArray<Number> zb_q;
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point;
              for(unsigned int d = 0; d < dim; ++d) {
                point[d] = point_vectorized[d][v];
              }
              zb_q[v] = zb.value(point);
            }

            const auto& h_s = phi_zeta_curr.get_value(q) + zb_q;

            phi.submit_value(h_s*phi.get_value(q), q);
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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                  fe_degree_zeta, fe_degree_u, fe_degree_c,
                  n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  assemble_diagonal_cell_term_hc(const MatrixFree<dim, Number>&               data,
                                 Vec&                                         dst,
                                 const unsigned int&                          ,
                                 const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities. ---*/
    FEEvaluation<dim, fe_degree_c, n_q_points_1d_u, 1, Number>    phi(data, 2);
    FEEvaluation<dim, fe_degree_zeta, n_q_points_1d_u, 1, Number> phi_zeta_curr(data, 0);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      phi_zeta_curr.reinit(cell);
      phi_zeta_curr.gather_evaluate(zeta_curr, EvaluationFlags::values);

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
          /*--- Evaluate the bathymetry ---*/
          const auto& point_vectorized = phi.quadrature_point(q);
          VectorizedArray<Number> zb_q;
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            zb_q[v] = zb.value(point);
          }

          const auto& h_s = phi_zeta_curr.get_value(q) + zb_q;

          phi.submit_value(h_s*phi.get_value(q), q);
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
           int fe_degree_zeta, int fe_degree_u, int fe_degree_c,
           int n_q_points_1d_zeta, int n_q_points_1d_u, int n_q_points_1d_c, typename Vec>
  void SWOperator<dim, n_stages,
                      fe_degree_zeta, fe_degree_u, fe_degree_c,
                      n_q_points_1d_zeta, n_q_points_1d_u, n_q_points_1d_c, Vec>::
  compute_diagonal() {
    AssertIndexRange(SW_stage, 7);
    Assert(SW_stage > 0, ExcInternalError());

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    const unsigned int dummy = 0;

    if(SW_stage == 1 || SW_stage == 4) {
      this->data->initialize_dof_vector(inverse_diagonal, 0);

      this->data->cell_loop(&SWOperator::assemble_diagonal_cell_term_zeta,
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
