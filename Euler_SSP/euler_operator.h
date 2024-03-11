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

// This is the class that implements the discretization for the advection equation
//
namespace Euler {
  using namespace dealii;

  // @sect{ <code>EULEROperator::EULEROperator</code> }
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  class EULEROperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    EULEROperator(); /*--- Default constructor ---*/

    EULEROperator(RunTimeParameters::Data_Storage& data); /*--- Constructor with some input related data ---*/

    void set_dt(const double time_step); /*--- Setter of the time-step. This is useful both for multigrid purposes and also
                                               in case of modifications of the time step. ---*/

    void set_Mach(const double Ma_); /*--- Setter of the Mach number. This is useful for multigrid purpose. ---*/

    void set_Euler_stage(const unsigned int stage); /*--- Setter of the equation currently under solution. ---*/

    void vmult_rhs_rho_update(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs
                                                                                  for the density. ---*/

    void vmult_rhs_momentum_update(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs
                                                                                       for the momentum. ---*/

    void vmult_rhs_energy_update(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs
                                                                                     for the energy. ---*/

    virtual void compute_diagonal() override {} /*--- Compute the diagonal for several preconditioners ---*/

  protected:
    double Ma; /*--- Mach number. ---*/
    double dt; /*--- Time step. ---*/

    unsigned int Euler_stage; /*--- Flag for the equation actually solved ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
    /*--- Assembler functions for the rhs related to the continuity equation. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const std::vector<Vec>&                      src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_rho_update(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const std::vector<Vec>&                      src,
                                           const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_rho_update(const MatrixFree<dim, Number>&               data,
                                               Vec&                                         dst,
                                               const std::vector<Vec>&                      src,
                                               const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler function related to the bilinear form of the continuity equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the rhs related to the momentum equation. ---*/
    void assemble_rhs_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const std::vector<Vec>&                      src,
                                                const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const std::vector<Vec>&                      src,
                                                const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                    Vec&                                         dst,
                                                    const std::vector<Vec>&                      src,
                                                    const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler function related to the bilinear form of the momentum equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the rhs related to the energy equation. ---*/
    void assemble_rhs_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const std::vector<Vec>&                      src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_energy_update(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const std::vector<Vec>&                      src,
                                              const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_energy_update(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const std::vector<Vec>&                      src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler function related to the bilinear form of the energy equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
                                          Vec&                                         dst,
                                          const Vec&                                   src,
                                          const std::pair<unsigned int, unsigned int>& cell_range) const;
  };


  // Default constructor
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  EULEROperator(): MatrixFreeOperators::Base<dim, Vec>(), Ma(), dt(), Euler_stage(1) {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  EULEROperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(),
                                                        Ma(data.Mach), dt(data.dt), Euler_stage(1) {}


  // Setter of time-step
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of Mach number
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  set_Mach(const double Ma_) {
    Ma = Ma_;
  }


  // Setter of Euler stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  set_Euler_stage(const unsigned int stage) {
    AssertIndexRange(stage, 4);
    Assert(stage > 0, ExcInternalError());

    Euler_stage = stage;
  }


  // Assemble rhs cell term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  assemble_rhs_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const std::vector<Vec>&                      src,
                                    const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read the old density and
    the available quantities. 'phi' will be used only to 'submit' the result.
    The second argument specifies which dof handler has to be used (in this implementation 0 stands for
    velocity, 1 for pressure and 2 for density). ---*/
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2),
                                                                 phi_rho(data, 2);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_rhou(data, 0);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      /*--- Now we need to assign the current cell to each FEEvaluation object and then to specify which src vector
      it has to read (the proper order is clearly delegated to the user, which has to pay attention in the function
      call to be coherent). All these considerations are valid also for the other assembler functions ---*/
      phi_rho.reinit(cell);
      phi_rho.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rhou.reinit(cell);
      phi_rhou.gather_evaluate(src[1], EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& rho  = phi_rho.get_value(q);
        const auto& rhou = phi_rhou.get_value(q);

        phi.submit_value(rho, q);
        /*--- submit_value is used for quantities to be tested against test functions ---*/
        phi.submit_gradient(dt*rhou, q);
        /*--- submit_gradient is used for quantities to be tested against gradient of test functions ---*/
      }

      /*--- 'integrate_scatter' is the responsible of distributing into dst.
            The flag parameter specifies if we are testing against the test function and/or its gradient ---*/
      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // Assemble rhs face term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  assemble_rhs_face_term_rho_update(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const std::vector<Vec>&                      src,
                                    const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read the available quantities.
          'true' means that we are reading the information from 'inside', whereas 'false' from 'outside' ---*/
    FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_p(data, true, 2),
                                                                     phi_m(data, false, 2),
                                                                     phi_rho_p(data, true, 2),
                                                                     phi_rho_m(data, false, 2);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_rhou_p(data, true, 0),
                                                                     phi_rhou_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_rhoE_p(data, true, 1),
                                                                     phi_rhoE_m(data, false, 1);

    /*--- Loop over all faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_rho_p.reinit(face);
      phi_rho_p.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rho_m.reinit(face);
      phi_rho_m.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rhou_p.reinit(face);
      phi_rhou_p.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhou_m.reinit(face);
      phi_rhou_m.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhoE_p.reinit(face);
      phi_rhoE_p.gather_evaluate(src[2], EvaluationFlags::values);
      phi_rhoE_m.reinit(face);
      phi_rhoE_m.gather_evaluate(src[2], EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus   = phi_p.get_normal_vector(q);

        const auto& rho_p    = phi_rho_p.get_value(q);
        const auto& rho_m    = phi_rho_m.get_value(q);
        const auto& rhou_p   = phi_rhou_p.get_value(q);
        const auto& rhou_m   = phi_rhou_m.get_value(q);
        const auto& rhoE_p   = phi_rhoE_p.get_value(q);
        const auto& rhoE_m   = phi_rhoE_m.get_value(q);

        /*--- Compute the pressure using EOS ---*/
        const auto& pres_p   = (EquationData::Cp_Cv - 1.0)*
                               (rhoE_p - 0.5*Ma*Ma*scalar_product(rhou_p, rhou_p)/rho_p);
        const auto& pres_m   = (EquationData::Cp_Cv - 1.0)*
                               (rhoE_m - 0.5*Ma*Ma*scalar_product(rhou_m, rhou_m)/rho_m);

        /*--- Compute the numerical flux ---*/
        const auto& avg_flux = 0.5*(rhou_p + rhou_m);
        const auto& lambda   = std::max(std::sqrt(scalar_product(rhou_p/rho_p, rhou_p/rho_p)) +
                                        1.0/Ma*std::sqrt(std::abs(EquationData::Cp_Cv*pres_p/rho_p)),
                                        std::sqrt(scalar_product(rhou_m/rho_m, rhou_m/rho_m)) +
                                        1.0/Ma*std::sqrt(std::abs(EquationData::Cp_Cv*pres_m/rho_m)));
        const auto& jump_rho = rho_p - rho_m;

        phi_p.submit_value(-dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda*jump_rho), q);
        phi_m.submit_value(dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda*jump_rho), q);
      }

      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  vmult_rhs_rho_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_rho_update,
                     &EULEROperator::assemble_rhs_face_term_rho_update,
                     &EULEROperator::assemble_rhs_boundary_term_rho_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  assemble_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const Vec&                                   src,
                                    const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2);

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


  // Assemble rhs cell term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  assemble_rhs_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0),
                                                                 phi_rhou(data, 0);
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_rhoE(data, 1);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho(data, 2);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho.reinit(cell);
      phi_rho.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rhou.reinit(cell);
      phi_rhou.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhoE.reinit(cell);
      phi_rhoE.gather_evaluate(src[2], EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& rho              = phi_rho.get_value(q);
        const auto& rhou             = phi_rhou.get_value(q);
        const auto& rhoE             = phi_rhoE.get_value(q);

        /*--- Compute the pressure from EOS ---*/
        const auto& pres             = (EquationData::Cp_Cv - 1.0)*(rhoE - 0.5*Ma*Ma*scalar_product(rhou, rhou)/rho);

        /*--- Compute the term coming from integration by parts ---*/
        const auto& tensor_product_u = outer_product(rhou, rhou/rho);
        Tensor<dim, dim, VectorizedArray<Number>> p_times_identity;
        for(unsigned int d = 0; d < dim; ++d) {
          p_times_identity[d][d] = pres;
        }

        phi.submit_value(rhou, q);
        phi.submit_gradient(dt*tensor_product_u + dt/(Ma*Ma)*p_times_identity, q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // Assemble rhs face term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  assemble_rhs_face_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                     phi_m(data, false, 0),
                                                                     phi_rhou_p(data, true, 0),
                                                                     phi_rhou_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_p(data, true, 2),
                                                                     phi_rho_m(data, false, 2);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_rhoE_p(data, true, 1),
                                                                     phi_rhoE_m(data, false, 1);

    /*--- Loop over all faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_rho_p.reinit(face);
      phi_rho_p.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rho_m.reinit(face);
      phi_rho_m.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rhou_p.reinit(face);
      phi_rhou_p.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhou_m.reinit(face);
      phi_rhou_m.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhoE_p.reinit(face);
      phi_rhoE_p.gather_evaluate(src[2], EvaluationFlags::values);
      phi_rhoE_m.reinit(face);
      phi_rhoE_m.gather_evaluate(src[2], EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus               = phi_p.get_normal_vector(q);

        const auto& rho_p                = phi_rho_p.get_value(q);
        const auto& rho_m                = phi_rho_m.get_value(q);
        const auto& rhou_p               = phi_rhou_p.get_value(q);
        const auto& rhou_m               = phi_rhou_m.get_value(q);
        const auto& rhoE_p               = phi_rhoE_p.get_value(q);
        const auto& rhoE_m               = phi_rhoE_m.get_value(q);

        /*--- Centered flux ---*/
        const auto& pres_p               = (EquationData::Cp_Cv - 1.0)*
                                           (rhoE_p - 0.5*Ma*Ma*scalar_product(rhou_p, rhou_p)/rho_p);
        const auto& pres_m               = (EquationData::Cp_Cv - 1.0)*
                                           (rhoE_m - 0.5*Ma*Ma*scalar_product(rhou_m, rhou_m)/rho_m);
        const auto& avg_pres             = 0.5*(pres_p + pres_m);

        const auto& avg_tensor_product_u = 0.5*(outer_product(rhou_p, rhou_p/rho_p) +
                                                outer_product(rhou_m, rhou_m/rho_m));

        /*--- Rusanov flux ---*/
        const auto& lambda               = std::max(std::sqrt(scalar_product(rhou_p/rho_p, rhou_p/rho_p)) +
                                                    1.0/Ma*std::sqrt(std::abs(EquationData::Cp_Cv*pres_p/rho_p)),
                                                    std::sqrt(scalar_product(rhou_m/rho_m, rhou_m/rho_m)) +
                                                    1.0/Ma*std::sqrt(std::abs(EquationData::Cp_Cv*pres_m/rho_m)));
        const auto& jump_rhou            = rhou_p - rhou_m;

        phi_p.submit_value(-dt*(avg_tensor_product_u*n_plus +
                                1.0/(Ma*Ma)*avg_pres*n_plus +
                                0.5*lambda*jump_rhou), q);
        phi_m.submit_value(dt*(avg_tensor_product_u*n_plus +
                               1.0/(Ma*Ma)*avg_pres*n_plus +
                               0.5*lambda*jump_rhou), q);
      }

      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  vmult_rhs_momentum_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_momentum_update,
                     &EULEROperator::assemble_rhs_face_term_momentum_update,
                     &EULEROperator::assemble_rhs_boundary_term_momentum_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  assemble_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over alla quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble rhs cell term for the energy update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  assemble_rhs_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, 1),
                                                                 phi_rhoE(data, 1);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_rhou(data, 0);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho(data, 2);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho.reinit(cell);
      phi_rho.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rhou.reinit(cell);
      phi_rhou.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhoE.reinit(cell);
      phi_rhoE.gather_evaluate(src[2], EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& rho  = phi_rho.get_value(q);
        const auto& rhou = phi_rhou.get_value(q);
        const auto& rhoE = phi_rhoE.get_value(q);

        /*--- Compute the pressure with the EOS ---*/
        const auto& pres = (EquationData::Cp_Cv - 1.0)*
                           (rhoE - 0.5*Ma*Ma*scalar_product(rhou, rhou)/rho);

        phi.submit_value(rhoE, q);
        phi.submit_gradient(dt*(Ma*Ma*0.5*scalar_product(rhou/rho, rhou/rho)*rhou +
                                (rhoE - 0.5*Ma*Ma*scalar_product(rhou, rhou)/rho + pres)*rhou/rho), q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // Assemble rhs face term for the energy update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  assemble_rhs_face_term_energy_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_p(data, true, 1),
                                                                     phi_m(data, false, 1),
                                                                     phi_rhoE_p(data, true, 1),
                                                                     phi_rhoE_m(data, false, 1);
    FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_p(data, true, 2),
                                                                     phi_rho_m(data, false, 2);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_rhou_p(data, true, 0),
                                                                     phi_rhou_m(data, false, 0);

    /*--- Loop over all faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_rho_p.reinit(face);
      phi_rho_p.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rho_m.reinit(face);
      phi_rho_m.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rhou_p.reinit(face);
      phi_rhou_p.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhou_m.reinit(face);
      phi_rhou_m.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhoE_p.reinit(face);
      phi_rhoE_p.gather_evaluate(src[2], EvaluationFlags::values);
      phi_rhoE_m.reinit(face);
      phi_rhoE_m.gather_evaluate(src[2], EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus       = phi_p.get_normal_vector(q);

        const auto& rho_p        = phi_rho_p.get_value(q);
        const auto& rho_m        = phi_rho_m.get_value(q);
        const auto& rhou_p       = phi_rhou_p.get_value(q);
        const auto& rhou_m       = phi_rhou_m.get_value(q);
        const auto& rhoE_p       = phi_rhoE_p.get_value(q);
        const auto& rhoE_m       = phi_rhoE_m.get_value(q);

        /*--- Compute the pressure with the EOS ---*/
        const auto& pres_p       = (EquationData::Cp_Cv - 1.0)*
                                   (rhoE_p - 0.5*Ma*Ma*scalar_product(rhou_p, rhou_p)/rho_p);
        const auto& pres_m       = (EquationData::Cp_Cv - 1.0)*
                                   (rhoE_m - 0.5*Ma*Ma*scalar_product(rhou_m, rhou_m)/rho_m);

        /*--- Centered flux ---*/
        const auto& avg_enthalpy = 0.5*((rhoE_p - 0.5*Ma*Ma/rho_p*scalar_product(rhou_p, rhou_p) + pres_p)*rhou_p/rho_p +
                                        (rhoE_m - 0.5*Ma*Ma/rho_m*scalar_product(rhou_m, rhou_m) + pres_m)*rhou_m/rho_m);
        const auto& avg_kinetic  = 0.5*(0.5*rhou_p*scalar_product(rhou_p/rho_p, rhou_p/rho_p) +
                                        0.5*rhou_m*scalar_product(rhou_m/rho_m, rhou_m/rho_m));

        /*--- Rusanov flux ---*/
        const auto& lambda       = std::max(std::sqrt(scalar_product(rhou_p/rho_p, rhou_p/rho_p)) +
                                            1.0/Ma*std::sqrt(std::abs(EquationData::Cp_Cv*pres_p/rho_p)),
                                            std::sqrt(scalar_product(rhou_m/rho_m, rhou_m/rho_m)) +
                                            1.0/Ma*std::sqrt(std::abs(EquationData::Cp_Cv*pres_m/rho_m)));
        const auto& jump_rhoE    = rhoE_p - rhoE_m;

        phi_p.submit_value(-dt*(scalar_product(avg_enthalpy, n_plus) +
                                Ma*Ma*scalar_product(avg_kinetic, n_plus) +
                                0.5*lambda*jump_rhoE), q);
        phi_m.submit_value(dt*(scalar_product(avg_enthalpy, n_plus) +
                               Ma*Ma*scalar_product(avg_kinetic, n_plus) +
                               0.5*lambda*jump_rhoE), q);
      }

      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  vmult_rhs_energy_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_energy_update,
                     &EULEROperator::assemble_rhs_face_term_energy_update,
                     &EULEROperator::assemble_rhs_boundary_term_energy_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  assemble_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
                                   Vec&                                         dst,
                                   const Vec&                                   src,
                                   const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number> phi(data, 1);

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
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(Euler_stage, 4);
    Assert(Euler_stage > 0, ExcInternalError());

    if(Euler_stage == 1) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_rho_update,
                            this, dst, src, false);
    }
    else if(Euler_stage == 2) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_momentum_update,
                            this, dst, src, false);
    }
    else if(Euler_stage == 3) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_energy_update,
                            this, dst, src, false);
    }
    else {
      Assert(false, ExcInternalError());
    }
  }

} // End of namespace
