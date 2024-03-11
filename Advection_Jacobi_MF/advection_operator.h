/* Author: Giuseppe Orlando, 2024. */

// We start by including all the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/meshworker/mesh_loop.h>

#include "runtime_parameters.h"
#include "equation_data.h"

// We include the code in a suitable namespace:
//
namespace Advection {
  using namespace dealii;

  // @sect{ <code>AdvectionImplicitOperator::AdvectionImplicitOperator</code> }

  // The following class sets effectively the weak formulation of the problem.
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  class AdvectionImplicitOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    AdvectionImplicitOperator(); /*--- Default constructor ---*/

    AdvectionImplicitOperator(RunTimeParameters::Data_Storage& data); /*--- Constructor with runtime parameters ---*/

    void set_dt(const double time_step); /*--- Setter of time step ---*/

    void set_advecting_field(const Vec& vel); /*--- Setter of the advection field ---*/

    /*--- Functions to assemble rhs ---*/
    void vmult_rhs(Vec& dst, const Vec& src) const;

    /*--- Compute diagonal ---*/
    virtual void compute_diagonal() override;

  protected:
    double dt; /*--- Time step parameter ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Assemble the weak formulation ---*/

  private:
    Vec velocity; /*--- Auxiliary vector to store the FE representatino fo the advecting field ---*/

    /*--- The following functions basically assemble the linear and bilinear forms. Their syntax is due to
          the base class MatrixFreeOperators::Base. We start with the rhs ---*/
    void assemble_rhs_cell_term(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const Vec&                                   src,
                                const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Functions for the bilinear form ---*/
    void assemble_cell_term(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const Vec&                                   src,
                            const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term(const MatrixFree<dim, Number>&               data,
                            Vec&                                         dst,
                            const Vec&                                   src,
                            const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const Vec&                                   src,
                                const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Functions for computing the diagonal of the level set ---*/
    void assemble_diagonal_cell_term(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const unsigned int&                          src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const unsigned int&                          src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const unsigned int&                          src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {}
  };


  // We start with the default constructor. It is important for MultiGrid, so it is fundamental
  // to properly set the parameters of the time scheme.
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  AdvectionImplicitOperator(): MatrixFreeOperators::Base<dim, Vec>(), dt() {}


  // We focus now on the constructor with runtime parameters storage
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  AdvectionImplicitOperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(), dt(data.dt) {}


  // Setter of time-step (called by Multigrid and in case a smaller time-step towards the end is needed)
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of time-step (called by Multigrid and in case a smaller time-step towards the end is needed)
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  set_advecting_field(const Vec& vel) {
    vel.update_ghost_values();

    velocity = vel;
  }


  // We are in a DG-MatrixFree framework, so it is convenient to compute separately cell contribution,
  // internal faces contributions and boundary faces contributions. We start by
  // assembling the rhs cell term.
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  assemble_rhs_cell_term(const MatrixFree<dim, Number>&               data,
                         Vec&                                         dst,
                         const Vec&                                   src,
                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> phi(data, 0),
                                                           phi_old(data, 0);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_old.reinit(cell);
      phi_old.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi_old.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the steps to compute the rhs
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  vmult_rhs(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->cell_loop(&AdvectionImplicitOperator::assemble_rhs_cell_term,
                          this, dst, src, true);
  }


  // We now assemble the cell term for the advection operator
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  assemble_cell_term(const MatrixFree<dim, Number>&               data,
                     Vec&                                         dst,
                     const Vec&                                   src,
                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1,   Number> phi(data, 0);
    FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi_u(data, 1);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      phi_u.reinit(cell);
      phi_u.gather_evaluate(velocity, EvaluationFlags::values);

      /*--- Now we loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& vel = phi_u.get_value(q);

        phi.submit_value(phi.get_value(q) + dt*scalar_product(vel, phi.get_gradient(q)), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // The following function assembles face term for the advection operator
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  assemble_face_term(const MatrixFree<dim, Number>&               data,
                     Vec&                                         dst,
                     const Vec&                                   src,
                     const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>   phi_p(data, true, 0),
                                                                 phi_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi_u_p(data, true, 1),
                                                                 phi_u_m(data, false, 1);

    /*--- Loop over all faces in the range ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, EvaluationFlags::values);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, EvaluationFlags::values);

      phi_u_p.reinit(face);
      phi_u_p.gather_evaluate(velocity, EvaluationFlags::values);
      phi_u_m.reinit(face);
      phi_u_m.gather_evaluate(velocity, EvaluationFlags::values);

      /*--- Loop over quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus         = phi_p.get_normal_vector(q);

        const auto& vel_p          = phi_u_p.get_value(q);
        const auto& vel_m          = phi_u_m.get_value(q);

        const auto& avg_flux_part1 = 0.5*(vel_p*phi_p.get_value(q) + vel_m*phi_m.get_value(q));
        const auto& avg_flux_part2 = 0.5*(vel_p + vel_m);
        const auto& lambda         = std::max(std::abs(scalar_product(vel_p, n_plus)),
                                              std::abs(scalar_product(vel_m, n_plus)));
        const auto& jump           = phi_p.get_value(q) - phi_m.get_value(q);

        phi_p.submit_value(dt*(scalar_product(avg_flux_part1 - avg_flux_part2*phi_p.get_value(q), n_plus) + 0.5*lambda*jump), q);
        phi_m.submit_value(-dt*(scalar_product(avg_flux_part1 - avg_flux_part2*phi_m.get_value(q), n_plus) + 0.5*lambda*jump), q);
      }

      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all previous steps. This is the overriden function that effectively performs the
  // matrix-vector multiplication.
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    this->data->loop(&AdvectionImplicitOperator::assemble_cell_term,
                     &AdvectionImplicitOperator::assemble_face_term,
                     &AdvectionImplicitOperator::assemble_boundary_term,
                     this, dst, src, false,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble diagonal cell term for the level set
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  assemble_diagonal_cell_term(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const unsigned int&                          ,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1,   Number> phi(data, 0);
    FEEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi_u(data, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cless ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      phi_u.reinit(cell);
      phi_u.gather_evaluate(velocity, EvaluationFlags::values);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& vel = phi_u.get_value(q);

          phi.submit_value(phi.get_value(q) + dt*scalar_product(vel, phi.get_gradient(q)), q);
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


  // The following function assembles diagonal face term for the level set
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  assemble_diagonal_face_term(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const unsigned int&                          ,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>   phi_p(data, true),
                                                                 phi_m(data, false);
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, dim, Number> phi_u_p(data, true, 1),
                                                                 phi_u_m(data, false, 1);

    AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
    AlignedVector<VectorizedArray<Number>> diagonal_p(phi_p.dofs_per_component),
                                           diagonal_m(phi_m.dofs_per_component);

    /*--- Loop over all faces in the range ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_m.reinit(face);

      phi_u_p.reinit(face);
      phi_u_p.gather_evaluate(velocity, EvaluationFlags::values);
      phi_u_m.reinit(face);
      phi_u_m.gather_evaluate(velocity, EvaluationFlags::values);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
          phi_p.submit_dof_value(VectorizedArray<Number>(), j);
          phi_m.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi_p.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_m.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_p.evaluate(EvaluationFlags::values);
        phi_m.evaluate(EvaluationFlags::values);

        /*--- Loop over quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus         = phi_p.get_normal_vector(q);

          const auto& vel_p          = phi_u_p.get_value(q);
          const auto& vel_m          = phi_u_m.get_value(q);

          const auto& avg_flux_part1 = 0.5*(vel_p*phi_p.get_value(q) + vel_m*phi_m.get_value(q));
          const auto& avg_flux_part2 = 0.5*(vel_p + vel_m);
          const auto& lambda         = std::max(std::abs(scalar_product(vel_p, n_plus)),
                                                std::abs(scalar_product(vel_m, n_plus)));
          const auto& jump           = phi_p.get_value(q) - phi_m.get_value(q);

          phi_p.submit_value(dt*(scalar_product(avg_flux_part1 - avg_flux_part2*phi_p.get_value(q), n_plus) + 0.5*lambda*jump), q);
          phi_m.submit_value(-dt*(scalar_product(avg_flux_part1 - avg_flux_part2*phi_m.get_value(q), n_plus) + 0.5*lambda*jump), q);
        }

        phi_p.integrate(EvaluationFlags::values);
        diagonal_p[i] = phi_p.get_dof_value(i);
        phi_m.integrate(EvaluationFlags::values);
        diagonal_m[i] = phi_m.get_dof_value(i);
      }

      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        phi_p.submit_dof_value(diagonal_p[i], i);
        phi_m.submit_dof_value(diagonal_m[i], i);
      }
      phi_p.distribute_local_to_global(dst);
      phi_m.distribute_local_to_global(dst);
    }
  }


  // Compute the diagonal
  //
  template<int dim, int fe_degree, int n_q_points_1d, typename Vec>
  void AdvectionImplicitOperator<dim, fe_degree, n_q_points_1d, Vec>::
  compute_diagonal() {
    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    const unsigned int dummy = 0;

    this->data->initialize_dof_vector(inverse_diagonal, 0);

    this->data->loop(&AdvectionImplicitOperator::assemble_diagonal_cell_term,
                     &AdvectionImplicitOperator::assemble_diagonal_face_term,
                     &AdvectionImplicitOperator::assemble_diagonal_boundary_term,
                     this, inverse_diagonal, dummy, false,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);

    /*--- For the preconditioner, we actually need the inverse of the diagonal ---*/
    for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != 0.0,
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
    }
  }

} // End of namespace Advection
