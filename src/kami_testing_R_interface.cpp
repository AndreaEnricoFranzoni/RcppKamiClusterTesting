#include <RcppEigen.h>


#include "utility/include_fdagwr.hpp"
#include "utility/traits_fdagwr.hpp"
#include "utility/concepts_fdagwr.hpp"
#include "utility/utility_fdagwr.hpp"
#include "utility/data_reader.hpp"
#include "utility/parameters_wrapper_fdagwr.hpp"

#include "basis/basis_include.hpp"
#include "basis/basis_bspline_systems.hpp"
#include "basis/basis_factory_proxy.hpp"

#include "functional_data/functional_data.hpp"
#include "functional_data/functional_data_covariates.hpp"

#include "weight_matrix/functional_weight_matrix_stat.hpp"
#include "weight_matrix/functional_weight_matrix_no_stat.hpp"
#include "weight_matrix/distance_matrix.hpp"
#include "weight_matrix/distance_matrix_pred.hpp"

#include "penalization_matrix/penalization_matrix.hpp"

#include "functional_matrix/functional_matrix.hpp"
#include "functional_matrix/functional_matrix_sparse.hpp"
#include "functional_matrix/functional_matrix_diagonal.hpp"
#include "functional_matrix/functional_matrix_operators.hpp"
#include "functional_matrix/functional_matrix_product.hpp"
#include "functional_matrix/functional_matrix_into_wrapper.hpp"


#include "fwr/fwr_factory.hpp"
#include "fwr_predictor/fwr_predictor_factory.hpp"


using namespace Rcpp;

//
// [[Rcpp::depends(RcppEigen)]]


//
// [[Rcpp::export]]
void installation_kami_testing(){   Rcout << "Testing for Kami Cluster"<< std::endl;}


//
// [[Rcpp::export]]
void kami_testing()
{
  //creation of a collection of systems of bsplines: in this case, there are two basis systems.
  // Both of them are constructed over 13 knots equally spaced. In the first system, there are 15 bsplines of grade 3,
  // while in the second system there are 14 bsplines of grade 2
  
  //basis domain geometry (from fdaPDE)
  using _DOMAIN_ = FDAGWR_TRAITS::basis_geometry; 
  //domain
  double a = -2.5;
  double b = 1;
  //number of basis systems
  std::size_t q = 2;
  //knots
  FDAGWR_TRAITS::Dense_Vector knots_basis = FDAGWR_TRAITS::Dense_Vector::LinSpaced(13, a, b);
  //basis degree
  std::vector<std::size_t> degree_basis{3,2};
  //basis number
  std::vector<std::size_t> number_basis{15,14};

  //creation of the collection of the basis system: the problem is in this constructor
  basis_systems< _DOMAIN_, bsplines_basis > bs(knots_basis, 
                                               degree_basis, 
                                               number_basis, 
                                               q);
  //evaluating the two basis systems in 0.3
  double loc = 0.3;   //abscissa of evaluation
  auto evaluation_basis_system_1 = bs.eval_base(loc,0);
  Rcout << "Evaluation of the first basis system in " << loc << std::endl;
  Rcout << evaluation_basis_system_1 << std::endl;
  auto evaluation_basis_system_2 = bs.eval_base(loc,1);
  Rcout << "Evaluation of the second basis system in " << loc << std::endl;
  Rcout << evaluation_basis_system_2 << std::endl;
  
}