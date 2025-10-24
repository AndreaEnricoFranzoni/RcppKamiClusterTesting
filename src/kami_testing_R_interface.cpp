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

