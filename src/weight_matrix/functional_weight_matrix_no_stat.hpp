// Copyright (c) 2025 Andrea Enrico Franzoni (andreaenrico.franzoni@gmail.com)
//
// This file is part of fdagwr
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of fdagwr and associated documentation files (the fdagwr software), to deal
// fdagwr without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of fdagwr, and to permit persons to whom fdagwr is
// furnished to do so, subject to the following conditions:
//
// fdagwr IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH fdagwr OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#ifndef FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_NON_STATIONARY_HPP
#define FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_NON_STATIONARY_HPP

#include "functional_weight_matrix.hpp"
#include "distance_matrix.hpp"
#include "distance_matrix_pred.hpp"

#include <iostream>


/*!
* @file functional_weight_matrix_no_stat.hpp
* @brief Construct the non-stationary weight matrix for performing the geographically weighted regression. For a given statistical unit, weights consist of the functional response reconstruction weights multiplied by the kernel function evaluated innthe distance within the given statistical units and all the others
* @author Andrea Enrico Franzoni
*/



/*!
* @brief Tag dispatching for how to evaluate the non stationary components (kernel function for the distances within statistical units)
* @tparam kernel_func: template parameter for the wanted kernel function to smooth the distances
*/
template <KERNEL_FUNC kernel_func>
using KERNEL_FUNC_T = std::integral_constant<KERNEL_FUNC, kernel_func>;


/*!
* @class functional_weight_matrix_non_stationary
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @tparam domain_type geometry of the basis domain
* @tparam basis_type type of basis used to construct the functional weights
* @tparam stationarity_t covariate type
* @tparam kernel_func the kernel function used to smooth the distances
* @tparam dist_meas the distance measure used to computed the distance
* @brief Class for constructing the functional weight matrix for non-stationary covariates
* @details For each statistical units, the functional matrix is a diagonal matrix of functions, n x n (n number of statistical units), where each diagonal element consists in the functional response reconstruction weight of the corresponding statistical unit multiplied by the kernel function evaluated in the distance within the statistical unit for which the matrix is computed and the one corresponding to the reconstruction weights
*/
template< typename INPUT = double, typename OUTPUT = double, class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::NON_STATIONARY, KERNEL_FUNC kernel_func = KERNEL_FUNC::GAUSSIAN, DISTANCE_MEASURE dist_meas = DISTANCE_MEASURE::EUCLIDEAN >  
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>) && fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
class functional_weight_matrix_non_stationary : public functional_weight_matrix_base< functional_weight_matrix_non_stationary<INPUT,OUTPUT,domain_type,basis_type,stationarity_t,kernel_func,dist_meas>, INPUT, OUTPUT, domain_type, basis_type >
{
    /*!std::function object stored*/
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    /*!std::function input type*/
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

private:

    /*!Vector, of dimension n, such that each element j is a vector of n functions, element i-th representing the weight for reconstructing element i-th multiplied by the kernel function evaluated in the distance within statistical unit j-th and i-th*/
    WeightMatrixType<INPUT,OUTPUT,stationarity_t> m_weights;
    /*!Kernel bandwith*/
    double m_kernel_bandwith;
    /*!Distance matrix*/
    distance_matrix<dist_meas> m_distance_matrix;
    /*!Distance matrix for pred*/
    distance_matrix_pred<dist_meas> m_distance_matrix_pred;

    /*!
    * @brief Evaluation of the gaussian kernel function for the non stationary weights
    * @param distance distance between two locations
    * @param bandwith kernel bandwith
    * @return the evaluation of the kernel function
    */
    double kernel_eval(double distance, double bandwith, KERNEL_FUNC_T<KERNEL_FUNC::GAUSSIAN>) const;

public:
    /*!
    * @brief Constructor for the non stationary weight matrix
    * @param y_recostruction_weights_fd functional response reconstruction weights
    * @param distance_matrix distance matrix of the training set
    * @param kernel_bwt kernel bandwith
    * @param number_threads number of threads for OMP
    */
    template< typename DIST_MATRIX_OBJ >
    functional_weight_matrix_non_stationary(const functional_data<domain_type,basis_type> &y_recostruction_weights_fd,
                                            DIST_MATRIX_OBJ&& distance_matrix,
                                            double kernel_bwt,
                                            int number_threads)
                                : 
                                  functional_weight_matrix_base<functional_weight_matrix_non_stationary,INPUT,OUTPUT,domain_type,basis_type>(y_recostruction_weights_fd,number_threads),
                                  m_kernel_bandwith(kernel_bwt),
                                  m_distance_matrix{std::forward<DIST_MATRIX_OBJ>(distance_matrix)}                                                                                          
                                {                                                 
                                  static_assert(stationarity_t == FDAGWR_COVARIATES_TYPES::NON_STATIONARY   ||
                                                stationarity_t == FDAGWR_COVARIATES_TYPES::EVENT            ||
                                                stationarity_t == FDAGWR_COVARIATES_TYPES::STATION,
                                                "Functional weight matrix for non stationary covariates needs FDAGWR_COVARIATES_TYPES::NON_STATIONARY or FDAGWR_COVARIATES_TYPES::EVENT or FDAGWR_COVARIATES_TYPES::STATION as template parameter");
                                }

    /*!
    * @brief Constructor for the non stationary weight matrix computed within training and prediction set
    * @param y_recostruction_weights_fd functional response reconstruction weights
    * @param distance_matrix_pred distance matrix within the training and the prediction set
    * @param kernel_bwt kernel bandwith
    * @param number_threads number of threads for OMP
    * @param pred a bool pleonastic parameter for differentiating the signature of the two constructors
    */
    template< typename DIST_MATRIX_OBJ >
    functional_weight_matrix_non_stationary(const functional_data<domain_type,basis_type> &y_recostruction_weights_fd,
                                            DIST_MATRIX_OBJ&& distance_matrix_pred,
                                            double kernel_bwt,
                                            int number_threads,
                                            bool pred)
                                : 
                                  functional_weight_matrix_base<functional_weight_matrix_non_stationary,INPUT,OUTPUT,domain_type,basis_type>(y_recostruction_weights_fd,number_threads),
                                  m_kernel_bandwith(kernel_bwt),
                                  m_distance_matrix_pred{std::forward<DIST_MATRIX_OBJ>(distance_matrix_pred)}                                                                                          
                                {                                                 
                                  static_assert(stationarity_t == FDAGWR_COVARIATES_TYPES::NON_STATIONARY   ||
                                                stationarity_t == FDAGWR_COVARIATES_TYPES::EVENT            ||
                                                stationarity_t == FDAGWR_COVARIATES_TYPES::STATION,
                                                "Functional weight matrix for non stationary covariates needs FDAGWR_COVARIATES_TYPES::NON_STATIONARY or FDAGWR_COVARIATES_TYPES::EVENT or FDAGWR_COVARIATES_TYPES::STATION as template parameter");
                                }

    /*!
    * @brief Getter for the functional non-stationary weight matrix
    * @return the private m_weights
    */
    const WeightMatrixType<INPUT,OUTPUT,stationarity_t>& weights() const {return m_weights;}

    /*!
    * @brief Evaluation of kernel function for the non-stationary weights. Tag-dispacther.
    * @param distance distance between two locations
    * @param bandwith kernel bandwith
    * @return the evaluation of the kernel function
    */
    double kernel_eval(double distance, double bandwith) const { return kernel_eval(distance,bandwith,KERNEL_FUNC_T<kernel_func>{});};

    /*!
    * @brief Function to compute non stationary weights
    * @details Semantic strongly influeneced by Eigen
    */
    inline
    void
    computing_weights()
    {
      //to shared the values with OMP
      auto n_stat_units = this->n();
      //preparing the container for the functional non-stationary weight matrix
      m_weights.resize(n_stat_units);

#ifdef _OPENMP
#pragma omp parallel for shared(m_distance_matrix,n_stat_units) num_threads(this->number_threads())
#endif
      for(std::size_t i = 0; i < n_stat_units; ++i)
      {
        //non stationary weights with respect to unit i-th 

        //Eigen vector with the distances with respect to unit i-th
        auto weights_non_stat_unit_i = m_distance_matrix[i]; 
        
        //applying the kernel function to correctly smoothing the distances
        std::transform(weights_non_stat_unit_i.data(),
                       weights_non_stat_unit_i.data() + weights_non_stat_unit_i.size(),
                       weights_non_stat_unit_i.data(),
                       [this](double dist){return this->kernel_eval(dist,this->m_kernel_bandwith);});

        //preparing the container for the functional non-stationary matrix of unit i-th 
        std::vector< F_OBJ > weights_unit_i;
        weights_unit_i.reserve(n_stat_units);

        //computing the interaction within kernel application to distances and response reconstruction, unit i-th and all the other ones
        for (std::size_t j = 0; j < n_stat_units; ++j)
        {          
          double alpha_i_j = weights_non_stat_unit_i[j];
          F_OBJ w_i_j = [j,alpha_i_j,this](F_OBJ_INPUT loc){ return alpha_i_j * this->y_recostruction_weights_fd().eval(loc,j);};
          weights_unit_i.push_back(w_i_j);
        }
        
        //storing the functional non-stationary matrix for unit i-th (corresponding to index unit_index)
        m_weights[i] = weights_unit_i;
      }
    }

    /*!
    * @brief Computing the weights within trainin and prediction set
    */
    inline 
    void
    compute_weights_pred()
    {
      std::size_t n_train = m_distance_matrix_pred.n_train();
      std::size_t n_pred  = m_distance_matrix_pred.n_pred();

      m_weights.resize(n_pred);

#ifdef _OPENMP
#pragma omp parallel for shared(m_distance_matrix_pred,n_train,n_pred) num_threads(this->number_threads())
#endif
      for(std::size_t i_pred = 0; i_pred < n_pred; ++i_pred)
      {
        std::vector<double> weights_non_stat_unit_i_pred = m_distance_matrix_pred.distances()[i_pred];

        //applying the kernel function to correctly smoothing the distances
        std::transform(weights_non_stat_unit_i_pred.begin(),
                       weights_non_stat_unit_i_pred.end(),
                       weights_non_stat_unit_i_pred.begin(),
                       [this](double dist){return this->kernel_eval(dist,this->m_kernel_bandwith);});

        //preparing the container for the functional non-stationary matrix of pred unit i_pred-th 
        std::vector< F_OBJ > weights_unit_i_pred;
        weights_unit_i_pred.reserve(n_train);

        //computing the interaction within kernel application to distances and response reconstruction, unit i-th and all the other ones
        for (std::size_t j_train = 0; j_train < n_train; ++j_train)
        {          
          double alpha_i_j = weights_non_stat_unit_i_pred[j_train];
          F_OBJ w_i_j = [j_train,alpha_i_j,this](F_OBJ_INPUT loc){ return alpha_i_j * this->y_recostruction_weights_fd().eval(loc,j_train);};
          weights_unit_i_pred.push_back(w_i_j);
        }
        
        //storing the functional non-stationary matrix for unit i-th (corresponding to index unit_index)
        m_weights[i_pred] = weights_unit_i_pred;
      }  
    }
};

#include "functional_weight_matrix_kernel_functions_eval.hpp"

#endif  /*FDAGWR_FUNCTIONAL_WEIGHT_MATRIX_NON_STATIONARY_HPP*/