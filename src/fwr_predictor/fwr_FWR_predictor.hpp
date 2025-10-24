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
// OUT OF OR IN CONNECTION WITH PPCKO OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#ifndef FWR_FWR_PREDICT_HPP
#define FWR_FWR_PREDICT_HPP

#include "fwr_predictor.hpp"

#include <iostream>

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class fwr_FWR_predictor final : public fwr_predictor<INPUT,OUTPUT>
{
private:
    //
    //FITTED MODEL
    //
    //Basis expansion coefficients for the regression coefficients: fitted values
    /*!Coefficients of the basis expansion for stationary regressors coefficients: Lcx1*/
    FDAGWR_TRAITS::Dense_Matrix m_bc_fitted;
    /*!Coefficients of the basis expansion for stationary regressors coefficients: every element is Lc_jx1*/
    std::vector< FDAGWR_TRAITS::Dense_Matrix > m_Bc_fitted;

    //Basis used for the regression coefficients
    /*!Basis for stationary covariates regressors (sparse qc x Lc)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_omega;
    /*!Their transpost (sparse Lc x qC)*/
    functional_matrix_sparse<INPUT,OUTPUT> m_omega_t;
    /*!Number of stationary covariates*/
    std::size_t m_qc;
    /*!Number of basis, in total, used to perform the basis expansion of the regressors coefficients for the stationary regressors coefficients*/
    std::size_t m_Lc; 
    /*!Number of basis, for each stationary covariate, to perform the basis expansion of the regressors coefficients for the stationary regressors coefficients*/
    std::vector<std::size_t> m_Lc_j;

    //Betas
    //stationary: qc x 1
    functional_matrix<INPUT,OUTPUT> m_BetaC;                        //elemento funzionale
    /*!Discrete evaluation of all the beta_c: a vector of dimension qc, containing, for all the stationary covariates, the discrete ev of the respective beta*/
    std::vector< std::vector< OUTPUT >> m_BetaC_ev;

public:
    /*!
    * @brief Constructor
    */
    template<typename FUNC_SPARSE_MATRIX_OBJ,
             typename SCALAR_MATRIX_OBJ_VEC>
    fwr_FWR_predictor(SCALAR_MATRIX_OBJ_VEC &&Bc_fitted,
                       FUNC_SPARSE_MATRIX_OBJ &&omega,
                       std::size_t qc,
                       std::size_t Lc,
                       const std::vector<std::size_t> &Lc_j,
                       INPUT a, 
                       INPUT b, 
                       int n_intervals_integration, 
                       double target_error, 
                       int max_iterations, 
                       std::size_t n_train, 
                       int number_threads)
            :   
                fwr_predictor<INPUT,OUTPUT>(a,b,n_intervals_integration,target_error,max_iterations,n_train,number_threads,false),
                m_Bc_fitted{std::forward<SCALAR_MATRIX_OBJ_VEC>(Bc_fitted)},
                m_omega{std::forward<FUNC_SPARSE_MATRIX_OBJ>(omega)},
                m_qc(qc),
                m_Lc(Lc),
                m_Lc_j(Lc_j)
            {
                //input coherency
                assert(m_Bc_fitted.size() == m_qc);
                assert((m_omega.rows() == m_qc) && (m_omega.cols() == Lc));
                 //compute the transpost
                m_omega_t = m_omega.transpose();
                //dewrappare i b_train (li incolonna)
                m_bc_fitted = this->operator_comp().dewrap_operator(m_Bc_fitted,m_Lc_j);
            }

    /*!
    * @brief Function to reconstruct the functional partial residuals
    */
    inline 
    void
    computePartialResiduals()
    override
    {}

    inline
    void
    computeBNew(const std::map<std::string,std::vector< functional_matrix_diagonal<INPUT,OUTPUT> >> &W)
    override
    {
        assert(W.empty());
    }

    /*!
    * @brief Compute stationary betas
    */
    inline
    void 
    computeStationaryBetas()
    override
    {
        m_BetaC = fm_prod(m_omega,m_bc_fitted);
    }

    /*!
    * @brief Compute non-stationary betas
    */
    inline
    void 
    computeNonStationaryBetas()
    override
    {}

    /*!
    * @brief Compute prediction
    */
    inline
    functional_matrix<INPUT,OUTPUT>
    predict(const std::map<std::string,functional_matrix<INPUT,OUTPUT>> &X_new)
    const
    override
    {
        assert(X_new.size() == 1);
        assert(X_new.at(fwr_FWR_predictor<INPUT,OUTPUT>::id_C).cols() == m_qc);
        //new covsariates
        auto Xc_new = X_new.at(fwr_FWR_predictor<INPUT,OUTPUT>::id_C);

        //y_new = X_new*beta = Xc_new*beta_c
        functional_matrix<INPUT,OUTPUT> y_new_C = fm_prod(Xc_new,m_BetaC,this->number_threads());    //n_pred x 1

        return y_new_C;
    }

    /*!
    * @brief Virtual method to obtain a discrete version of the betas
    */
    inline 
    void 
    evalBetas(const std::vector<INPUT> &abscissa)
    override
    {
        m_BetaC_ev = this->operator_comp().eval_func_betas(m_BetaC,m_qc,abscissa);
    }

    /*!
    * @brief Getter for the coefficient of the basis expansion of the stationary regressors coefficients
    */
    inline 
    BTuple 
    bCoefficients()
    const 
    override
    {
        return std::tuple{m_Bc_fitted};
    }

    /*!
    * @brief Getter for the. etas evaluated along the abscissas
    */
    inline 
    BetasTuple 
    betas() 
    const
    override
    {
        return std::tuple{m_BetaC_ev};
    }

};

#endif  /*FWR_FWR_PREDICT_HPP*/