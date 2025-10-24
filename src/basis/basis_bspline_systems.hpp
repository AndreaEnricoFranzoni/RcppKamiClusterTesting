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


#ifndef FDAGWR_BASIS_SYSTEM_HPP
#define FDAGWR_BASIS_SYSTEM_HPP


#include "basis_include.hpp"
#include <iostream>


/*!
* @file basis_bspline_systems.hpp
* @brief Contains the class for a collection of system of basis of the same type
* @author Andrea Enrico Franzoni
*/



/*!
* @class basis_systems
* @tparam domain_type the domain over which the basis is constructed
* @tparam basis_type template template param that indicates the type of basis
* @brief class for a collection of systems of basis of a given type
*/
template< class domain_type = FDAGWR_TRAITS::basis_geometry, template <typename> class basis_type = bsplines_basis > 
    requires fdagwr_concepts::as_interval<domain_type> && fdagwr_concepts::as_basis<basis_type<domain_type>>
class basis_systems
{

private:
    /*!Geometry over which the basis systems are constructed*/
    domain_type m_knots;

    /*!Order of basis for each system of basis*/
    std::vector<std::size_t> m_basis_degrees;

    /*!Number of basis for each system of basis*/
    std::vector<std::size_t> m_numbers_of_basis;

    /*!Number of systems of basis*/
    std::size_t m_q;

    /*!Vector containing the collection of basis systems*/
    std::vector<basis_type<domain_type>> m_systems_of_basis;

public:
    /*!
    * @brief Class constructor for a collection of basis systems
    * @param knots Eigen::VectorXd containing the knots over which each one of the basis system is constructed
    * @param basis_degrees vector of integers containing the degrees of each one of the basis systems
    * @param numbers_of_basis vector of integers containing the number of basis of each one of the basis systems
    * @param q the number of basis systems
    * @todo to use gcc, would be necessary to use .push_back() instead of .emplace_back(). However, for bsplines basis, fdaPDE 
    *       code crashes when calling the function on R for segfault problems, since it is storing the bsplines with pointers and it is creating temporaries
    */
    basis_systems(const FDAGWR_TRAITS::Dense_Vector & knots,
                  const std::vector<std::size_t> & basis_degrees,
                  const std::vector<std::size_t> & numbers_of_basis,
                  std::size_t q)            
                  :    
                        m_knots(knots),
                        m_basis_degrees(basis_degrees),
                        m_numbers_of_basis(numbers_of_basis),
                        m_q(q)
                     {
                        //constructing systems of bsplines given knots and orders of the basis  
                        m_systems_of_basis.reserve(m_q);
                        for(std::size_t i = 0; i < m_q; ++i){  
                            m_systems_of_basis.emplace_back(knots, m_basis_degrees[i], m_numbers_of_basis[i]);
                        }
                     }

    /*!
    * @brief Getter for the geometry over which the basis systems are constructed
    */
    const domain_type& knots() const {return m_knots;}

    /*!
    * @brief Getter for the systems of basis (returning a reference since fdaPDE stores the basis as a pointer to them)
    * @return the private m_systems_of_basis
    */
    const std::vector<basis_type<domain_type>>& systems_of_basis() const {return m_systems_of_basis;}

    /*!
    * @brief Getter for the order of basis for each system of basis
    * @return the private m_basis_orders
    */
    const std::vector<std::size_t>& basis_degrees() const {return m_basis_degrees;}

    /*!
    * @brief Getter for the number of basis for each system of basis
    * @return the private m_number_of_basis
    */
    const std::vector<std::size_t>& numbers_of_basis() const {return m_numbers_of_basis;}

    /*!
    * @brief Getter for the number of basis systems
    * @return the private m_q
    */
    std::size_t q() const {return m_q;}

    /*!
    * @brief evaluating the system of basis i-th in location location
    * @param location the abscissa over which evaluating the basis of a given basis system
    * @param basis_i which basis system is evaluated
    * @return an Eigen::MatrixXd of dimensions 1 x m_number_of_basis (of the system basis_i-th)
    */
    inline 
    FDAGWR_TRAITS::Dense_Matrix 
    eval_base(const double &location, std::size_t basis_i) 
    const
    {
        return m_systems_of_basis[basis_i].eval(location);
    }
};

#endif  /*FDAGWR_BASIS_SYSTEM_HPP*/