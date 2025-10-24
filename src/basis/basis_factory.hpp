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


#ifndef FDAGWR_BASIS_FACTORY_HPP
#define FDAGWR_BASIS_FACTORY_HPP


#include "../utility/include_fdagwr.hpp"
#include "../utility/traits_fdagwr.hpp"
#include "../utility/concepts_fdagwr.hpp"
#include "../utility/factory.hpp"
#include "../utility/factory_proxy.hpp"
#include "basis_include.hpp"


/*!
* @file basis_factory.hpp
* @brief Contains the factory definition for the basis
* @author Andrea Enrico Franzoni
*/



/*!
* @namespace basis_factory
* @brief Contains elements to define a factory for the basis
*/
namespace basis_factory{

    /*!
    * @brief Identifier for the factory, std::strinf
    */
    using basisIdentifier = std::string;

    /*!
    * @brief Builder for the concrete basis objects. Input parameters of the callable objects are the ones used by the constructors, that have all the same input parameters
    * @note If the constructor of the basis concrete objects changes, this type definition and factory_proxy.hpp have to be changed accordingly
    */
    using basisBuilder = std::function<std::unique_ptr<basis_base_class<FDAGWR_TRAITS::basis_geometry>>(const FDAGWR_TRAITS::Dense_Vector &, std::size_t, std::size_t)>;

    /*!
    * @typedef basisFactory
    * @brief The factory for the basis
    */
    typedef generic_factory::Factory< basis_base_class<FDAGWR_TRAITS::basis_geometry>, basisIdentifier, basisBuilder> basisFactory;  // Use standard Builder // Use standard Builder

    /*!
    * @brief The proxy for the factory for the basis
    * @tparam ConcreteProduct the concrete object constructed
    */
    template<typename ConcreteProduct>
    using basisProxy = generic_factory::Proxy<basisFactory,ConcreteProduct>;

}   //end namespace basis_factory


#endif  /*FDAGWR_BASIS_FACTORY_HPP*/