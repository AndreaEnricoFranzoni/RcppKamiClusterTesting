// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_GEOFRAME_H__
#define __FDAPDE_GEOFRAME_H__

#include "header_check.h"

namespace fdapde {
namespace internals {
  
inline void throw_geoframe_error(const std::string& msg) { throw std::runtime_error("GeoFrame: " + msg); }
#define geoframe_assert(condition, msg)                                                                                \
    if (!(condition)) { internals::throw_geoframe_error(msg); }

}   // namespace internals

template <typename... Triangulation_> struct GeoFrame {
    fdapde_static_assert(sizeof...(Triangulation_) > 0, AT_LEAST_ONE_TRIANGULATION_REQUIRED);
   public:
    using This = GeoFrame<Triangulation_...>;
    using Triangulation = std::tuple<std::decay_t<Triangulation_>...>;
    static constexpr int Order = sizeof...(Triangulation_);

    struct layer_t {
        using Triangulation = std::tuple<std::decay_t<Triangulation_>...>;
        using storage_t = internals::scalar_data_layer;
        static constexpr int Order = sizeof...(Triangulation_);

        layer_t() noexcept : geo_data_(), data_(), category_(), name_() { }
        template <typename CategoryType, typename LayerType>
        layer_t(const std::string& name, const CategoryType& category, const LayerType& geo_data) :
            geo_data_(std::make_shared<LayerType>(geo_data)),
            data_(std::addressof(reinterpret_cast<LayerType*>(geo_data_.get())->data())),
            name_(name) {
            geoframe_assert(category.size() == Order, "bad layer construction, no matching order.");
            std::copy(category.begin(), category.end(), category_.begin());
	    // store pointers to spatial indexes
            internals::for_each_index_in_pack<Order>([&, this]<int Ns>() {
                geo_index_[Ns] = reinterpret_cast<void*>(
                  std::addressof(reinterpret_cast<LayerType*>(geo_data_.get())->template geometry<Ns>()));
            });
        }
        // observers
        const std::string& name() const { return name_; }
        const storage_t& data() const { return *data_; }
        storage_t& data() { return *data_; }
        void* geo_data() { return geo_data_.get(); }
        const void* geo_data() const { return geo_data_.get(); }
        const std::array<ltype, Order>& category() const { return category_; }
        bool contains(const std::string& colname) const { return data_->contains(colname); }
        int rows() const { return data_->rows(); }
        int cols() const { return data_->cols(); }
        int size() const { return data_->size(); }
        std::vector<std::string> colnames() const { return data_->colnames(); }
        void* geo_index(int n) const { return geo_index_.at(n); }
        // accessors
        template <typename T> decltype(auto) col(size_t col) { return data_->template col<T>(col); }
        template <typename T> decltype(auto) col(size_t col) const { return data_->template col<T>(col); }
        template <typename T> decltype(auto) col(const std::string& colname) { return data_->template col<T>(colname); }
        template <typename T> decltype(auto) col(const std::string& colname) const {
            return data_->template col<T>(colname);
        }
        // modifiers
        template <typename DataT>
            requires(internals::is_vector_like_v<DataT>)
        void add_column(const std::string& colname, const DataT& data) {
            data_->append_vec(colname, data);
        }
        template <typename T>
        void add_block(const std::string& colname, const Eigen::Matrix<T, Dynamic, Dynamic>& data) {
            data_->append_blk(colname, data);
        }
        // output stream
        friend std::ostream& operator<<(std::ostream& os, const layer_t& layer) {
            return operator<<(os, *layer.data_);
        }
       private:
        std::shared_ptr<void> geo_data_;       // type erased geometric layer
        std::array<void*, Order> geo_index_;   // pointers to (type-erased) geometric indexes
        storage_t* data_;
        std::array<ltype, Order> category_;
        std::string name_;
    };
   private:
    template <typename T> constexpr auto ltype_from_layer_tag() const {
        using T_ = std::decay_t<T>;
        if constexpr (std::is_same_v<T_, point_layer_tag>) return ltype::point;
        if constexpr (std::is_same_v<T_, areal_layer_tag>) return ltype::areal;
    }
   public:
    static constexpr std::array<int, Order> local_dim {Triangulation_::local_dim...};
    static constexpr std::array<int, Order> embed_dim {Triangulation_::embed_dim...};
    using index_t = int;
    using size_t  = std::size_t;

    // constructors
    GeoFrame() noexcept : triangulation_(), layers_(), n_layers_(0) { }
    explicit GeoFrame(Triangulation_&... triangulation) noexcept :
        triangulation_(std::make_tuple(std::addressof(triangulation)...)), layers_(), n_layers_(0) { }
    // modifiers
   private:
    template <typename... GeoInfo, typename... Args>
        requires(
          sizeof...(GeoInfo) == Order &&
          (internals::is_any_same_v<
             GeoInfo, std::tuple<internals::polygon_layer_descriptor, internals::point_layer_descriptor>> &&
           ...))
    auto& insert_scalar_layer_(const std::string& name, Args&&... args) {
        fdapde_static_assert(sizeof...(GeoInfo) == Order, BAD_LAYER_CONSTRUCTION__NO_MATCHING_ORDER);
	fdapde_assert(!name.empty() && !has_layer(name));
        using geo_layer_t = GeoLayer<Triangulation, std::tuple<GeoInfo...>>;
        layers_.emplace_back(
          name,                                                                // layer name
          internals::apply_index_pack<sizeof...(GeoInfo)>([&]<int... Ns>() {   // layer category
              return std::array<ltype, sizeof...(GeoInfo)> {ltype_from_layer_tag<typename GeoInfo::layer_tag>()...};
          }),
          internals::apply_index_pack<sizeof...(GeoInfo)>(   // data
            [&, this]<int... Ns>() { return geo_layer_t(std::forward<Args>(args)...); }));
        layer_name_to_idx_[name] = n_layers_;
        n_layers_++;
        return geo_cast<GeoInfo...>(operator[](name));
    }
   public:
    template <typename... GeoInfo, typename GeoData>
    auto& insert_scalar_layer(const std::string& name, GeoData&& data) {
        return insert_scalar_layer_<GeoInfo...>(name, triangulation_, data);
    }
    template <typename... GeoInfo, typename LayerType>
    auto& insert_scalar_layer(
      const std::string& name, const internals::random_access_geo_row_view<LayerType>& row_filter,
      const std::vector<std::string>& cols) {
        return insert_scalar_layer_<GeoInfo...>(name, row_filter, cols);
    }
    template <typename... GeoInfo, typename LayerType>
    auto& insert_scalar_layer(
      const std::string& name, const internals::random_access_geo_row_view<LayerType>& row_filter) {
        return insert_scalar_layer_<GeoInfo...>(name, row_filter);
    }
    auto& load_shp(const std::string& name, const std::string& filename) {
        fdapde_assert(!name.empty() && !has_layer(name));
        auto& l = insert_scalar_layer_<POLYGON>(name, triangulation_);
        l.load_shp(filename);
        return geo_cast<POLYGON>(operator[](name));
    }
    // observers
    int n_layers() const { return n_layers_; }
    const std::array<ltype, Order>& category(int layer_id) const { return layers_[layer_id].category(); }
    bool has_layer(const std::string& name) const {
        for (const layer_t& layer : layers_) {
            if (layer.name() == name) { return true; }
        }
        return false;
    }
    bool contains(const std::string& column) const {   // true if column is in at least one layer
        for (int i = 0; i < n_layers_; ++i) {
            if (operator[](i).contains(column)) { return true; }
        }
        return false;
    }
    std::vector<std::string> laynames() const {
        std::vector<std::string> names;
        for (const auto& [name, id] : layer_name_to_idx_) { names.push_back(name); }
        return names;
    }
    std::vector<std::string> colnames() const {
        std::vector<std::string> names;
        for (int i = 0; i < n_layers_; ++i) {
            std::vector<std::string> c = layers_[i].colnames();
            names.insert(names.end(), c.begin(), c.end());
        }
        return names;
    }
    template <int N> decltype(auto) triangulation() const { return *std::get<N>(triangulation_); }
    // indexed access
    const layer_t& operator[](int idx) const { return layers_[idx]; }
    layer_t& operator[](int idx) { return layers_[idx]; }
    layer_t& operator[](const std::string& layer_name) { return layers_.at(layer_name_to_idx_.at(layer_name)); }
    const layer_t& operator[](const std::string& layer_name) const {
        return layers_.at(layer_name_to_idx_.at(layer_name));
    }
  
    // iterator
    class iterator {
        const GeoFrame* gf_;
        int index_;
       public:
        using value_type = internals::scalar_data_layer;
        using pointer = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator(const GeoFrame* gf, int index) : gf_(gf), index_(index) { }
        const value_type* operator->() const { return std::addressof(gf_->operator[](index_)); }
        const value_type& operator*() const { return gf_->operator[](index_); }
        iterator& operator++() {
            index_++;
            return *this;
        }
        const std::array<ltype, Order>& category() const { return get().category(); }
        reference data() const { return operator*(); }
        const layer_t& get() const { return gf_->layers_[index_]; }
        friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
        friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
    };
    iterator begin() const { return iterator(this, 0); }
    iterator end() const { return iterator(this, n_layers_); }
   private:
    // data members
    std::tuple<std::add_pointer_t<std::decay_t<Triangulation_>>...> triangulation_;
    std::vector<layer_t> layers_;
    int n_layers_ = 0;
    std::unordered_map<std::string, int> layer_name_to_idx_;
};

// casts a GeoFrame layer to an instance of GeoLayer<GeoInfo...>
template <typename... GeoInfo, typename DataLayer> decltype(auto) geo_cast(DataLayer&& data_layer) {
    using DataLayer_ = std::remove_reference_t<DataLayer>;
    using GeoLayer_ = GeoLayer<typename std::decay_t<DataLayer>::Triangulation, std::tuple<GeoInfo...>>;
    return *reinterpret_cast<std::conditional_t<std::is_const_v<DataLayer_>, std::add_const_t<GeoLayer_>, GeoLayer_>*>(
      data_layer.geo_data());
}
// retrieve and casts a GeoFrame layer index
template <int N, typename GeoInfo, typename DataLayer> decltype(auto) geo_index_cast(DataLayer&& data_layer) {
    fdapde_static_assert(N < std::decay_t<DataLayer>::Order, OUT_OF_BOUND_ACCESS);
    using DataLayer_ = std::remove_reference_t<DataLayer>;
    using IndexT = internals::layer_type_from_layer_tag<
      GeoInfo, std::tuple_element_t<N, typename std::decay_t<DataLayer>::Triangulation>>;
    return *reinterpret_cast<std::conditional_t<std::is_const_v<DataLayer_>, std::add_const_t<IndexT>, IndexT>*>(
      data_layer.geo_index(N));
}

}   // namespace fdapde

#endif // __FDAPDE_GEOFRAME_H__
