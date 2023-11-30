//
// Created by andrei.pata on 7/10/19.
//
#include "mapillary.h"

namespace nie {
namespace dataset {

torch::data::TensorExample Mapillary::get(size_t index) {
    return loader_.Next();
}

torch::optional<size_t> Mapillary::size() const {
    return { loader_.Size() };
}

}  // dataset
}  // nie
