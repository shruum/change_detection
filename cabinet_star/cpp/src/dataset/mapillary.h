//
// Created by andrei.pata on 7/10/19.
//

#pragma once

#include "../util/util.h"

namespace nie {
namespace dataset {

// Handles only one image at a time.
class Mapillary : public torch::data::Dataset<Mapillary, torch::data::TensorExample> {
  public:
    explicit Mapillary(util::Loader &files) : loader_{files} {}

    torch::data::TensorExample get(size_t index) override;
    torch::optional<size_t> size() const override;

  private:
    util::Loader &loader_;
};

}  // namespace dataset
}  // namespace nie