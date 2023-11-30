//
// Created by andrei.pata on 7/22/19.
//

#pragma once

#include <string>
#include <vector>

// In case of requests for JSON file as an ordered collection: https://github.com/nlohmann/json#order-of-object-keys .
#include <nlohmann/json.hpp>

namespace nie {
namespace util {
namespace json {

using json = nlohmann::json;

struct Label {
    std::vector<int> color;
    bool instances;
    std::string readable;
    std::string name;
    bool evaluate;
};

void to_json(json &j, const Label &l);

void from_json(const json &j, Label &l);

std::vector<Label> GetLabels(std::string const &file_name);

}  // json
}  // util
}  // nie