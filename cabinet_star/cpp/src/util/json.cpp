//
// Created by andrei.pata on 7/22/19.
//

#include "json.h"

#include <fstream>

namespace nie {
namespace util {
namespace json {

using json = nlohmann::json;

void to_json(json &j, const Label &l) {
    j = json{{"color", l.color},
             {"instances", l.instances},
             {"readable", l.readable},
             {"name"}, l.name,
             {"evaluate"}, l.evaluate};
}

void from_json(const json &j, Label &l) {
    j.at("color").get_to(l.color);
    j.at("instances").get_to(l.instances);
    j.at("readable").get_to(l.readable);
    j.at("name").get_to(l.name);
    j.at("evaluate").get_to(l.evaluate);
}

std::vector<Label> GetLabels(std::string const &file_name) {
    std::ifstream file(file_name);
    json j;
    file >> j;

    auto labels = j.find("labels");
    auto nie_labels = std::vector<Label>{};
    if (labels != j.end()) {
        for (auto const &label : *labels) {
            nie_labels.emplace_back(label.get<Label>());
        }
    }

    return nie_labels;
}

}  // json
}  // util
}  // nie