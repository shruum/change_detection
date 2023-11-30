//
// Created by andrei.pata on 7/10/19.
//

#include "util.h"

#include <dirent.h>
#include <iostream>

#include "nvtx.h"

namespace nie {
namespace util {

Loader::Loader(std::string data_folder, int const target_size)
    : data_folder_{std::move(data_folder)},
      target_size_{target_size} {
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(data_folder_.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..") && ent->d_type != DT_DIR) {
                file_names_.emplace_back(ent->d_name);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Couldn't open directory: " << data_folder_ << "!\n";
    }
}

std::string Loader::DirName() const {
    return data_folder_;
}

std::string Loader::Name() const {
    return file_names_[cur_];
}

std::string Loader::BaseName() const {
    return file_names_[cur_].substr(0, file_names_[cur_].rfind('.'));
}

std::string Loader::FullName() const {
    return data_folder_ + '/' + file_names_[cur_];
}

torch::Tensor Loader::Next() {
    nie::nvtx::PUSH_RANGE("Img read", nie::nvtx::Color::kBlue);
    cur_img_ = cv::imread(data_folder_ + '/' + file_names_[++cur_]);
    adj_size_.height = target_size_;
    adj_size_.width = target_size_;

    nie::nvtx::POP_RANGE();  // Img read

    return ImgToTensor(cur_img_, target_size_, adj_size_);
}

torch::Tensor ImgToTensor(cv::Mat img, int const target_size, cv::Size &adj_size) {
    nie::nvtx::PUSH_RANGE("ImgToTensor", nie::nvtx::Color::kDarkGreen);
    // std::cout << "Initial image dimension: " << img.cols << " X " << img.rows << "\n";

    auto ratio = (float) img.cols / img.rows;
    auto cols_crop = target_size;
    auto rows_crop = target_size;
    auto offset_cols = 0;
    auto offset_rows = 0;

    auto pref_diff_cols = abs(img.cols - target_size);
    auto pref_diff_rows = abs(img.rows - target_size);

    if (pref_diff_cols < pref_diff_rows) {
        adj_size.height = static_cast<int>(target_size / ratio);
        cv::resize(img, img, cv::Size(target_size, adj_size.height));

        rows_crop = static_cast<int>(adj_size.height / 32) * 32;
        offset_rows = (adj_size.height - rows_crop) / 2;
    } else {
        adj_size.width = static_cast<int>(target_size * ratio);
        cv::resize(img, img, cv::Size(adj_size.width, target_size));

        cols_crop = static_cast<int>(adj_size.width / 32) * 32;
        offset_cols = (adj_size.width - cols_crop) / 2;
    }

    auto const roi = cv::Rect{offset_cols, offset_rows, cols_crop, rows_crop};
    img = img(roi).clone();

    // std::cout << "Cropped image dimension: " << img.cols << " X " << img.rows << "\n";

    cvtColor(img, img, cv::COLOR_BGR2RGB);

    cv::Mat rgb[3];
    cv::split(img, rgb);
    cv::Mat channelsConcatenated;
    vconcat(rgb[0], rgb[1], channelsConcatenated);
    vconcat(channelsConcatenated, rgb[2], channelsConcatenated);

    auto t = torch::from_blob(channelsConcatenated.data,
                              {1, img.channels(), img.rows, img.cols},
                              at::kByte).clone().to(at::kFloat).div(255);
    nie::nvtx::POP_RANGE();  // ImgToTensor

    return t;
};

cv::Mat PostProcess(at::Tensor const &predict, Loader const &nie_loader, size_t const num_labels) {
    auto predict_cv = cv::Mat{cv::Size{static_cast<int>(predict.sizes()[1]),
                                       static_cast<int>(predict.sizes()[0])},
                              CV_8U,
                              predict.data_ptr()};

    auto predict_cv_padded = cv::Mat{cv::Size{nie_loader.GetAdjSize().width,
                                              nie_loader.GetAdjSize().height},
                                     CV_8U};
    // Add padding for cropped images.
    CopyMakeBorder(predict_cv, predict_cv_padded, num_labels);

    cv::resize(predict_cv_padded, predict_cv_padded, cv::Size(nie_loader.Get().cols, nie_loader.Get().rows));

    return predict_cv_padded;
}

void CopyMakeBorder(cv::Mat const &src, cv::Mat &dst, size_t const num_labels) {
    int const rows_diff = dst.rows - src.rows;
    int const cols_diff = dst.cols - src.cols;

    int const top = rows_diff / 2;
    int bottom = rows_diff - top;

    int const left = cols_diff / 2;
    int right = cols_diff - left;

    // (num_labels - 1) = background.
    cv::copyMakeBorder(src, dst, top, bottom, left, right, cv::BORDER_CONSTANT, num_labels - 1);
}

void SaveToBinary(cv::Mat const &mat, std::string const &dst, Loader const &loader) {
    nie::nvtx::PUSH_RANGE("SaveToBinary", nie::nvtx::Color::kBlue);
    auto size = mat.total();
    auto out_data = mat.data;

    auto binary_file_name = dst + '/';
    binary_file_name.append(loader.BaseName() + ".dat");
    std::ofstream file_out(binary_file_name);

    for (size_t i = 0; i < size; ++i) {
        file_out.write(reinterpret_cast<char *>(out_data), sizeof(unsigned char));
        out_data++;
    }

    file_out.close();
    nie::nvtx::POP_RANGE();  // SaveToBinary
}

std::vector<std::vector<std::vector<cv::Point>>> MaskToPolygons(cv::Mat const &mat,
                                                                size_t num_labels,
                                                                unsigned min_area) {
    nie::nvtx::PUSH_RANGE("MaskToPolygons", nie::nvtx::Color::kBlue);
    auto labels_polygons = std::vector<std::vector<std::vector<cv::Point>>>{};

    // (num_labels - 1) - don't search for background.
    for (size_t i = 0; i < num_labels; ++i) {
        auto contours = std::vector<std::vector<cv::Point>>{};

        auto mask = mat.clone();
        mask.setTo(0, mask != i);
        mask.setTo(1, mask == i);

        cv::findContours(mask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        contours.erase(std::remove_if(contours.begin(), contours.end(),
                                      [min_area](auto contour) { return cv::contourArea(contour) < min_area; }),
                       contours.end());

        labels_polygons.push_back(std::move(contours));
    }

    nie::nvtx::POP_RANGE();  // MaskToPolygons

    return labels_polygons;
}

void SaveToJson(std::vector<std::vector<std::vector<cv::Point>>> const &labels_polygons,
                nlohmann::json &json_obj,
                Loader const &loader,
                std::vector<json::Label> const &labels,
                bool pretty_json,
                std::string const &dst) {
    json_obj["imgs"][loader.BaseName()]["inputlas"];
    json_obj["imgs"][loader.BaseName()]["width"] = loader.Get().cols;
    json_obj["imgs"][loader.BaseName()]["height"] = loader.Get().rows;
    json_obj["imgs"][loader.BaseName()]["path"] = loader.FullName();

    for (size_t i = 0; i < labels_polygons.size(); ++i) {
        for (auto const &polygon : labels_polygons[i]) {
            nlohmann::json object;

            object["bbox"]["xmax"] = nullptr;
            object["bbox"]["xmin"] = nullptr;
            object["bbox"]["ymax"] = nullptr;
            object["bbox"]["ymin"] = nullptr;
            object["category"] = labels[i].name;
            auto polygon_as_vect = std::vector<size_t>{};
            for (auto const &point : polygon) {
                polygon_as_vect.emplace_back(point.x);
                polygon_as_vect.emplace_back(point.y);
            }
            object["polygon"] = polygon_as_vect;
            object["score"] = nullptr;
            object["color"] = labels[i].color;
            object["contourl"]["PArr"];
            object["contourl"]["PSArr"];
            object["contourl"]["CArrclass"];

            json_obj["imgs"][loader.BaseName()]["objects"].push_back(object);
        }
    }

    if (!dst.empty()) {
        auto json_file_name = dst + '/';
        json_file_name.append(loader.BaseName() + ".json");
        auto json_file = std::ofstream{json_file_name};
        json_file << (pretty_json ? std::setw(2) : std::setw(0)) << json_obj << "\n";
        json_file.close();
    }
}

void SaveToPng(std::vector<std::vector<std::vector<cv::Point>>> const &labels_polygons,
               Loader const &loader,
               std::vector<json::Label> const &labels,
               std::string const &dst) {
    auto file_name = dst + '/';
    file_name.append(loader.BaseName() + ".png");

    auto img = cv::Mat{cv::Size(loader.Get().cols, loader.Get().rows), CV_8UC3};

    for (size_t i = 0; i < labels_polygons.size(); ++i) {
        auto color = labels[i].color;

        // Need to draw contours iterating over the collection using `contourIdx` to avoid incorrect results.
        // See the note from `cv::drawContours` description.
        for (size_t contourIdx = 0; contourIdx < labels_polygons[i].size(); ++contourIdx) {
            cv::drawContours(img, labels_polygons[i], contourIdx,
                             cv::Scalar(color[0], color[1], color[2]), cv::FILLED);
        }
    }

    cvtColor(img, img, cv::COLOR_RGB2BGR);

    cv::imwrite(file_name, img);
}

}  // namespace util
}  // namespace nie