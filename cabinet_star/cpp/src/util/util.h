//
// Created by andrei.pata on 7/10/19.
//

#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "json.h"

namespace nie {
namespace util {

/**
 * @brief Loads files from a provided folder and provide one by one.
 */
class Loader {
  public:
    /**
     * @brief Constructs a vector with the names of the regular files found in a provided folder.
     * @param data_folder Folder path from where to get the file's names.
     */
    explicit Loader(std::string data_folder, int target_size);

    size_t Size() const { return file_names_.size(); }

    std::string DirName() const;
    std::string Name() const;
    std::string BaseName() const;
    std::string FullName() const;

    /**
     * @brief Provide the current cv::Mat.
     * @return Reference to the current cv::Mat.
     */
    cv::Mat const &Get() const { return cur_img_; }

    cv::Size const &GetAdjSize() const { return adj_size_; }

    /**
     * @brief Updates `cur_` and `cur_img_` with the next image and provides the corresponding current cv::Mat.
     * @return Reference to the next cv::Mat.
     */
    torch::Tensor Next();

  private:
    std::string data_folder_;
    std::vector<std::string> file_names_;
    cv::Mat cur_img_;
    cv::Size adj_size_;  // New size after changing one side to 1024 and the other keeping the original ratio.
    int target_size_;    // Desired size of one of the sides.
    int cur_{-1};
};

/**
 * @brief Print 'items' elements but no more than 'max'. It could be printed starting from 'p' or in reverse order
 * starting from 'p+max-1'.
 * @tparam T Type
 * @param p Memory address.
 * @param items Number of elements to be printed.
 * @param max Maximum number of elements to be printed.
 * @param ascending Printing order.
 */
template<typename T>
void PrintData(void const *const p, size_t items, size_t max, bool ascending = true) {
    auto data = reinterpret_cast<T const *const>(p);
    if (items > max) {
        items = max;
    }

    if (ascending) {
        for (size_t i = 0; i < items; ++i) {
            std::cout << +data[i] << "\n";
        }
    } else {
        for (size_t i = 1; i <= items; ++i) {
            std::cout << +data[max - i] << "\n";
        }
    }
}

/**
 * @brief A raw memory representation of a binary file, image file or torch::Tensor 'data_ptr()'.
 * @tparam T Primitive data type.
 */
template<typename T>
class Data {
  public:
    /**
     * @brief Constructor from a binary file.
     * @param file_name Binary file name.
     * @param shape Shape of the original data.
     */
    Data(std::string const &file_name, std::vector<int> shape) : file_name_{file_name}, shape_{std::move(shape)} {
        std::ifstream file(file_name, std::ios::binary);
        data_ = std::shared_ptr<T[]>(new T[Size() * sizeof(T)], [](T const *p) { delete[] p; });
        auto tmp = data_.get();
        if (file.is_open()) {
            while (file.read(reinterpret_cast<char *>(tmp), sizeof(T))) { tmp++; }
            file.close();
        } else {
            std::cerr << "Error while opening file: " << file_name << "!\n";
        }
    }

    /**
     * @brief Constructor from a image with 3 channels.
     * @param file_name Image file name.
     */
    explicit Data(std::string const &file_name) : file_name_{file_name} {
        cv::Mat img = cv::imread(file_name, cv::IMREAD_COLOR);
        shape_.emplace_back(3);
        shape_.emplace_back(img.cols);
        shape_.emplace_back(img.rows);

        data_ = std::shared_ptr<T[]>(new T[Size() * sizeof(T)], [](T const *p) { delete[] p; });
        std::memcpy(data_.get(), img.data, Size() * sizeof(T));
    }

    /**
     * @brief Constructor from a torch::Tensor.
     * @param tensor torch::Tensor.
     */
    explicit Data(torch::Tensor const &tensor) {
        file_name_ = "Tensor";
        for (int i = 0; i < tensor.dim(); ++i) { shape_.emplace_back(tensor.sizes()[i]); }

        data_ = std::shared_ptr<T[]>(new T[Size() * sizeof(T)], [](T const *p) { delete[] p; });
        std::memcpy(data_.get(), tensor.data_ptr(), Size() * sizeof(T));
    }

    T *GetDataPtr() const { return data_.get(); }

    std::vector<int> const &Shape() const { return shape_; }

    int Size() const {
        auto size = int{0};

        if (!shape_.empty()) {
            size = 1;
            for (auto const &item : shape_) { size *= item; }
        }

        return size;
    }

    void DisplayImage(int type, double fx = 0, double fy = 0) {
        auto img = cv::Mat(cv::Size{shape_[1], shape_[2]}, type, data_.get());
        if (fx || fy) {
            cv::resize(img, img, cv::Size(), fx, fy);
        }
        cv::imshow(file_name_, img);
        cv::waitKey(0);
    }

    void PrintData(bool ascending = true) {
        util::PrintData<T>(data_.get(), Size(), Size(), ascending);
    }

    void PrintDataItems(size_t items, bool ascending = true) {
        util::PrintData<T>(data_.get(), items, Size(), ascending);
    }

  private:
    std::string file_name_ = "Unknown";
    std::shared_ptr<T[]> data_;
    std::vector<int> shape_;
};  // class Data

/**
 * @brief Convert an image to tensor.
 *
 * Converts from BGR to RGB. Splits the interleaved BGR channels and rearranges them in tensor format.
 * Normalizes the pixels values [0, 255] -> [0.f, 1.f].
 *
 * @param loc Image file name.
 * @return Tensor
 */
torch::Tensor ImgToTensor(cv::Mat loc, int target_size, cv::Size &adj_size);

/**
 * @brief Post process the prediction: Add borders and resize back to the original size.
 * @param predict The prediction.
 * @param nie_loader The loader.
 * @return The post-processed prediction as a cv::Mat.
 */
cv::Mat PostProcess(at::Tensor const &predict, Loader const &nie_loader, size_t num_labels);

/**
 * @brief Wrapper to cv::CopyMakeBorder. Add border (zeros) to `src` to get to the `dst` size.
 * @param src Source image.
 * @param dst Destination image.
 */
void CopyMakeBorder(cv::Mat const &src, cv::Mat &dst, size_t num_labels);

/**
 * @brief Get a vector of identified polygons of each class in a cv::Mat.
 * @param mat The matrix to be checked.
 * @param num_classes The number of classes.
 * @param min_area The minimum area of accepted polygons.
 * @return The vectors of polygons for each class.
 */
std::vector<std::vector<std::vector<cv::Point>>> MaskToPolygons(cv::Mat const &mat,
                                                                size_t num_classes,
                                                                unsigned min_area);

/**
 * @brief Save a cv::Mat to a binary file.
 * @param mat The matrix to be saved.
 * @param dst The destination folder.
 * @param loader The loader.
 */
void SaveToBinary(cv::Mat const &mat, std::string const &dst, Loader const &loader);

/**
 * @brief Save polygons to a JSON file that has the template: `template.json`.
 * @param labels_polygons The polygons per each label that were found.
 * @param json_obj The JSON file where to save the predictions.
 * @param loader The loader.
 * @param labels The labels read from `config.json` file.
 * @param pretty_json Whether to make JSON file more readable or not.
 * @param dst The output directory. Don't save to file if empty string.
 */
void SaveToJson(std::vector<std::vector<std::vector<cv::Point>>> const &labels_polygons,
                nlohmann::json &json_obj,
                Loader const &loader,
                std::vector<json::Label> const &labels,
                bool pretty_json,
                std::string const &dst);

/**
 * @brief Color polygons and save result as a PNG file.
 * @param labels_polygons The polygons per each label that were found.
 * @param loader The loader.
 * @param labels The labels read from `config.json` file.
 * @param dst The output directory. Don't save to file if empty string.
 */
void SaveToPng(std::vector<std::vector<std::vector<cv::Point>>> const &labels_polygons,
               Loader const &loader,
               std::vector<json::Label> const &labels,
               std::string const &dst);

};  // namespace util
};  // namespace nie
