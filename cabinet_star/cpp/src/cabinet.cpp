#include <iostream>
#include <memory>

#include <args.hxx>
#include <cuda_runtime_api.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "dataset/mapillary.h"
#include "util/nvtx.h"

int main(int argc, char *argv[]) {
    args::ArgumentParser parser("Cabinet C++ inference application.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::Group mandatory(parser, "This group is all mandatory:", args::Group::Validators::All);
    args::ValueFlag<std::string> classes_config_file(mandatory,
                                                     "string",
                                                     "'config.json' file containing ordered list of colors and labels "\
                                                     "corresponding to classes.",
                                                     {"classes_config_file"});
    args::ValueFlag<std::string> data_path(mandatory,
                                           "string",
                                           "Path to input data.",
                                           {"data_path"});
    args::ValueFlag<std::string> model_file(mandatory,
                                            "string",
                                            "Model file to be used. Must be in torch script format.",
                                            {"model_file"});
    args::ValueFlag<std::string> output_path(mandatory,
                                             "string",
                                             "Path to where to output the predictions.",
                                             {"output_path"});

    args::Group optional(parser, "This group is optional:", args::Group::Validators::DontCare);
    args::Flag generate_binaries(optional,
                                 "bool",
                                 "Generate binaries from raw predictions ('min_area' is not applied).",
                                 {"generate_binaries"});
    args::Flag generate_png(optional,
                            "bool",
                            "Generate colorful labelmap outputs.",
                            {"generate_png"});
    args::ValueFlag<unsigned int> min_area(optional,
                                           "unsigned int",
                                           "Minimum size area of prediction. (Default: 50).",
                                           {"min_area"},
                                           50);
    args::Flag no_normalization(optional,
                                "bool",
                                "Disable normalization.",
                                {"no_normalization"});
    args::Flag one_json(optional,
                        "bool",
                        "Output all predictions in the same JSON file.",
                        {"one_json"});
    args::Flag pretty_json(optional,
                           "bool",
                           "Generate JSON file in a more readable format. This increases the file size.",
                           {"pretty_json"});
    args::ValueFlag<unsigned int> resize_to(optional,
                                            "unsigned int",
                                            "Size before inference. (Default: 1024).",
                                            {"resize_to"},
                                            1024);

    try {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Help &) {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (const args::ValidationError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    auto dst = args::get(output_path);

    auto file = std::ifstream{};

    torch::NoGradGuard no_grad;
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // Deserialize the ScriptModule.
    torch::jit::script::Module model = torch::jit::load(args::get(model_file), device);
    model.eval();
    std::cout << "Model loaded to " << (torch::cuda::is_available() ? "GPU" : "CPU") << "\n\n";

    auto nie_loader = nie::util::Loader(args::get(data_path), args::get(resize_to));

    auto mean = std::vector<double>{0.485, 0.456, 0.406};
    auto stddev = std::vector<double>{0.229, 0.224, 0.225};
    if (args::get(no_normalization)) {
        mean.assign({0, 0, 0});
        stddev.assign({1, 1, 1});
    }
    auto data_set = nie::dataset::Mapillary(nie_loader).map(torch::data::transforms::Normalize<void>(mean, stddev));
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(data_set), 1);

    auto labels = nie::util::json::GetLabels(args::get(classes_config_file));
    nlohmann::json json_objs;

    for (auto &batch : *data_loader) {
        nie::nvtx::PUSH_RANGE(nie_loader.Name(), nie::nvtx::Color::kGreen);
        auto input_tensor = batch.data()->data;
        input_tensor = input_tensor.to(device);

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input_tensor);

        std::cout << "Run prediction on " << nie_loader.Name() << "\n";
        nie::nvtx::PUSH_RANGE("forward", nie::nvtx::Color::kRed);
        auto outputs = model.forward(inputs).toTuple()->elements();
        nie::nvtx::POP_RANGE();  // forward

        for (auto const &output : outputs) {
            nie::nvtx::PUSH_RANGE("post-process", nie::nvtx::Color::kDarkGreen);
            auto predict = std::get<1>(output.toTensor().max(1)).squeeze().to(at::kCPU).to(at::kByte);

            auto predict_proc = nie::util::PostProcess(predict, nie_loader, labels.size());
            nie::nvtx::POP_RANGE();  // post-process

            if (args::get(generate_binaries)) {
                nie::util::SaveToBinary(predict_proc, dst, nie_loader);
            }

            auto polygons = nie::util::MaskToPolygons(predict_proc, labels.size(), args::get(min_area));

            nlohmann::json json_obj;
            if (args::get(one_json)) {
                nie::util::SaveToJson(polygons, json_obj, nie_loader, labels, args::get(pretty_json), "");
                json_objs["imgs"][nie_loader.BaseName()] = json_obj["imgs"][nie_loader.BaseName()];
            } else {
                nie::util::SaveToJson(polygons, json_obj, nie_loader, labels, args::get(pretty_json), dst);
            }

            if (args::get(generate_png)) {
                nie::util::SaveToPng(polygons, nie_loader, labels, dst);
            }
        }
        nie::nvtx::POP_RANGE();  // nie_loader.Name()
    }

    // All predictions goes in one single very big JSON file.
    if (args::get(one_json)) {
        auto json_file_name = dst + "/predicts.json";
        auto json_file = std::ofstream{json_file_name};
        json_file << (args::get(pretty_json) ? std::setw(2) : std::setw(0)) << json_objs << "\n";
        json_file.close();
    }

    cudaDeviceReset();
}
