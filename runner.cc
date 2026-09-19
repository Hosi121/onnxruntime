// Load one CPU runtime per process. Read commands from standard input.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

#include <dlfcn.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc != 6 && argc != 7) {
      throw std::runtime_error(
          "Use: runner library model input shape threads [profile_prefix]");
    }
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
      throw std::runtime_error(dlerror());
    auto get_base = reinterpret_cast<const OrtApiBase *(*)()>(
        dlsym(library, "OrtGetApiBase"));
    if (!get_base)
      throw std::runtime_error(dlerror());
    const OrtApi* api = get_base()->GetApi(ORT_API_VERSION);
    if (!api)
      throw std::runtime_error("The runtime API version does not match.");
    Ort::InitApi(api);

    std::vector<int64_t> shape;
    std::stringstream shape_stream(argv[4]);
    std::string dimension;
    while (std::getline(shape_stream, dimension, ','))
      shape.push_back(std::stoll(dimension));
    size_t count = 1;
    for (int64_t d : shape) {
      if (d <= 0 ||
          static_cast<uint64_t>(d) > SIZE_MAX / sizeof(float) / count) {
        throw std::runtime_error("Invalid input shape.");
      }
      count *= static_cast<size_t>(d);
    }
    std::vector<float> input(count);
    std::ifstream input_file(argv[3], std::ios::binary | std::ios::ate);
    if (!input_file ||
        input_file.tellg() !=
            static_cast<std::streamoff>(input.size() * sizeof(float))) {
      throw std::runtime_error("The input file size does not match the shape.");
    }
    input_file.seekg(0);
    input_file.read(reinterpret_cast<char *>(input.data()),
                    input.size() * sizeof(float));
    if (!input_file)
      throw std::runtime_error("Cannot read the input file.");

    Ort::Env environment(ORT_LOGGING_LEVEL_ERROR, "col2im_bench");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(std::stoi(argv[5]));
    options.SetInterOpNumThreads(1);
    // The other test process must stay idle between commands.
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    if (argc == 7)
      options.EnableProfiling(argv[6]);
    Ort::Session session(environment, argv[2], options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
      throw std::runtime_error(
          "This runner requires one input and one output.");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char *input_names[] = {input_name.get()};
    const char *output_names[] = {output_name.get()};
    auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_value = Ort::Value::CreateTensor<float>(
        memory, input.data(), input.size(), shape.data(), shape.size());
    std::vector<Ort::Value> output;
    auto run = [&]() {
      output = session.Run(Ort::RunOptions{nullptr}, input_names, &input_value,
                           1, output_names, 1);
    };
    std::cout << "{\"ready\":true,\"version\":\""
              << get_base()->GetVersionString() << "\",\"build\":\""
              << Ort::GetApi().GetBuildInfoString() << "\"}\n"
              << std::flush;

    std::string command;
    while (std::cin >> command) {
      if (command == "quit")
        break;
      if (command == "warmup" || command == "time") {
        int iterations;
        std::cin >> iterations;
        if (iterations <= 0)
          throw std::runtime_error("Invalid iteration count.");
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i)
          run();
        const auto end = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(end - start).count() /
            iterations;
        std::cout << std::setprecision(12) << "{\"ms\":" << ms
                  << ",\"iterations\":" << iterations << "}\n";
      } else if (command == "dump") {
        std::string path;
        std::cin >> path;
        if (output.empty())
          throw std::runtime_error("Run the model before the output dump.");
        auto info = output.front().GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          throw std::runtime_error("The output must have type float32.");
        }
        const size_t output_count = info.GetElementCount();
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(
                       output.front().GetTensorData<float>()),
                   output_count * sizeof(float));
        if (!file)
          throw std::runtime_error("Cannot write the output file.");
        std::cout << "{\"elements\":" << output_count << "}\n";
      } else {
        throw std::runtime_error("Unknown command.");
      }
      std::cout << std::flush;
    }
    if (argc == 7) {
      auto path = session.EndProfilingAllocated(allocator);
      std::cout << "{\"profile\":\"" << path.get() << "\"}\n";
    }
    // Keep the library open until all runtime objects are destroyed.
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
