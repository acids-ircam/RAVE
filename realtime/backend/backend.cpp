#include "backend.h"
#include <algorithm>
#include <iostream>
#include <stdlib.h>

Backend::Backend() : m_loaded(0) { at::init_num_threads(); }

void Backend::perform(std::vector<float *> in_buffer,
                      std::vector<float *> out_buffer, int n_vec,
                      std::string method) {
  torch::NoGradGuard no_grad;

  auto params = get_method_params(method);
  if (!params.size())
    return;

  auto in_dim = params[0];
  auto in_ratio = params[1];
  auto out_dim = params[2];
  auto out_ratio = params[3];

  if (!m_loaded)
    return;

  // COPY BUFFER INTO A TENSOR
  std::vector<at::Tensor> tensor_in;
  for (int i(0); i < in_buffer.size(); i++) {
    tensor_in.push_back(torch::from_blob(in_buffer[i], {1, 1, n_vec}));
  }
  auto cat_tensor_in = torch::cat(tensor_in, 1);
  cat_tensor_in = cat_tensor_in.reshape({1, in_dim, -1, in_ratio}).mean(-1);

  std::vector<torch::jit::IValue> inputs = {cat_tensor_in};

  // PROCESS TENSOR
  at::Tensor tensor_out;
  try {
    tensor_out = m_model.get_method(method)(inputs).toTensor();
    tensor_out =
        tensor_out.repeat_interleave(out_ratio).reshape({1, out_dim, -1});
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return;
  }

  int out_channels(tensor_out.size(1)), out_n_vec(tensor_out.size(2));

  // CHECKS ON TENSOR SHAPE
  if (out_channels != out_buffer.size()) {
    std::cout << "bad out_buffer size, expected " << out_channels
              << " buffers, got " << out_buffer.size() << "!\n";
    return;
  }

  if (out_n_vec != n_vec) {
    std::cout << "model output size is not consistent, expected " << n_vec
              << " samples, got " << out_n_vec << "!\n";
    return;
  }

  auto out_ptr = tensor_out.contiguous().data_ptr<float>();

  for (int i(0); i < out_buffer.size(); i++) {
    memcpy(out_buffer[i], out_ptr + i * n_vec, n_vec * sizeof(float));
  }
}

int Backend::load(std::string path) {
  try {
    m_model = torch::jit::load(path);
    m_model.eval();
    m_loaded = 1;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

std::vector<std::string> Backend::get_available_methods() {
  std::vector<std::string> methods;
  for (const auto &m : m_model.get_methods())
    methods.push_back(m.name());
  return methods;
}

std::vector<std::string> Backend::get_available_attributes() {
  std::vector<std::string> attributes;
  for (const auto &attribute : m_model.named_attributes())
    attributes.push_back(attribute.name);
  return attributes;
}

std::vector<int> Backend::get_method_params(std::string method) {
  auto am = get_available_methods();
  std::vector<int> params;

  if (std::find(am.begin(), am.end(), method) != am.end()) {

    auto p = m_model.attr(method + "_params").toTensor();

    for (int i(0); i < 4; i++)
      params.push_back(p[i].item().to<int>());
  }
  return params;
}
