#include "backend.h"
#include <iostream>
#include <stdlib.h>

Backend::Backend() : m_loaded(0) { at::init_num_threads(); }

void Backend::perform(std::vector<float *> in_buffer,
                      std::vector<float *> out_buffer, int n_vec,
                      std::string method) {
  torch::NoGradGuard no_grad;

  if (!m_loaded)
    return;

  // COPY BUFFER INTO A TENSOR
  std::vector<at::Tensor> tensor_in;
  for (int i(0); i < in_buffer.size(); i++) {
    tensor_in.push_back(torch::from_blob(in_buffer[i], {1, 1, n_vec}));
  }
  auto cat_tensor_in = torch::cat(tensor_in, 1);
  std::vector<torch::jit::IValue> inputs = {cat_tensor_in};

  // PROCESS TENSOR
  at::Tensor tensor_out;
  try {
    tensor_out = m_model.get_method(method)(inputs).toTensor();
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