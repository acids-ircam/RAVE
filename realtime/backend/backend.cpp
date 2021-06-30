#include "backend.h"
#include <iostream>
#include <stdlib.h>

backend::backend() : m_loaded(0) { at::init_num_threads(); }

void backend::perform(std::vector<float *> in_buffer,
                      std::vector<float *> out_buffer, int n_vec,
                      std::string method) {
  torch::NoGradGuard no_grad;
  if (m_loaded) {
    // COPY BUFFER INTO A TENSOR
    std::vector<at::Tensor> tensor_in;
    for (int i(0); i < in_buffer.size(); i++) {
      tensor_in.push_back(torch::from_blob(in_buffer[i], {1, 1, n_vec}));
    }
    auto cat_tensor_in = torch::cat(tensor_in, 1);
    std::vector<torch::jit::IValue> inputs = {cat_tensor_in};

    // PROCESS TENSOR
    auto tensor_out = m_model.get_method(method)(inputs).toTensor();
    int out_channels(tensor_out.size(1)), out_n_vec(tensor_out.size(2));

    if (out_channels != out_buffer.size()) {
      std::cout << "bad out_buffer size, expected " << out_channels
                << " buffers, got " << out_buffer.size() << "!\n";
      return;
    }

    auto out_ptr = tensor_out.contiguous().data_ptr<float>();

    for (int i(0); i < out_buffer.size(); i++) {
      memcpy(out_buffer[i], out_ptr + i * n_vec, n_vec * sizeof(float));
    }
  }
}

int backend::load(std::string path) {
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
