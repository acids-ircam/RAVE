#include "backend.h"
#include <stdlib.h>

backend::backend() : m_loaded(0) { at::init_num_threads(); }

void backend::perform(std::vector<float *> in_buffer,
                      std::vector<float *> out_buffer, int n_vec,
                      std::string method) {
  torch::NoGradGuard no_grad;
  if (m_loaded) {

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
