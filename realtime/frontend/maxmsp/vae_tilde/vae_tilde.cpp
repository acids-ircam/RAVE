#include "../../../backend/backend.h"
#include "c74_min.h"
#include <string>
#include <thread>
#include <vector>

#define BUFFER_SIZE 2048

using namespace c74::min;

class vae : public object<vae>, public vector_operator<> {
public:
  MIN_DESCRIPTION{"Audio Variational Auto Encoder."};
  MIN_TAGS{"audio, deep learning, ai"};
  MIN_AUTHOR{"Antoine Caillon"};

  inlet<> m_audio_input{this, "(signal) Audio input", "signal"};
  outlet<> m_audio_output{this, "(signal) Audio output", "signal"};

  vae(const atoms &args = {}) : m_head(0), compute_thread(nullptr) {
    if (!args.size()) {
      cout << "vae~ must be initialized with a path to a pretrained model!"
           << endl;
      return;
    }
    if (args.size() > 0) {
      m_path = std::string(args[0]);
    }
    if (args.size() > 1) {
      m_method = std::string(args[1]);
    } else {
      m_method = "forward";
    }

    if (m_model.load(m_path)) {
      cout << "error during loading" << endl;
      return;
    } else {
      cout << "successfully loaded model" << endl;
    }

    auto params = m_model.get_method_params(m_method);

    if (!params.size()) {
      cout << "method " << m_method << " not found, using forward instead"
           << endl;
      m_method = "forward";
      params = m_model.get_method_params(m_method);
    }

    m_in_dim = params[0];
    m_in_ratio = params[1];
    m_out_dim = params[2];
    m_out_ratio = params[3];

    cout << "using method " << m_method << ", ";
    cout << "with " << m_in_dim << " input(s)";
    if (m_in_ratio != 1) {
      cout << " (" << m_in_ratio << "x downsampled)";
    }
    cout << " and " << m_out_dim << " output(s)";
    if (m_out_ratio != 1) {
      cout << " (" << m_out_ratio << "x upsampled)";
    }
    cout << "." << endl;
  }

  Backend m_model;

  std::string m_path, m_method;
  int m_head, m_in_dim, m_in_ratio, m_out_dim, m_out_ratio;

  float m_in_buffer[2 * BUFFER_SIZE], m_out_buffer[2 * BUFFER_SIZE];
  std::thread *compute_thread;

  void operator()(audio_bundle input, audio_bundle output);
};

void thread_perform(vae *vae_instance, std::vector<float *> in_buffer,
                    std::vector<float *> out_buffer, int n_vec,
                    std::string method) {
  vae_instance->m_model.perform(in_buffer, out_buffer, n_vec, method);
}

void vae::operator()(audio_bundle input, audio_bundle output) {

  auto in = input.samples(0);
  auto out = output.samples(0);

  for (int i(0); i < input.frame_count(); i++) {
    m_in_buffer[i + m_head] = float(in[i]);
    out[i] = double(m_out_buffer[i + m_head]);
  }

  m_head += input.frame_count();

  if (!(m_head % BUFFER_SIZE)) {
    if (compute_thread) {
      compute_thread->join();
    }

    m_head = m_head % (2 * BUFFER_SIZE);

    int offset_head = (m_head + BUFFER_SIZE) % (2 * BUFFER_SIZE);
    std::vector<float *> in_buffer{m_in_buffer + offset_head};
    std::vector<float *> out_buffer{m_out_buffer + offset_head};

    compute_thread = new std::thread(thread_perform, this, in_buffer,
                                     out_buffer, BUFFER_SIZE, m_method);
  }
}

MIN_EXTERNAL(vae);
