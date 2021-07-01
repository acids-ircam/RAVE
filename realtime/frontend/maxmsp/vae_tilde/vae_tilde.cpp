#include "../../../backend/backend.h"
#include "c74_min.h"
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

  vae() : m_head(0), compute_thread(nullptr) {}

  Backend m_model;
  float m_in_buffer[2 * BUFFER_SIZE], m_out_buffer[2 * BUFFER_SIZE];
  int m_head;
  std::thread *compute_thread;

  message<> load{this, "load", "load a pretrained ai model into the external",
                 MIN_FUNCTION{auto status = m_model.load(args[0]);

  if (status != 0) {
    cout << "failed loading model" << endl;
  } else {
    cout << "successfully loaded model" << endl;
  }
  return {};
}
}
;

void operator()(audio_bundle input, audio_bundle output);
}
;

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
                                     out_buffer, BUFFER_SIZE, "forward");
  }
}

MIN_EXTERNAL(vae);
