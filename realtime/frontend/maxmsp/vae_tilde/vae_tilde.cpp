#include "../../../backend/backend.h"
#include "c74_min.h"

using namespace c74::min;

class vae : public object<vae>, public vector_operator<> {
public:
  MIN_DESCRIPTION{"Audio Variational Auto Encoder."};
  MIN_TAGS{"audio, deep learning, ai"};
  MIN_AUTHOR{"Antoine Caillon"};

  inlet<> m_audio_input{this, "(signal) Audio input", "signal"};
  outlet<> m_audio_output{this, "(signal) Audio output", "signal"};

  Backend m_model;

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

void operator()(audio_bundle input, audio_bundle output) {}
}
;

MIN_EXTERNAL(vae);
