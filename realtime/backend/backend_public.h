#pragma once
#include <string>
#include <vector>

class BackendPublic {
public:
  BackendPublic();
  ~BackendPublic();
  void perform(std::vector<float *> in_buffer, std::vector<float *> out_buffer,
               int n_vec, std::string method);
  int load(std::string path);
};
