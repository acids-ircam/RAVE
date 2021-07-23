#include "c74_min.h"
#include "c74_min_unittest.h"
#include "vae_tilde.cpp"
#include <iostream>

SCENARIO("object produces correct output") {
  ext_main(nullptr);

  GIVEN("An instance of vae~ without parameters") {
    vae my_object;
    WHEN("a buffer is given") {
      sample_vector input(4096);
      sample_vector output;

      for (int i(0); i < 10; i++) {
        for (auto x : input) {
          auto y = my_object(x);
          output.push_back(y);
        }
      }
    }
  }

  GIVEN("An instance of vae~ with parameters") {
    atom path("/Users/acaillon/Desktop/vae.ts"), method("forward");
    atoms args = {path, method};
    vae my_object = vae(args);

    WHEN("a buffer is given") {
      sample_vector input(4096);
      sample_vector output;

      for (int i(0); i < 10; i++) {
        for (auto x : input) {
          auto y = my_object(x);
          output.push_back(y);
        }
      }
    }
  }
}