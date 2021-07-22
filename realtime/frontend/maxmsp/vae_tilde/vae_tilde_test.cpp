#include "c74_min_unittest.h" // required unit test header
#include "vae_tilde.cpp"
#include <iostream>

SCENARIO("object produces correct output") {
  ext_main(nullptr); // every unit test must call ext_main() once to configure
                     // the class

  GIVEN("An instance of vae~") {

    test_wrapper<vae> an_instance;
    vae &my_object = an_instance;

    WHEN("a buffer is given") {
      sample_vector input(4096);
      sample_vector output;

      for (int i(0); i < 10; i++) {
        for (auto x : input) {
          auto y = my_object(x);
          // std::cout << y << std::endl;
          output.push_back(y);
        }
      }
    }
  }
}