#include <iostream>
#include <fdeep/fdeep.hpp>

//     int layers=5, neurons=1024;

int main()
{
    std::vector<float> input{1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0};
    float gtruth = 17.5876;

    const auto model = fdeep::load_model("fdeep_model.json"); // , true, fdeep::dev_null_logger);

    const auto result = model.predict_single_output({fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(46)), input)});
    
    std::cout << "Predicted: " << result << "\nGround Truth: " << gtruth << std::endl;
}