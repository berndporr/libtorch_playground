#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <torch/torch.h>

int main(int, char **) {

    auto x = torch::ones({2, 2}, torch::requires_grad());
    std::cout << x << std::endl;
    auto y = x + 2;
    std::cout << y << std::endl;
    std::cout << y.grad_fn()->name() << std::endl;
    auto z = y * y * 3;
    auto out = z.mean();
    
    std::cout << z << std::endl;
    std::cout << z.grad_fn()->name() << std::endl;
    std::cout << out << std::endl;
    std::cout << out.grad_fn()->name() << std::endl;
    
    auto a = torch::randn({2, 2});
    a = ((a * 3) / (a - 1));
    std::cout << a.requires_grad() << std::endl;
    
    a.requires_grad_(true);
    std::cout << a.requires_grad() << std::endl;
    
    auto b = (a * a).sum();
    std::cout << b.grad_fn()->name() << std::endl;
    
    out.backward();
    
    
//  vector-Jacobian product

    x = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
    y = x * x;  // y = [1, 4, 9]
    auto v = torch::tensor({0.1, 1.0, 0.0001});
    y.retain_grad();
    y.backward(v);
    
    std::cout << "y = " << y << std::endl;
    std::cout << "y.grad_fn()->name() = " << y.grad_fn()->name() << std::endl;
    
    std::cout << "x.grad() = " << x.grad() << std::endl;
    std::cout << "y.grad() = " << y.grad() << std::endl;
    
    return 0;
}
