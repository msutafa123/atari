// main.cpp - XOR problem solver using Tensor Library
// C++17 compliant

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>

// Include necessary headers
#include "Tensor.h"
#include "TensorOps.h"
#include "LinearForward.h"
#include "ActivationForward.h"
#include "Sequential.h"

using namespace tensor;

int main() {
    std::cout << "=== XOR Problem Solver using Tensor Library ===" << std::endl;

    // Create XOR dataset
    // Input: 2D points (0,0), (0,1), (1,0), (1,1)
    // Output: 0, 1, 1, 0 respectively
    Tensor<float> inputs({ 4, 2 });
    Tensor<float> targets({ 4, 1 });

    // (0,0) -> 0
    inputs.at({ 0, 0 }) = 0.0f;
    inputs.at({ 0, 1 }) = 0.0f;
    targets.at({ 0, 0 }) = 0.0f;

    // (0,1) -> 1
    inputs.at({ 1, 0 }) = 0.0f;
    inputs.at({ 1, 1 }) = 1.0f;
    targets.at({ 1, 0 }) = 1.0f;

    // (1,0) -> 1
    inputs.at({ 2, 0 }) = 1.0f;
    inputs.at({ 2, 1 }) = 0.0f;
    targets.at({ 2, 0 }) = 1.0f;

    // (1,1) -> 0
    inputs.at({ 3, 0 }) = 1.0f;
    inputs.at({ 3, 1 }) = 1.0f;
    targets.at({ 3, 0 }) = 0.0f;

    // Create a neural network with one hidden layer
    forward::Sequential<float> model;

    // Input layer (2) -> Hidden layer (4) with ReLU activation
    auto fc1 = std::make_shared<forward::LinearForward<float>>(2, 4);
    auto relu = std::make_shared<forward::ReLUForward<float>>();

    // Hidden layer (4) -> Output layer (1) with Sigmoid activation
    auto fc2 = std::make_shared<forward::LinearForward<float>>(4, 1);
    auto sigmoid = std::make_shared<forward::SigmoidForward<float>>();

    // Add layers to model
    model.add(fc1);
    model.add(relu);
    model.add(fc2);
    model.add(sigmoid);

    std::cout << "Neural Network Structure:" << std::endl;
    model.summary();

    // Initialize weights with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& param : fc1->parameters()) {
        for (size_t i = 0; i < param->size(); ++i) {
            param->data()[i] = dist(gen);
        }
    }

    for (auto& param : fc2->parameters()) {
        for (size_t i = 0; i < param->size(); ++i) {
            param->data()[i] = dist(gen);
        }
    }

    // Training parameters
    float learning_rate = 0.1f;
    int epochs = 10000;

    std::cout << "\nTraining the network to solve XOR..." << std::endl;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        // Forward pass
        Tensor<float> outputs = model.forward(inputs);

        // Calculate MSE loss
        Tensor<float> diff = outputs - targets;
        Tensor<float> squared_diff = diff * diff;

        for (size_t i = 0; i < squared_diff.size(); ++i) {
            total_loss += squared_diff.data()[i];
        }
        total_loss /= squared_diff.size();

        // Backward pass (manual implementation since we don't have full autograd)
        // This is a simplified backpropagation for the specific XOR case

        // Output layer gradients
        Tensor<float> d_outputs({ 4, 1 });
        for (size_t i = 0; i < 4; ++i) {
            float o = outputs.at({ i, 0 });
            float t = targets.at({ i, 0 });
            // Gradient of MSE with respect to output
            float d_loss = 2 * (o - t) / 4;
            // Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            d_outputs.at({ i, 0 }) = d_loss * o * (1 - o);
        }

        // Get hidden layer output (after ReLU)
        Tensor<float> hidden_outputs = relu->forward(fc1->forward(inputs));

        // Update fc2 weights
        for (size_t i = 0; i < 4; ++i) { // hidden neurons
            for (size_t j = 0; j < 4; ++j) { // samples
                float grad = d_outputs.at({ j, 0 }) * hidden_outputs.at({ j, i });
                fc2->weights().at({ i, 0 }) -= learning_rate * grad;
            }
        }

        // Update fc2 bias
        for (size_t j = 0; j < 4; ++j) { // samples
            fc2->bias().at({ 0 }) -= learning_rate * d_outputs.at({ j, 0 });
        }

        // Calculate hidden layer gradients
        Tensor<float> d_hidden({ 4, 4 });
        for (size_t i = 0; i < 4; ++i) { // samples
            for (size_t j = 0; j < 4; ++j) { // hidden neurons
                // Gradient from output layer
                float grad = d_outputs.at({ i, 0 }) * fc2->weights().at({ j, 0 });
                // Apply ReLU derivative (1 if input > 0, 0 otherwise)
                if (hidden_outputs.at({ i, j }) > 0) {
                    d_hidden.at({ i, j }) = grad;
                }
                else {
                    d_hidden.at({ i, j }) = 0;
                }
            }
        }

        // Update fc1 weights
        for (size_t i = 0; i < 2; ++i) { // input features
            for (size_t j = 0; j < 4; ++j) { // hidden neurons
                float grad = 0;
                for (size_t k = 0; k < 4; ++k) { // samples
                    grad += d_hidden.at({ k, j }) * inputs.at({ k, i });
                }
                fc1->weights().at({ i, j }) -= learning_rate * grad;
            }
        }

        // Update fc1 bias
        for (size_t j = 0; j < 4; ++j) { // hidden neurons
            float grad = 0;
            for (size_t k = 0; k < 4; ++k) { // samples
                grad += d_hidden.at({ k, j });
            }
            fc1->bias().at({ j }) -= learning_rate * grad;
        }

        // Print progress
        if (epoch % 1000 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss << std::endl;
        }
    }

    // Test the trained model
    std::cout << "\nTesting the trained model:\n" << std::endl;
    Tensor<float> test_outputs = model.forward(inputs);

    for (size_t i = 0; i < 4; ++i) {
        std::cout << "Input: [" << inputs.at({ i, 0 }) << ", " << inputs.at({ i, 1 }) << "]"
            << ", Target: " << targets.at({ i, 0 })
            << ", Predicted: " << test_outputs.at({ i, 0 }) << std::endl;
    }

    std::cout << "\n=== XOR Problem Solved! ===" << std::endl;

    return 0;
}