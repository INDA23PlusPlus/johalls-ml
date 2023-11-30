#include "mnist/include/mnist/mnist_reader.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <random>
#include <span>
#include <thread>
#include <vector>

const auto dataset = mnist::read_dataset<std::vector, std::vector, std::uint8_t, std::uint8_t>();

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using usize = std::size_t;
using isize = std::make_signed_t<usize>;

using f32 = float;
using f64 = double;

constexpr usize WIDTH = 28;
constexpr usize HEIGHT = 28;


// credit: me
// https://github.com/chopingu/hajar/tree/main/src/neural_net_testing
namespace gya {
template<class T, usize... sizes>
class layer_array {
private:
    constexpr static auto layer_sizes = std::array{sizes...};

    constexpr static auto indices = [] {
        std::array<usize, sizeof...(sizes)> arr{};
        for (usize i = 0, sum = 0; i < arr.size(); ++i) {
            arr[i] = sum;
            sum += std::array{sizes...}[i];
        }
        return arr;
    }();

public:
    std::array<T, (sizes + ...)> m_data{};

    constexpr std::span<T> operator[](usize idx) {
        return std::span<T>(m_data.data() + indices[idx], m_data.data() + indices[idx] + layer_sizes[idx]);
    }

    constexpr std::span<T const> operator[](usize idx) const {
        return std::span<T const>(m_data.data() + indices[idx], m_data.data() + indices[idx] + layer_sizes[idx]);
    }

    constexpr usize size() const {
        return sizeof...(sizes);
    }

    constexpr std::span<T> front() {
        return operator[](0);
    }

    constexpr std::span<T const> front() const {
        return operator[](0);
    }

    constexpr std::span<T> back() {
        return operator[](size() - 1);
    }

    constexpr std::span<T const> back() const {
        return operator[](size() - 1);
    }

    constexpr auto fill(T const &value) {
        m_data.fill(value);
    }

    constexpr auto data() {
        return m_data.data();
    }

    constexpr auto data() const {
        return m_data.data();
    }
};
template<class T, usize... sizes>
class weight_array {
    template<class G>
    struct matrix_ref {
        friend weight_array;

    private:
        G *m_data;
        usize m_subarray_len;

        // private to disallow creating a matrix_ref outside of weight_array
        constexpr matrix_ref(G *data, usize subarray_len) : m_data{data}, m_subarray_len{subarray_len} {}

    public:
        constexpr std::span<G> operator[](usize idx) {
            return std::span<G>{m_data + idx * m_subarray_len, m_data + (idx + 1) * m_subarray_len};
        }

        constexpr std::span<G const> operator[](usize idx) const {
            return std::span<G const>{m_data + idx * m_subarray_len, m_data + (idx + 1) * m_subarray_len};
        }

        constexpr usize size() const {
            return m_subarray_len;
        }
    };

private:
    constexpr static auto layer_sizes = std::array{sizes...};

    constexpr static auto indices = [] {
        std::array<usize, sizeof...(sizes)> arr{};
        usize sum = 0;
        for (usize i = 0; i < sizeof...(sizes) - 1; ++i) {
            arr[i] = sum;
            sum += layer_sizes[i] * layer_sizes[i + 1];
        }
        arr.back() = sum;
        return arr;
    }();

public:
    std::array<T, indices.back()> m_data{};

    constexpr matrix_ref<T> operator[](usize idx) {
        return matrix_ref<T>{m_data.data() + indices[idx], layer_sizes[idx + 1]};
    }

    constexpr matrix_ref<T const> operator[](usize idx) const {
        return matrix_ref<T const>{m_data.data() + indices[idx], layer_sizes[idx + 1]};
    }

    constexpr usize size() const {
        return sizeof...(sizes);
    }

    constexpr auto front() {
        return operator[](0);
    }

    constexpr auto front() const {
        return operator[](0);
    }

    constexpr auto back() {
        return operator[](size() - 1);
    }

    constexpr auto back() const {
        return operator[](size() - 1);
    }

    constexpr auto fill(T const &value) {
        m_data.fill(value);
    }

    constexpr auto data() {
        return m_data.data();
    }

    constexpr auto data() const {
        return m_data.data();
    }
};
} // namespace gya
using namespace gya;

f64 lrelu(f64 x) {
    if (x < 0)
        return 0.05 * x;
    else
        return x;
}

f64 lrelu_d(f64 x) {
    if (x < 0)
        return 0.05;
    else
        return 1;
}

f64 sigmoid(f64 x) {
    return 1.0 / (1.0 + std::exp(-x));
}

f64 sigmoid_d(f64 x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

using layer_array_t = layer_array<f64, WIDTH * HEIGHT, 1024, 128, 32, 10>;
using weight_array_t = weight_array<f64, WIDTH * HEIGHT, 1024, 128, 32, 10>;

layer_array_t network_values;
weight_array_t weights;
layer_array_t biases;

std::mutex summation_mutex;
weight_array_t weights_derivative_accum;
layer_array_t biases_derivative_accum;

int main() {
    for (auto &i: weights.m_data) i = 0.1 + (rand() * (1. / RAND_MAX) - .5) * 0.02;
    for (auto &i: biases.m_data) i = 0.1 + (rand() * (1. / RAND_MAX) - .5) * 0.02;

    std::vector<usize> backing_permutation(dataset.training_images.size());
    std::iota(std::begin(backing_permutation), std::end(backing_permutation), 0);

    std::mt19937_64 mt{std::random_device{}()};
    f64 learning_rate = 0.01;
    auto forward_prop = [&](auto &img) {
        for (usize i = 0; i < network_values[0].size(); ++i) {
            network_values[0][i] = img[i] / 255.0;
        }

        {
            for (usize layer = 1; layer + 1 < network_values.size(); ++layer) {
                for (usize i = 0; i < network_values[layer].size(); ++i) {
                    auto sum = biases[layer][i];
                    for (usize j = 0; j < network_values[layer - 1].size(); ++j) {
                        sum += weights[layer - 1][j][i] * network_values[layer - 1][j];
                    }
                    network_values[layer][i] = lrelu(sum);
                }
            }
            const auto layer = network_values.size() - 1;
            for (usize i = 0; i < network_values[layer].size(); ++i) {
                auto sum = biases[layer][i];
                for (usize j = 0; j < network_values[layer - 1].size(); ++j) {
                    sum += weights[layer - 1][j][i] * network_values[layer - 1][j];
                }
                network_values[layer][i] = sigmoid(sum);
            }
        }
    };
    constexpr auto num_threads = 256;
    std::vector<std::function<void(void)>> tasks[num_threads];
    std::jthread threads[num_threads];
    std::atomic<bool> do_work[num_threads];
    std::atomic<usize> not_done_threads = 0;
    std::atomic<bool> program_is_done = false;

    auto do_tasks = [&]() {
        for (auto &i: do_work) i = true;
        not_done_threads = num_threads;
        while (not_done_threads > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    auto add_task = [&, i = 0](auto &&task) mutable {
        tasks[i].push_back({std::move(task)});
        i = (i + 1) % num_threads;
    };

    for (usize i = 0; i < num_threads; ++i) {
        new (&threads[i]) std::jthread{[&, i]() {
            while (!program_is_done) {
                if (!do_work[i]) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    std::this_thread::yield();
                    continue;
                } else {
                    while (!tasks[i].empty()) {
                        tasks[i].back()();
                        tasks[i].pop_back();
                    }
                    --not_done_threads;
                    do_work[i] = false;
                }
            }
        }};
    }

    for (usize epoch = 0; epoch <= 1'000'000; ++epoch) {
        std::shuffle(std::begin(backing_permutation), std::end(backing_permutation), mt);

        std::span permutation = std::span(backing_permutation).subspan(0, 1000);


        for (auto idx: permutation) {
            add_task([&, idx] {
                auto &img = dataset.training_images[idx];
                auto &label = dataset.training_labels[idx];
                forward_prop(img);
                { // backward prop

                    layer_array_t correct_vals{};

                    std::span correct_output = correct_vals.back();
                    correct_output[label] = 1;
                    std::unique_ptr backing = std::make_unique<weight_array_t>();
                    weight_array_t &weight_derivatives = *backing;
                    for (auto &i: weight_derivatives.m_data) i = 0;
                    layer_array_t bias_derivatives{};

                    const std::span output{network_values.back()};
                    const usize num_layers = network_values.size();

                    // process last layer
                    for (usize node = 0; node < output.size(); ++node) {
                        bias_derivatives.back()[node] =
                                sigmoid_d(output[node]) * (output[node] - correct_output[node]);
                    }

                    // process all layers between last and first layer (exclusive)
                    for (usize layer = num_layers - 2; layer > 0; --layer) {
                        for (usize node = 0; node < network_values[layer].size(); ++node) {
                            f64 sum = 0;
                            for (usize next_node = 0; next_node < network_values[layer + 1].size(); ++next_node) {
                                const f64 weight_derivative = network_values[layer][node] * bias_derivatives[layer + 1][next_node];
                                sum += weight_derivative;
                                weight_derivatives[layer][node][next_node] = weight_derivative;
                            }
                            const f64 bias_derivative = lrelu_d(network_values[layer][node]) * sum;
                            bias_derivatives[layer][node] = bias_derivative;
                        }
                    }

                    // process first layer
                    for (usize node = 0; node < network_values[0].size(); ++node) {
                        for (usize next_node = 0; next_node < network_values[1].size(); ++next_node) {
                            const f64 weight_derivative = network_values[0][node] * bias_derivatives[1][next_node];
                            weight_derivatives[0][node][next_node] = weight_derivative;
                        }
                    }

                    std::scoped_lock lock(summation_mutex);
                    for (usize i = 0; i < weights.m_data.size(); ++i)
                        weights_derivative_accum.m_data[i] += (1.0 / permutation.size()) * weight_derivatives.m_data[i];
                    for (usize i = 0; i < biases.m_data.size(); ++i)
                        biases_derivative_accum.m_data[i] += (1.0 / permutation.size()) * bias_derivatives.m_data[i];
                }
            });
        }

        do_tasks();

        if (epoch % 100 == 0) {
            for (usize i = 0; i < weights.m_data.size(); ++i)
                weights.m_data[i] -= std::clamp<f64>(learning_rate * weights_derivative_accum.m_data[i], -1, 1);
            for (usize i = 0; i < biases.m_data.size(); ++i)
                biases.m_data[i] -= std::clamp<f64>(learning_rate * biases_derivative_accum.m_data[i], -1, 1);
            std::cerr.precision();

            std::atomic<f64> sum_error = 0;
            std::atomic<usize> num_correct = 0;
            const usize num_test_images = 1000;
            for (usize i = 0, iters = num_test_images; iters-- > 0; i = (i + 17) % dataset.test_images.size()) {
                add_task([&, i] {
                    forward_prop(dataset.test_images[i]);
                    auto label = dataset.test_labels[i];
                    std::array<f64, 10> correct{};
                    correct[label] = 1.0;
                    usize biggest = 0;
                    for (usize j = 0; j < 10; ++j) {
                        sum_error += std::pow(network_values.back()[j] - correct[j], 2);
                        if (network_values.back()[j] > network_values.back()[biggest]) {
                            biggest = j;
                        }
                    }
                    assert(label < 10);
                    assert(biggest < 10);
                    num_correct += biggest == label;
                });
            }
            do_tasks();

            std::cerr << std::setprecision(4) << std::fixed;
            std::cerr << "loss: " << std::setw(8) << (sum_error / num_test_images) / network_values.back().size() << "\t\t";
            std::cerr << "accuracy: " << std::setw(8) << num_correct * 100.0 / num_test_images << "%\t\t";
            std::cerr << "learning rate: " << std::setw(8) << learning_rate << ' ';
            std::cerr << std::endl;
            learning_rate *= 0.99;
        }
    }
    program_is_done = true;
}
