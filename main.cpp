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

void lrelu(std::span<f64> s, std::span<f64> d) {
    for (usize i = 0; i < s.size(); ++i)
        d[i] = lrelu(s[i]);
}

void lrelu_d(std::span<f64> s, std::span<f64> d) {
    for (usize i = 0; i < s.size(); ++i)
        d[i] = lrelu_d(s[i]);
}

void softmax(std::span<f64> s, std::span<f64> d) {
    f64 sum = 0;

    for (usize i = 0; i < s.size(); ++i) {
        d[i] = std::exp(s[i]);
        sum += std::exp(s[i]);
    }

    for (usize i = 0; i < s.size(); ++i) {
        d[i] /= sum;
    }
}

void lrelu_d(std::span<f64> s) {
    for (auto &i: s) i = lrelu(i);
}


struct network {
    usize num_layers, num_batches;
    std::vector<usize> layer_sizes;
    std::vector<std::vector<f64>> outputs;
    std::vector<std::vector<f64>> activations;
    std::vector<std::vector<f64>> costs;
    std::vector<std::vector<std::vector<f64>>> weights;
    std::vector<std::vector<std::vector<f64>>> gradients;

    network(std::span<usize> layer_sizes_inp) : num_batches{},
                                                num_layers{layer_sizes_inp.size()},
                                                layer_sizes(layer_sizes_inp.size()),
                                                outputs(layer_sizes_inp.size()),
                                                activations(layer_sizes_inp.size()),
                                                costs(layer_sizes_inp.size()),
                                                weights(layer_sizes_inp.size()),
                                                gradients(layer_sizes_inp.size()) {


        for (usize i = 0; i < num_layers; ++i) {
            layer_sizes[i] = layer_sizes_inp[i] + 1;
        }

        for (usize i = 0; i < num_layers; ++i) {
            outputs[i] = std::vector<f64>(layer_sizes[i]);
            activations[i] = std::vector<f64>(layer_sizes[i]);
            costs[i] = std::vector<f64>(layer_sizes[i]);
        }

        for (usize i = 1; i < num_layers; ++i) {
            weights[i] = std::vector<std::vector<f64>>(layer_sizes[i], std::vector<f64>(layer_sizes[i - 1]));
            gradients[i] = std::vector<std::vector<f64>>(layer_sizes[i], std::vector<f64>(layer_sizes[i - 1]));
        }
    }

    void randomize_weights(auto &uniform) {
        for (usize layer = 1; layer < num_layers; ++layer) {
            for (usize i = 0; i < layer_sizes[layer]; ++i) {
                for (usize j = 0; j < layer_sizes[layer - 1]; ++j) {
                    weights[layer][i][j] = std::uniform_real_distribution<f64>{-0.5, 0.5}(uniform);
                }
            }
        }
    }

    void forward(std::span<f64> inp) {
        for (usize i = 0; i < inp.size(); ++i)
            outputs[0][i] = inp[i];

        for (usize layer = 1; layer < num_layers; ++layer) {
            std::span inp = outputs[layer - 1];
            inp[0] = 1; // bias
            std::span weight = weights[layer];
            std::span act = activations[layer];
            std::span out = outputs[layer];

            for (usize i = 0; i < layer_sizes[layer]; ++i)
                act[i] = 0;

            for (usize i = 0; i < layer_sizes[layer]; ++i)
                for (usize j = 0; j < layer_sizes[layer - 1]; ++j)
                    act[i] += weight[i][j] * inp[j];

            if (layer + 1 < num_layers)
                lrelu(act, out);
            else
                softmax(act, out);
        }
    }

    void backward(std::span<f64> desired) {
        // last layer
        costs.back()[0] = 0;
        for (usize i = 1; i < layer_sizes.back(); ++i)
            costs.back()[i] = outputs.back()[i] - desired[i - 1];

        // all other layers
        for (usize layer = num_layers - 1; layer-- > 0;) {
            std::span next_cost = costs[layer + 1];
            std::span next_weight = weights[layer + 1];
            std::span cur_cost = costs[layer];
            std::span cur_act = activations[layer];

            std::vector<f64> derivatives(layer_sizes[layer]);
            lrelu_d(cur_act, derivatives);

            for (usize i = 0; i < layer_sizes[layer]; ++i) {
                for (usize j = 0; j < layer_sizes[layer + 1]; ++j) {
                    cur_cost[i] += derivatives[i] * next_cost[j] * next_weight[j][i];
                }
            }
        }

        for (usize layer = 1; layer < num_layers; ++layer) {
            for (usize i = 0; i < layer_sizes[layer]; ++i) {
                for (usize j = 0; j < layer_sizes[layer - 1]; ++j) {
                    gradients[layer][i][j] += outputs[layer - 1][j] * costs[layer][i];
                }
            }
        }
        num_batches += 1;
    }

    void apply_gradients(f64 learning_rate) {
        auto fix_nan_inf = [](f64 &x) {
            if (std::isnan(x) || std::isinf(x)) x = 0;
        };
        for (usize layer = 1; layer < num_layers; ++layer) {
            for (usize i = 0; i < layer_sizes[layer]; ++i) {
                for (usize j = 0; j < layer_sizes[layer - 1]; ++j) {
                    fix_nan_inf(gradients[layer][i][j]);
                    weights[layer][i][j] -= std::clamp<f64>(gradients[layer][i][j] * learning_rate / num_batches, -1, 1);
                    fix_nan_inf(weights[layer][i][j]);
                    gradients[layer][i][j] = 0;
                }
            }
        }
        num_batches = 0;
    }
};

int main() {
    std::mt19937_64 mt{std::random_device{}()};

    std::vector<usize> backing_permutation(dataset.training_images.size());
    std::iota(std::begin(backing_permutation), std::end(backing_permutation), 0);

    std::vector<usize> layer_sizes = {28 * 28, 16, 10};
    network best(layer_sizes);
    usize best_num_correct = -1;
    network net(layer_sizes);

    if (false) {
        std::ifstream dat("best_network");
        for (auto &vv: std::span(net.weights).subspan(1)) {
            for (auto &v: vv) {
                for (auto &i: v) {
                    dat >> i;
                }
            }
        }
    } else {
        net.randomize_weights(mt);
    }
    std::vector<usize> full_permutation(dataset.training_images.size());
    std::iota(std::begin(full_permutation), std::end(full_permutation), 0);

    usize images_per_epoch = 1024;

    for (usize epoch = 0; epoch < 100'000; ++epoch) {
        std::shuffle(std::begin(full_permutation), std::end(full_permutation), mt);

        std::span permutation = std::span{full_permutation}.subspan(0, images_per_epoch);

        f64 loss = 0;
        usize num_correct = 0;
        for (auto idx: permutation) {
            auto &img = dataset.training_images[idx];
            auto &label = dataset.training_labels[idx];

            std::array<f64, 10> target{};
            target[label] = 1;

            std::array<f64, 28 * 28> input;
            for (usize i = 0; i < 28 * 28; ++i)
                input[i] = img[i] / 255.0;
            net.forward(input);
            net.backward(target);

            usize highest = 0;

            for (usize i = 0; i < 10; ++i) {
                if (net.outputs.back()[i] > net.outputs.back()[highest]) {
                    highest = i;
                }
                loss -= target[i] * std::log(net.outputs.back()[i]);
            }
            num_correct += highest == label;
        }
        if (num_correct > best_num_correct) {
            best = net;
            best_num_correct = num_correct;

            std::ofstream outfile("best_network");
            outfile.precision(10);
            for (auto &vv: std::span(best.weights).subspan(1)) {
                for (auto &v: vv) {
                    for (auto &i: v) {
                        outfile << i << ' ';
                    }
                }
            }
        }

        // https://www.jonathan-petitcolas.com/2017/12/28/converting-image-to-ascii-art.html
        char lightness[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.                 ";
        for (usize i = 0; i < 28; ++i) {
            for (usize j = 0; j < 28; ++j) {
                std::cerr << lightness[dataset.training_images[permutation[0]][i * 28 + j] * 80 / 255];
            }
            std::cerr << '\n';
        }
        std::array<f64, 28 * 28> input;
        for (usize i = 0; i < 28 * 28; ++i)
            input[i] = dataset.training_images[permutation[0]][i] / 255.0;
        net.forward(input);
        auto choice = 0;
        for (usize i = 0; i < 10; ++i) {
            std::cerr << net.outputs.back()[i] << ' ';
            if (net.outputs.back()[i] > net.outputs.back()[choice]) choice = i;
        }
        std::cerr << "\nnetwork choice: " << choice << " with probability: " << net.outputs.back()[choice] * 100 << "%\n";
        std::cerr << "correct:" << (int) dataset.training_labels[permutation[0]] << '\n';
        net.apply_gradients(0.01);
        std::cerr << "loss=" << loss / images_per_epoch << " accuracy=" << num_correct * 1.0 / images_per_epoch << "\n";
    }
}
