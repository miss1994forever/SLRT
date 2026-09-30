// CPU-only reconstruction of the frozen 153-D ridge fit from raw NPY payloads.
// The Python wrapper checks every input hash and validates predictions.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int D = 153;
constexpr int F = 5;

template <typename T> std::vector<T> read_exact(const std::string& path, size_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<T> value(count);
    in.read(reinterpret_cast<char*>(value.data()), count * sizeof(T));
    if (!in || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("unexpected input length: " + path);
    return value;
}

struct Stats {
    double n = 0, sy = 0;
    std::vector<double> sx = std::vector<double>(D, 0);
    std::vector<double> sxx = std::vector<double>(D * D, 0);
    std::vector<double> sxy = std::vector<double>(D, 0);
};

struct Model {
    std::vector<double> mean = std::vector<double>(D);
    std::vector<double> scale = std::vector<double>(D);
    std::vector<double> weight = std::vector<double>(D);
    double intercept = 0;
};

Model fit(const Stats& all, const Stats& held) {
    Model model;
    const double n = all.n - held.n;
    const double sy = all.sy - held.sy;
    model.intercept = sy / n;
    std::vector<double> matrix(D * D), rhs(D);
    for (int i = 0; i < D; ++i) {
        const double sx = all.sx[i] - held.sx[i];
        model.mean[i] = sx / n;
        const double second = (all.sxx[i * D + i] - held.sxx[i * D + i]) / n;
        const double variance = std::max(0.0, second - model.mean[i] * model.mean[i]);
        model.scale[i] = std::sqrt(variance);
        if (model.scale[i] < 1e-6) model.scale[i] = 1.0;
        rhs[i] = (all.sxy[i] - held.sxy[i] - sx * model.intercept) / model.scale[i];
    }
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            const double centered = all.sxx[i * D + j] - held.sxx[i * D + j]
                - n * model.mean[i] * model.mean[j];
            matrix[i * D + j] = centered / (model.scale[i] * model.scale[j]);
        }
        matrix[i * D + i] += 100.0;
    }
    // Cholesky with an explicit symmetry average to limit accumulation drift.
    std::vector<double> lower(D * D, 0), intermediate(D);
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j <= i; ++j) {
            double v = 0.5 * (matrix[i * D + j] + matrix[j * D + i]);
            for (int k = 0; k < j; ++k) v -= lower[i * D + k] * lower[j * D + k];
            if (i == j) {
                if (v <= 0 || !std::isfinite(v)) throw std::runtime_error("ridge Gram not positive definite");
                lower[i * D + j] = std::sqrt(v);
            } else lower[i * D + j] = v / lower[j * D + j];
        }
    }
    for (int i = 0; i < D; ++i) {
        double v = rhs[i];
        for (int j = 0; j < i; ++j) v -= lower[i * D + j] * intermediate[j];
        intermediate[i] = v / lower[i * D + i];
    }
    for (int i = D - 1; i >= 0; --i) {
        double v = intermediate[i];
        for (int j = i + 1; j < D; ++j) v -= lower[j * D + i] * model.weight[j];
        model.weight[i] = v / lower[i * D + i];
    }
    return model;
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("usage: fit K.raw target.raw folds.raw pred.raw model.bin row_count");
        const size_t rows = std::stoull(argv[6]);
        auto x = read_exact<float>(argv[1], rows * D);
        auto y = read_exact<float>(argv[2], rows);
        auto fold = read_exact<int8_t>(argv[3], rows);
        auto expected = read_exact<float>(argv[4], rows);
        Stats all;
        std::vector<Stats> by_fold(F);
        for (size_t r = 0; r < rows; ++r) {
            const int f = fold[r];
            if (f < 0 || f >= F) throw std::runtime_error("invalid fold");
            const float* row = x.data() + r * D;
            auto& held = by_fold[f];
            all.n += 1; held.n += 1;
            all.sy += y[r]; held.sy += y[r];
            for (int i = 0; i < D; ++i) {
                const double value = row[i];
                all.sx[i] += value; held.sx[i] += value;
                all.sxy[i] += value * y[r]; held.sxy[i] += value * y[r];
                for (int j = 0; j <= i; ++j) {
                    const double product = value * row[j];
                    all.sxx[i * D + j] += product;
                    held.sxx[i * D + j] += product;
                }
            }
        }
        for (Stats* stats : {&all})
            for (int i = 0; i < D; ++i) for (int j = 0; j < i; ++j)
                stats->sxx[j * D + i] = stats->sxx[i * D + j];
        for (auto& stats : by_fold)
            for (int i = 0; i < D; ++i) for (int j = 0; j < i; ++j)
                stats.sxx[j * D + i] = stats.sxx[i * D + j];
        std::vector<Model> models;
        for (int f = 0; f < F; ++f) models.push_back(fit(all, by_fold[f]));
        double max_error = 0, sum_sq = 0;
        size_t worst = 0;
        for (size_t r = 0; r < rows; ++r) {
            const Model& m = models[fold[r]];
            const float* row = x.data() + r * D;
            double predicted = m.intercept;
            for (int i = 0; i < D; ++i)
                predicted += (row[i] - m.mean[i]) / m.scale[i] * m.weight[i];
            const double error = std::abs(predicted - expected[r]);
            sum_sq += error * error;
            if (error > max_error) { max_error = error; worst = r; }
        }
        std::ofstream out(argv[5], std::ios::binary);
        if (!out) throw std::runtime_error("cannot write model");
        for (const auto& m : models) {
            for (const auto* part : {&m.mean, &m.scale, &m.weight})
                out.write(reinterpret_cast<const char*>(part->data()), D * sizeof(double));
            out.write(reinterpret_cast<const char*>(&m.intercept), sizeof(double));
        }
        std::cout << "{\"rows\":" << rows << ",\"max_abs_prediction_error\":" << max_error
                  << ",\"rmse_prediction_error\":" << std::sqrt(sum_sq / rows)
                  << ",\"worst_row\":" << worst << "}\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
