#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <tuple>
#include <vector>

static int usage() {
    std::cerr << "      Usage: ./strassen [DEBUG] [DIMENSION] [INPUT]\n";
    return -1;
}

static int to_int(std::string str) {
    int res;
    std::stringstream ss;

    ss << str;
    ss >> res;

    return res;
}

struct global_stats {
    long multiplications;
    long additions;
    long ints_allocated;
};
static global_stats stats;

struct matrix_data {
    std::shared_ptr<std::vector<int>> data;
    int N;
};

inline int ceil_divide(int x) { return x / 2 + (x % 2 != 0); }

struct matrix {
    int i, j;
    int n;
    matrix_data data;

    matrix(int N) {
        i = 0;
        j = 0;
        n = N;

        data.data = std::make_shared<std::vector<int>>(N * N);
        data.N = N;

        stats.ints_allocated += N * N;
    }

    matrix() {}

    matrix sub(int x, int y) {
        matrix submatrix;

        submatrix.data = data;
        submatrix.n = ceil_divide(n);

        submatrix.i = i + x * submatrix.n;
        submatrix.j = j + y * submatrix.n;

        return submatrix;
    }

    inline int at(int x, int y) const {
        if (x + i < data.N && y + j < data.N) {
            return data.data->at(x + i + data.N * (y + j));
        }

        return 0;
    }

    inline void set(int x, int y, int value) {
        if (x + i < data.N && y + j < data.N) {
            data.data->at(x + i + data.N * (y + j)) = value;
        }
    }
};

std::ostream& operator<<(std::ostream& os, const matrix& mat) {
    for (int x = 0; x < mat.n; ++x) {
        for (int y = 0; y < mat.n; ++y) {
            os << mat.at(x, y) << " ";
        }
        os << "\n";
    }

    return os;
}

int main(int argc, const char** argv) {
    // Validate inputs
    if (argc == 4) {
        std::vector<std::string> args(argv + 1, argv + argc);

        int debug = to_int(args.at(0));
        int dimension = to_int(args.at(1));
        int cutoff = 1;

        if (dimension > 0) {
            // Allocate matrices
            matrix a(dimension);
            matrix b(dimension);

            if ((debug & 0x01) != 0) {
                srand(time(NULL));
                for (int i = 0; i < dimension; ++i) {
                    for (int j = 0; j < dimension; ++j) {
                        a.set(i, j, rand() % 2);
                    }
                }
                for (int i = 0; i < dimension; ++i) {
                    for (int j = 0; j < dimension; ++j) {
                        b.set(i, j, rand() % 2);
                    }
                }
                cutoff = to_int(args.at(2));
            } else {
                // Read data from file
                std::string line;
                std::ifstream file(args.at(2));

                if (file.is_open()) {
                    int i = 0;
                    while (getline(file, line)) {
                        if (i < dimension * dimension) {
                            a.data.data->at(i) = to_int(line);
                        } else {
                            b.data.data->at(i - dimension * dimension) =
                                to_int(line);
                        }
                        ++i;
                    }
                    file.close();
                } else {
                    // Error handling
                    std::cerr << "      Unable to open file: \"" << args.at(2)
                              << "\"" << std::endl;
                    return -1;
                }
            }

            using matrix_operation =
                std::function<void(matrix, matrix, matrix)>;

            matrix_operation sum;
            sum = [&](matrix a, matrix b, matrix c) {
                for (int y = 0; y < c.n; ++y) {
                    for (int x = 0; x < c.n; ++x) {
                        c.set(x, y, a.at(x, y) + b.at(x, y));
                        ++stats.additions;
                    }
                }
            };

            matrix_operation sub;
            sub = [&](matrix a, matrix b, matrix c) {
                for (int y = 0; y < c.n; ++y) {
                    for (int x = 0; x < c.n; ++x) {
                        c.set(x, y, a.at(x, y) - b.at(x, y));
                        ++stats.additions;
                    }
                }
            };

            matrix_operation mul;
            mul = [&](matrix a, matrix b, matrix c) {
                assert(a.n == b.n && b.n == c.n);

                if (c.n <= cutoff) {
                    for (int x = 0; x < c.n; ++x) {
                        for (int y = 0; y < c.n; ++y) {
                            for (int z = 0; z < c.n; ++z) {
                                c.set(x, y,
                                      c.at(x, y) + a.at(x, z) * b.at(z, y));
                                ++stats.additions;
                                ++stats.multiplications;
                            }
                        }
                    }
                } else {
                    int m = ceil_divide(c.n);
                    matrix sum0(m);
                    matrix sum1(m);

                    // Calculate smaller products
                    matrix m0(m);
                    sum(a.sub(0, 0), a.sub(1, 1), sum0);
                    sum(b.sub(0, 0), b.sub(1, 1), sum1);
                    mul(sum0, sum1, m0);

                    matrix m1(m);
                    sum(a.sub(1, 0), a.sub(1, 1), sum0);
                    mul(sum0, b.sub(0, 0), m1);

                    matrix m2(m);
                    sub(b.sub(0, 1), b.sub(1, 1), sum1);
                    mul(a.sub(0, 0), sum1, m2);

                    matrix m3(m);
                    sub(b.sub(1, 0), b.sub(0, 0), sum1);
                    mul(a.sub(1, 1), sum1, m3);

                    matrix m4(m);
                    sum(a.sub(0, 0), a.sub(0, 1), sum0);
                    mul(sum0, b.sub(1, 1), m4);

                    matrix m5(m);
                    sub(a.sub(1, 0), a.sub(0, 0), sum0);
                    sum(b.sub(0, 0), b.sub(0, 1), sum1);
                    mul(sum0, sum1, m5);

                    matrix m6(m);
                    sub(a.sub(0, 1), a.sub(1, 1), sum0);
                    sum(b.sub(1, 0), b.sub(1, 1), sum1);
                    mul(sum0, sum1, m6);

                    // Use products to compute total product
                    matrix c00 = c.sub(0, 0);
                    matrix c10 = c.sub(1, 0);
                    matrix c01 = c.sub(0, 1);
                    matrix c11 = c.sub(1, 1);

                    sum(m0, m3, c00);
                    sum(c00, m6, c00);
                    sub(c00, m4, c00);
                    sum(m2, m4, c01);
                    sum(m1, m3, c10);
                    sum(m0, m2, c11);
                    sum(c11, m5, c11);
                    sub(c11, m1, c11);
                }
            };

            auto start = std::chrono::high_resolution_clock::now();
            matrix c(dimension);
            mul(a, b, c);
            auto end = std::chrono::high_resolution_clock::now();

            if ((debug & 0x04) != 0) {
                std::cout
                    << std::chrono::duration_cast<std::chrono::milliseconds>(
                           end - start)
                           .count()
                    << std::endl;
            }

            if ((debug & 0x02) != 0) {
                std::cout << "A:\n" << a;
                std::cout << "B:\n" << b;
                std::cout << "C:\n" << c;
            }

            stats.ints_allocated -= 3 * dimension * dimension;

            if ((debug & 0x08) != 0) {
                std::cout << "additions: " << stats.additions << '\n';
                std::cout << "multiplications: " << stats.multiplications
                          << '\n';
                std::cout << "ints_allocated: " << stats.ints_allocated << '\n';
            }

            return 0;
        }
    }

    return usage();
}
