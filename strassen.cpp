#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

enum debug_flags {
    RANDOM = 0x01,  // should generate random instead of reading from a file
    PRINT = 0x02,   // print matrices to the screen
    VERIFY = 0x04,  // verify the matrix multiplication
    TIME = 0x08,    // time the functions
};

static int usage() {
    std::cerr << "      Usage: ./strassen [DEBUG] [DIMENSION] [INPUT]\n";
    std::cerr << "          debug flags:\n";
    std::cerr << "              RANDOM      :" << debug_flags::RANDOM << "\n";
    std::cerr << "              PRINT       :" << debug_flags::PRINT << "\n";
    std::cerr << "              VERIFY      :" << debug_flags::VERIFY << "\n";

    return -1;
}

static int to_int(std::string str) {
    int res;
    std::stringstream ss;

    ss << str;
    ss >> res;

    return res;
}

static void time(std::function<void(void)> func) {
    // Perform the multiplications
    auto start = std::chrono::system_clock::now();
    func();
    auto end = std::chrono::system_clock::now();

    auto dur = end - start;
    auto s = std::chrono::duration_cast<std::chrono::seconds>(dur);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur -= s);

    std::cout << s.count() << "s " << ms.count() << "ms\n";
}

static inline int ceil_divide(int x) { return x / 2 + (x % 2 != 0); }

int padding_size(int dimension, int cutoff) {
    int power = 0;
    while (dimension > cutoff) {
        power++;
        dimension = ceil_divide(dimension);
    }
    return dimension * (1 << power);
}

// Represents a two dimension set of integers
class matrix_data {
   public:
    int dimension;
    std::shared_ptr<std::vector<int>> data;

    matrix_data(int dimension) : dimension(dimension) {
        data = std::make_shared<std::vector<int>>(dimension * dimension, 0);
    }

    inline bool in_bounds(int i, int j) const {
        return i >= 0 && j >= 0 && i + dimension * j < int(data->size());
    }

    int get(int i, int j) const { return data->at(i + dimension * j); }

    int& at(int i, int j) { return data->at(i + dimension * j); }
    int& at(int i) { return data->at(i); }
};

// Represents a submatrix which points to some two dimensional block of data
class submatrix {
   public:
    submatrix(matrix_data data) : data(data) {
        i = 0;
        j = 0;

        dimension = data.dimension;
    };

    matrix_data data;
    int i, j, dimension;

    // Garbage variable for use in 'at()' method
    int padding = 0;

    submatrix sub(int x, int y) {
        submatrix sub(data);

        sub.dimension = ceil_divide(dimension);
        sub.i = i + x * sub.dimension;
        sub.j = j + y * sub.dimension;

        return sub;
    }

    inline bool in_bounds(int x, int y) {
        return data.in_bounds(x + i, y + j) &&
               (x >= 0 && x < dimension && y >= 0 && y < dimension);
    }

    void clear() {
        for (int j = 0; j < dimension; ++j) {
            for (int i = 0; i < dimension; ++i) {
                at(i, j) = 0;
            }
        }
    }

    int& at(int x, int y) {
        if (!in_bounds(x, y)) {
            padding = 0;
            return padding;
        }
        return data.at(x + i, y + j);
    }

    bool operator==(const submatrix& other) const {
        if (dimension != other.dimension) return false;

        for (int x = 0; x < other.dimension; ++x) {
            for (int y = 0; y < other.dimension; ++y) {
                if (data.get(x, y) != other.data.get(x, y)) return false;
            }
        }

        return true;
    }
};

// Overloaded print operator
std::ostream& operator<<(std::ostream& os, submatrix m) {
    for (int x = 0; x < m.dimension; ++x) {
        for (int y = 0; y < m.dimension; ++y) {
            os << m.at(x, y) << " ";
        }
        os << "\n";
    }

    return os;
}

// Submatrix addition with data target
//  assumes that 'c' is cleared
void sum(submatrix a, submatrix b, submatrix c) {
    int dimension = c.dimension;

    for (int j = 0; j < dimension; ++j) {
        for (int i = 0; i < dimension; ++i) {
            c.at(i, j) = a.at(i, j) + b.at(i, j);
        }
    }
}

// Submatrix subtraction with data target
//  assumes that 'c' is cleared
void sub(submatrix a, submatrix b, submatrix c) {
    int dimension = c.dimension;

    for (int j = 0; j < dimension; ++j) {
        for (int i = 0; i < dimension; ++i) {
            c.at(i, j) = a.at(i, j) - b.at(i, j);
        }
    }
}

void linear_mul(submatrix a, submatrix b, submatrix c) {
    int dimension = c.dimension;

    for (int k = 0; k < dimension; ++k) {
        for (int i = 0; i < dimension; ++i) {
            int r = a.at(i, k);
            for (int j = 0; j < dimension; ++j) {
                c.at(i, j) += r * b.at(k, j);
            }
        }
    }
}

void strassen_mul(submatrix a, submatrix b, submatrix c, int cutoff) {
    int dimension = c.dimension;
    a.dimension = dimension;
    b.dimension = dimension;

    // Scratch space
    matrix_data scratch_space(dimension);

    std::function<void(submatrix, submatrix, submatrix, submatrix)>
        strassen_mul_recursion;
    strassen_mul_recursion = [&](submatrix A, submatrix B, submatrix C,
                                 submatrix S) {
        // Clear result
        C.clear();

        if (C.dimension % 2 == 1 || C.dimension <= cutoff) {
            linear_mul(A, B, C);
            return;
        }

        submatrix A00 = A.sub(0, 0);
        submatrix A01 = A.sub(0, 1);
        submatrix A10 = A.sub(1, 0);
        submatrix A11 = A.sub(1, 1);

        submatrix B00 = B.sub(0, 0);
        submatrix B01 = B.sub(0, 1);
        submatrix B10 = B.sub(1, 0);
        submatrix B11 = B.sub(1, 1);

        submatrix C00 = C.sub(0, 0);
        submatrix C01 = C.sub(0, 1);
        submatrix C10 = C.sub(1, 0);
        submatrix C11 = C.sub(1, 1);

        // Storage space for product
        submatrix M = S.sub(0, 0);

        // Recusrive scratch space
        submatrix SR = S.sub(0, 1);

        // Storage for sums
        submatrix sum0 = S.sub(1, 0);
        submatrix sum1 = S.sub(1, 1);

        // Calculate M1
        sum(A00, A11, sum0);
        sum(B00, B11, sum1);
        strassen_mul_recursion(sum0, sum1, M, SR);
        sum(C00, M, C00);
        sum(C11, M, C11);

        // Calculate M2
        sum(A10, A11, sum0);
        strassen_mul_recursion(sum0, B00, M, SR);
        sum(C10, M, C10);
        sub(C11, M, C11);

        // Calculate M3
        sub(B01, B11, sum0);
        strassen_mul_recursion(A00, sum0, M, SR);
        sum(C01, M, C01);
        sum(C11, M, C11);

        // Calculate M4
        sub(B10, B00, sum0);
        strassen_mul_recursion(A11, sum0, M, SR);
        sum(C00, M, C00);
        sum(C10, M, C10);

        // Calculate M5
        sum(A00, A01, sum0);
        strassen_mul_recursion(sum0, B11, M, SR);
        sub(C00, M, C00);
        sum(C01, M, C01);

        // Caclulate M6
        sub(A10, A00, sum0);
        sum(B00, B01, sum1);
        strassen_mul_recursion(sum0, sum1, M, SR);
        sum(C11, M, C11);

        // Calculate M7
        sub(A01, A11, sum0);
        sum(B10, B11, sum1);
        strassen_mul_recursion(sum0, sum1, M, SR);
        sum(C00, M, C00);
    };

    strassen_mul_recursion(a, b, c, scratch_space);
}

int main(int argc, const char** argv) {
    if (argc != 4) return usage();

    std::vector<std::string> args(argv + 1, argv + argc);

    // Parse input parameters
    int debug = to_int(args.at(0));
    int dimension = to_int(args.at(1));
    int cutoff = 32;

    if (dimension <= 0) return usage();

    // Allocate input matrices
    matrix_data a(dimension);
    matrix_data b(dimension);

    if ((debug & debug_flags::RANDOM) != 0) {
        // Randomly populate matrices instead of reading from file
        srand(time(NULL));
        for (int i = 0; i < dimension * dimension; ++i) {
            a.at(i) = rand() % 2;
            b.at(i) = rand() % 2;
        }
        cutoff = std::max(to_int(args.at(2)), 1);
    } else {
        // Read data from file
        std::string line;
        std::ifstream file(args.at(2));

        if (file.is_open()) {
            int i = 0;
            while (getline(file, line)) {
                if (i < dimension * dimension) {
                    a.at(i) = to_int(line);
                } else {
                    b.at(i - dimension * dimension) = to_int(line);
                }
                ++i;
            }
            file.close();
        } else {
            // Error handling
            std::cerr << "      Unable to open file: \"" << args.at(2) << "\""
                      << std::endl;
            return -1;
        }
    }

    matrix_data c_padded(padding_size(dimension, cutoff));
    submatrix c(c_padded);
    c.dimension = dimension;

    // Perform the multiplications
    auto task = [&]() { strassen_mul(a, b, c_padded, cutoff); };
    if ((debug & debug_flags::TIME) != 0) {
        std::cout << "strassen: ";
        time(task);
    } else {
        task();
    }

    if ((debug & debug_flags::PRINT) != 0) {
        std::cout << "A:\n" << a;
        std::cout << "B:\n" << b;
        std::cout << "C:\n" << c_padded;
    }

    if ((debug & debug_flags::VERIFY) != 0) {
        matrix_data check(dimension);

        auto task = [&]() { linear_mul(a, b, check); };
        if ((debug & debug_flags::TIME) != 0) {
            std::cout << "linear: ";
            time(task);
        } else {
            task();
        }

        if ((debug & debug_flags::PRINT) != 0) {
            std::cout << "check:\n" << check;
        }
        assert(submatrix(c) == submatrix(check));
    }

    // Print diagonal to standard output
    if (debug == 0) {
        for (int i = 0; i < dimension; ++i) {
            std::cout << c.at(i, i) << "\n";
        }
    }
}
