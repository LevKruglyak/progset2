#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

enum debug_flags {
    RANDOM = 0x01,  // should generate random instead of reading from a file
    PRINT = 0x02,   // print matrices to the screen
    TIME = 0x04,    // display the time
    VERIFY = 0x08,  // verify the matrix multiplication
};

static int usage() {
    std::cerr << "      Usage: ./strassen [DEBUG] [DIMENSION] [INPUT]\n";
    std::cerr << "          debug flags:\n";
    std::cerr << "              RANDOM: " << debug_flags::RANDOM << "\n";
    std::cerr << "              PRINT: " << debug_flags::PRINT << "\n";
    std::cerr << "              TIME: " << debug_flags::TIME << "\n";
    std::cerr << "              VERIFY: " << debug_flags::VERIFY << "\n";

    return -1;
}

static int to_int(std::string str) {
    int res;
    std::stringstream ss;

    ss << str;
    ss >> res;

    return res;
}

// Represents a two dimension set of integers
class matrix_data {
   public:
    int dimension;
    std::shared_ptr<std::vector<int>> data;

    matrix_data(int dimension) : dimension(dimension) {
        data = std::make_shared<std::vector<int>>(dimension * dimension, 0);
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

    int& at(int x, int y) {
        if (x < 0 || x >= dimension || y < 0 || y >= dimension) {
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

// Submatrix addition with data target
//  assumes that 'c' is cleared
void sum(submatrix a, submatrix b, submatrix c) {
    int dimension = std::min(a.dimension, b.dimension);
    assert(dimension <= c.dimension);

    for (int j = 0; j < dimension; ++j) {
        for (int i = 0; i < dimension; ++i) {
            c.at(i, j) = a.at(i, j) + b.at(i, j);
        }
    }
}

void linear_mul(submatrix a, submatrix b, submatrix c) {
    int dimension = std::max(a.dimension, b.dimension);
    assert(dimension <= c.dimension);

    for (int k = 0; k < dimension; ++k) {
        for (int i = 0; i < dimension; ++i) {
            int r = a.at(i, k);
            for (int j = 0; j < dimension; ++j) {
                c.at(i, j) += r * b.at(k, j);
            }
        }
    }
}

// Overloaded print operator
std::ostream& operator<<(std::ostream& os, submatrix m) {
    for (int x = 0; x < m.dimension; ++x) {
        for (int y = 0; y < m.dimension; ++y) {
            os << m.at(x + m.i, y + m.j) << " ";
        }
        os << "\n";
    }

    return os;
}

int main(int argc, const char** argv) {
    if (argc != 4) return usage();

    std::vector<std::string> args(argv + 1, argv + argc);

    // Parse input parameters
    int debug = to_int(args.at(0));
    int dimension = to_int(args.at(1));
    int cutoff = 1;

    if (dimension <= 0) return usage();

    // Allocate input matrices
    matrix_data a(dimension);
    matrix_data b(dimension);
    matrix_data c(dimension);

    if ((debug & debug_flags::RANDOM) != 0) {
        // Randomly populate matrices instead of reading from file
        srand(time(NULL));
        for (int i = 0; i < dimension * dimension; ++i) {
            a.at(i) = rand() % 2;
            b.at(i) = rand() % 2;
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

    // Perform the multiplications
    linear_mul(a, b, c);

    if ((debug & debug_flags::PRINT) != 0) {
        std::cout << "A:\n" << a;
        std::cout << "B:\n" << b;
        std::cout << "C:\n" << c;
    }

    if ((debug & debug_flags::VERIFY) != 0) {
        matrix_data check(dimension);
        linear_mul(a, b, check);
        assert(submatrix(c) == submatrix(check));
    }
}
