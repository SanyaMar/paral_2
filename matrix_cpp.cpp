#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

vector<vector<int>> generate_matrix(int rows, int cols) {
	if (rows <= 0 || cols <= 0) {
		throw runtime_error("Size of the matrix must be positive");
	}

	vector<vector<int>> matrix(rows, vector<int>(cols));
	random_device rd;
	mt19937 generator(rd());
	uniform_int_distribution<> distrib(-1000, 1000);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			matrix[i][j] = distrib(generator);
		}
	}

	return matrix;
}

void save_matrix_to_file(const vector<vector<int>> &matrix, const string &filename) {
	ofstream outFile(filename);
	if (!outFile) {
		throw runtime_error("Failed to open file for writing: " + filename);
	}

	for (const auto &row : matrix) {
		for (const auto &elem : row) {
			outFile << elem << " ";
		}
		outFile << "\n";
	}

	outFile.close();
}

vector<vector<int>> read_matrix_from_file(const string &filename, int rows, int cols) {
	vector<vector<int>> matrix(rows, vector<int>(cols));
	ifstream inFile(filename);
	if (!inFile) {
		throw runtime_error("Failed to open file for reading: " + filename);
	}
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			inFile >> matrix[i][j];
		}
	}
	inFile.close();
	return matrix;
}

vector<vector<int>> multiply_matrices(const vector<vector<int>> &A, const vector<vector<int>> &B, int NUM_THREADS) {
	int rowsA = A.size(), colsA = A[0].size();
	int rowsB = B.size(), colsB = B[0].size();

	if (colsA != rowsB) {
		throw runtime_error("Matrix dimensions do not match for multiplication");
	}

	vector<vector<int>> C(rowsA, vector<int>(colsB, 0));
	omp_set_num_threads(NUM_THREADS);

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < rowsA; ++i) {
		for (int j = 0; j < colsB; ++j) {
			int sum = 0;
			for (int k = 0; k < colsA; ++k) {
				sum += A[i][k] * B[k][j];
			}
			C[i][j] = sum;
		}
	}

	return C;
}


void run_for_threads(int num_threads) {
	string base_folder = to_string(num_threads) + "_threads";
	string folderC = base_folder + "/result_matrix";
	string report_dir = "reports";
	string report_file = report_dir + "/report_" + to_string(num_threads) + ".txt";

	fs::create_directories(folderC);
	fs::create_directories(report_dir);

	ofstream report(report_file);
	if (!report) {
		cerr << "Error opening " << report_file << " for writing" << endl;
		return;
	}
	report << "Matrix Size | Avg Execution Time (ms)\n";

	vector<int> sizes = {10, 50, 100, 500, 1000, 1500,2000};
	int trials = 5;

	for (int size : sizes) {
		try {
			cout << "[" << num_threads << " threads] Processing size: " << size << "x" << size << "...\n";
			long long total_time = 0;

			for (int t = 1; t <= trials; ++t) {
				string filenameA = "generated_matrices/matrixA" + to_string(size) + "_trial" + to_string(t) + ".txt";
				string filenameB = "generated_matrices/matrixB" + to_string(size) + "_trial" + to_string(t) + ".txt";
				string filenameC = folderC + "/result_matrix" + to_string(size) + "_trial" + to_string(t) + ".txt";

				vector<vector<int>> A = read_matrix_from_file(filenameA, size, size);
				vector<vector<int>> B = read_matrix_from_file(filenameB, size, size);

				auto start = high_resolution_clock::now();
				vector<vector<int>> C = multiply_matrices(A, B, num_threads);
				auto finish = high_resolution_clock::now();

				total_time += duration_cast<milliseconds>(finish - start).count();
				save_matrix_to_file(C, filenameC);
			}

			double average_time = static_cast<double>(total_time) / trials;
			report << setw(6) << size << " | " << setw(8) << fixed << setprecision(2) << average_time << "\n";

		} catch (const exception &e) {
			cerr << "Error: " << e.what() << endl;
		}
	}

	report.close();
}

int main() {
	vector<int> sizes = {10, 50, 100, 500, 1000, 1500, 2000};
	int trials = 5;
	fs::create_directories("generated_matrices");

	// Сначала сгенерировать все матрицы
	for (int size : sizes) {
		for (int t = 1; t <= trials; ++t) {
			vector<vector<int>> A = generate_matrix(size, size);
			vector<vector<int>> B = generate_matrix(size, size);
			string filenameA = "generated_matrices/matrixA" + to_string(size) + "_trial" + to_string(t) + ".txt";
			string filenameB = "generated_matrices/matrixB" + to_string(size) + "_trial" + to_string(t) + ".txt";
			save_matrix_to_file(A, filenameA);
			save_matrix_to_file(B, filenameB);
		}
	}


	for (int threads : {1, 2, 5, 10}) {
		run_for_threads(threads);
	}

	cout << "All processing completed.\n";
	return 0;
}
