import os
import numpy as np


def read_matrix(filename, rows, cols):
    with open(filename, 'r') as f:
        data = [list(map(int, line.strip().split())) for line in f.readlines()]
        if len(data) != rows or any(len(row) != cols for row in data):
            raise ValueError(f"Matrix size mismatch in file {filename}")
        return np.array(data)


def verify_result(A, B, C, tol=1e-6):
    computed = A @ B
    return np.allclose(computed, C, atol=tol), computed


def main():
    sizes = [10, 50, 100, 500, 1000, 1500, 2000]
    trials = 5
    thread_configs = [1, 2, 5, 10]
    report_lines = []

    for threads in thread_configs:
        report_lines.append(
            f"\n==== Verification for {threads} threads ====\n")
        result_dir = f"{threads}_threads/result_matrix"

        for size in sizes:
            for t in range(1, trials + 1):
                filenameA = f"generated_matrices/matrixA{size}_trial{t}.txt"
                filenameB = f"generated_matrices/matrixB{size}_trial{t}.txt"
                filenameC = f"{result_dir}/result_matrix{size}_trial{t}.txt"

                try:
                    A = read_matrix(filenameA, size, size)
                    B = read_matrix(filenameB, size, size)
                    C = read_matrix(filenameC, size, size)

                    match, computed = verify_result(A, B, C)

                    if match:
                        report_lines.append(f"[OK] size {size} trial {t}")
                    else:
                        report_lines.append(
                            f"[FAIL] size {size} trial {t} — mismatch detected")
                        diff = np.abs(computed - C)
                        max_diff = np.max(diff)
                        report_lines.append(
                            f"        Max difference: {max_diff}")

                except Exception as e:
                    report_lines.append(
                        f"[ERROR] size {size} trial {t} — {str(e)}")

    with open("verification_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print("Verification completed.")


if __name__ == "__main__":
    main()
