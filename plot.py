import matplotlib.pyplot as plt


def parse_report(filename):
    sizes = []
    times = []
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) != 2:
                continue
            size = int(parts[0].strip())
            time_ms = float(parts[1].strip())
            time_sec = time_ms / 1000
            sizes.append(size)
            times.append(time_sec)
    return sizes, times


def main():
    plt.figure(figsize=(10, 6))

    thread_counts = [1, 2, 5, 10]
    for threads in thread_counts:
        filename = f"reports/report_{threads}.txt"
        try:
            sizes, times = parse_report(filename)
            plt.plot(sizes, times, marker='o', label=f"{threads} threads")
        except FileNotFoundError:
            print(f"Report not found: {filename}")

    plt.title(
        "Зависимость среднего времени выполнения перемножения матриц от размера")
    plt.xlabel("Размер матрицы (N x N)")
    plt.ylabel("Среднее время (сек)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("image.png")
    plt.show()


if __name__ == "__main__":
    main()
