import matplotlib.pyplot as plt


def load_data(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    idx = 0
    num_pts = int(lines[idx])
    idx += 1
    points = []
    for _ in range(num_pts):
        x, y = map(float, lines[idx].split())
        points.append((x, y))
        idx += 1

    num_edges = int(lines[idx])
    idx += 1
    edges = []
    for _ in range(num_edges):
        a, b = map(int, lines[idx].split())
        edges.append((a, b))
        idx += 1

    return points, edges


def plot(points, edges):
    xs, ys = zip(*points)
    plt.scatter(xs, ys, c='blue', label='Points')

    for i, (x, y) in enumerate(points):
        plt.text(x + 0.1, y + 0.1, str(i), fontsize=9)

    for a, b in edges:
        x1, y1 = points[a]
        x2, y2 = points[b]
        plt.plot([x1, x2], [y1, y2], 'r-')

    plt.title("Multifragment Heuristic Output")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pts, eds = load_data("cmake-build-debug/output.txt")
    plot(pts, eds)
