import json
import matplotlib.pyplot as plt

def plot_z_vs_time(data, title="z vs Time"):
    # Sort by time in case input is unordered
    data_sorted = sorted(data, key=lambda d: d["time"])

    times = [d["time"] for d in data_sorted]
    zs = [d["z"] for d in data_sorted]

    # Separate measured vs extrapolated for marker coloring
    times_measured = [d["time"] for d in data_sorted if d["label"] == "measured"]
    z_measured = [d["z"] for d in data_sorted if d["label"] == "measured"]

    times_extrap = [d["time"] for d in data_sorted if d["label"] == "extrapolated"]
    z_extrap = [d["z"] for d in data_sorted if d["label"] == "extrapolated"]

    plt.figure(figsize=(8, 4))

    # Base line through all points
    plt.plot(times, zs, color="gray", linewidth=1, alpha=0.7)

    # Measured points (blue)
    plt.scatter(times_measured, z_measured,
                color="blue", label="measured", zorder=3)

    # Extrapolated points (red)
    plt.scatter(times_extrap, z_extrap,
                color="red", label="extrapolated", zorder=3)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("z")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("z_vs_time.png", dpi=150)


if __name__ == "__main__":

    # Change this path to ball coordinates or sticker coordinates
    with open("sticker_coords.json", "r") as f:
        data = json.load(f)

    plot_z_vs_time(data, title="z vs Time (example)")
