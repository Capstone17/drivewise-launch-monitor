import json
import matplotlib.pyplot as plt

def plot_z_vs_time(data, title="z vs Time", outfile=None):
    # Sort by time
    data_sorted = sorted(data, key=lambda d: d["time"])

    times = [d["time"] for d in data_sorted]
    zs = [d["z"] for d in data_sorted]

    # Default missing labels to "measured"
    for d in data_sorted:
        if "label" not in d:
            d["label"] = "measured"

    times_measured = [d["time"] for d in data_sorted if d["label"] == "measured"]
    z_measured = [d["z"] for d in data_sorted if d["label"] == "measured"]

    times_extrap = [d["time"] for d in data_sorted if d["label"] == "extrapolated"]
    z_extrap = [d["z"] for d in data_sorted if d["label"] == "extrapolated"]

    plt.figure(figsize=(8, 4))

    # Line through all points
    plt.plot(times, zs, color="gray", linewidth=1, alpha=0.7)

    # Measured points (always present)
    plt.scatter(times_measured, z_measured,
                color="blue", label="measured", zorder=3)

    # Extrapolated points (only if any exist)
    if times_extrap:
        plt.scatter(times_extrap, z_extrap,
                    color="red", label="extrapolated", zorder=3)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("z")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if outfile is None:
        # For headless environments, prefer saving to file
        plt.savefig("z_vs_time.png", dpi=150)
    else:
        plt.savefig(outfile, dpi=150)

    plt.close()



if __name__ == "__main__":

    # Change this path to ball coordinates or sticker coordinates
    with open("sticker_coords.json", "r") as f:
        data = json.load(f)

    plot_z_vs_time(data, title="z vs Time (example)")
