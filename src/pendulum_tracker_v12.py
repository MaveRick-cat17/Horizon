import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks


INPUT = "horizon_pendulum_v11.npz"
OUTPUT = "horizon_pendulum_v12.npz"

# V11 pivot from the successful calibration
DEFAULT_PIVOT = (253.5, 59.0)


def find_array(data, names):
    """Find the first matching array from a list of likely NPZ key names."""
    for name in names:
        if name in data.files:
            arr = np.asarray(data[name])
            if arr.ndim == 1:
                return arr.astype(float), name
    return None, None


def main():
    print("=" * 70)
    print("HORIZON — V12 PHYSICS-CLEAN PENDULUM PROCESSOR")
    print("=" * 70)

    data = np.load(INPUT, allow_pickle=True)

    print("NPZ keys:", data.files)

    t, t_key = find_array(
        data,
        ["time", "times", "t", "timestamps"]
    )

    x, x_key = find_array(
        data,
        ["x", "x_positions", "x_pos", "bob_x", "xs"]
    )

    y, y_key = find_array(
        data,
        ["y", "y_positions", "y_pos", "bob_y", "ys"]
    )

    if x is None or y is None:
        raise RuntimeError(
            "Could not find x/y bob coordinates in horizon_pendulum_v11.npz.\n"
            f"Available keys: {data.files}"
        )

    n = min(len(x), len(y))

    x = x[:n]
    y = y[:n]

    if t is None:
        fps = 30.0
        t = np.arange(n) / fps
        print("Time array not found; using 30 FPS.")
    else:
        t = t[:n]

    # ------------------------------------------------------------
    # Pivot
    # ------------------------------------------------------------

    pivot = DEFAULT_PIVOT

    for key in ["pivot", "pivot_point", "PIVOT"]:
        if key in data.files:
            p = np.asarray(data[key]).flatten()
            if len(p) >= 2:
                pivot = (float(p[0]), float(p[1]))
                break

    px, py = pivot

    print(f"Using x array : {x_key}")
    print(f"Using y array : {y_key}")
    print(f"Pivot         : ({px:.1f}, {py:.1f})")
    print(f"Samples       : {n}")

    # ------------------------------------------------------------
    # 1. Basic finite-data mask
    # ------------------------------------------------------------

    finite = np.isfinite(x) & np.isfinite(y)

    if finite.sum() < 20:
        raise RuntimeError("Too few valid x/y samples.")

    # ------------------------------------------------------------
    # 2. Pendulum geometry
    #
    # A real pendulum bob should stay approximately on a circle
    # centred at the pivot.
    # ------------------------------------------------------------

    radius = np.full(n, np.nan)

    radius[finite] = np.hypot(
        x[finite] - px,
        y[finite] - py
    )

    median_radius = np.nanmedian(radius)

    # Allow ~10% radius variation. This is deliberately generous.
    geometry_ok = (
        finite
        & (np.abs(radius - median_radius) < 0.10 * median_radius)
    )

    print(f"Median pendulum radius : {median_radius:.2f}px")
    print(
        f"Geometry-valid samples : "
        f"{geometry_ok.sum()} / {n} "
        f"({100 * geometry_ok.mean():.1f}%)"
    )

    # ------------------------------------------------------------
    # 3. IMPORTANT FIX:
    #
    # Measure angle from the DOWNWARD vertical.
    #
    # theta = atan2(dx, dy)
    #
    # Therefore:
    #   bob directly below pivot -> 0 degrees
    #   bob right               -> +theta
    #   bob left                -> -theta
    #
    # This avoids the V11 +/-180 degree angle error.
    # ------------------------------------------------------------

    theta = np.full(n, np.nan)

    theta[geometry_ok] = np.degrees(
        np.arctan2(
            x[geometry_ok] - px,
            y[geometry_ok] - py
        )
    )

    # ------------------------------------------------------------
    # 4. Reject physically impossible frame-to-frame jumps.
    #
    # At 30 FPS a normal pendulum cannot suddenly jump tens of
    # degrees between adjacent frames.
    # ------------------------------------------------------------

    clean = geometry_ok.copy()

    ANGLE_JUMP_LIMIT = 10.0  # degrees/frame

    for i in range(1, n):
        if clean[i] and clean[i - 1]:
            jump = abs(theta[i] - theta[i - 1])

            if jump > ANGLE_JUMP_LIMIT:
                clean[i] = False
                theta[i] = np.nan

    print(
        f"After temporal validation: "
        f"{clean.sum()} / {n} samples"
    )

    # ------------------------------------------------------------
    # 5. Interpolate ONLY short gaps.
    #
    # Never invent long sections of motion.
    # ------------------------------------------------------------

    theta_clean = theta.copy()

    max_gap = 6

    i = 0

    while i < n:
        if np.isfinite(theta_clean[i]):
            i += 1
            continue

        start = i

        while i < n and not np.isfinite(theta_clean[i]):
            i += 1

        end = i - 1
        gap = end - start + 1

        left = start - 1
        right = i

        if (
            gap <= max_gap
            and left >= 0
            and right < n
            and np.isfinite(theta_clean[left])
            and np.isfinite(theta_clean[right])
        ):
            theta_clean[start:right] = np.interp(
                t[start:right],
                [t[left], t[right]],
                [theta_clean[left], theta_clean[right]]
            )

    # ------------------------------------------------------------
    # 6. Light smoothing for numerical differentiation.
    # ------------------------------------------------------------

    valid_clean = np.isfinite(theta_clean)

    theta_smooth = theta_clean.copy()

    count = valid_clean.sum()

    if count >= 11:
        vals = theta_clean[valid_clean]

        window = min(11, len(vals))

        if window % 2 == 0:
            window -= 1

        if window >= 5:
            theta_smooth[valid_clean] = savgol_filter(
                vals,
                window_length=window,
                polyorder=2
            )

    # ------------------------------------------------------------
    # 7. Basic physics diagnostics
    # ------------------------------------------------------------

    finite_smooth = np.isfinite(theta_smooth)

    amplitude = np.max(np.abs(theta_smooth[finite_smooth]))

    # Estimate period from positive peaks.
    peak_indices, _ = find_peaks(
        theta_smooth[finite_smooth],
        distance=10,
        prominence=2
    )

    valid_times = t[finite_smooth]
    peak_times = valid_times[peak_indices]

    if len(peak_times) >= 2:
        periods = np.diff(peak_times)
        period = np.median(periods)
        frequency = 1.0 / period
    else:
        period = np.nan
        frequency = np.nan

    print()
    print("PHYSICS CHECK")
    print("-" * 70)
    print(f"Maximum |theta| : {amplitude:.2f} deg")

    if np.isfinite(period):
        print(f"Estimated period: {period:.3f} s")
        print(f"Estimated freq. : {frequency:.3f} Hz")
    else:
        print("Period estimate : insufficient peaks")

    print()
    print("Expected result:")
    print("  ✓ theta should oscillate smoothly around 0°")
    print("  ✓ no repeated +180° / -180° jumps")
    print("  ✓ bob trajectory should form a circular arc")
    print("  ✓ short detection gaps may be interpolated")
    print()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    np.savez(
        OUTPUT,
        time=t,
        x=x,
        y=y,
        theta_raw=theta,
        theta=theta_smooth,
        radius=radius,
        valid=finite_smooth,
        pivot=np.array([px, py]),
        pendulum_length=median_radius,
        estimated_period=period,
        estimated_frequency=frequency,
    )

    print(f"Saved: {OUTPUT}")

    # ------------------------------------------------------------
    # Plot 1 — corrected angle
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 5))
    plt.plot(
        t,
        theta_smooth,
        label="Clean θ(t)"
    )
    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )
    plt.xlabel("Time (s)")
    plt.ylabel("θ (degrees)")
    plt.title("Horizon — V12 Corrected Pendulum Angle θ(t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "horizon_v12_angle.png",
        dpi=150
    )

    plt.show()

    # ------------------------------------------------------------
    # Plot 2 — spatial trajectory
    # ------------------------------------------------------------

    plt.figure(figsize=(10, 7))

    plt.plot(
        x[finite],
        y[finite],
        label="Tracked bob"
    )

    plt.scatter(
        [px],
        [py],
        s=80,
        label="Pivot"
    )

    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.title("Horizon — V12 Validated Pendulum Trajectory")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(
        "horizon_v12_trajectory.png",
        dpi=150
    )

    plt.show()

    print()
    print("=" * 70)
    print("V12 PROCESSING COMPLETE")
    print("=" * 70)
    print()
    print("DO NOT run SINDy yet.")
    print("First inspect the two V12 plots.")
    print()


if __name__ == "__main__":
    main()
