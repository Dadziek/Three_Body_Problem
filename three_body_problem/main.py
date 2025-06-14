import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def simulate_live(
    mass1=10, mass2=20, mass3=30,
    pos1_start=(-10.0, 10.0, -11.0), vel1_start=(-3.0, 0.0, 0.0),
    pos2_start=(0.0, 0.0, 0.0), vel2_start=(0.0, 0.0, 0.0),
    pos3_start=(10.0, 10.0, 12.0), vel3_start=(3.0, 0.0, 0.0),
    dt=0.01, steps=5000, g_const=9.8
):
    plt.style.use('dark_background')

    def compute_accelerations(r1, r2, r3, m1, m2, m3):
        def accel(p_a, p_b, m_b):
            r = p_a - p_b
            dist = np.linalg.norm(r)
            if dist == 0:
                return np.zeros(3)
            return -g_const * m_b * r / dist**3

        a1 = accel(r1, r2, m2) + accel(r1, r3, m3)
        a2 = accel(r2, r3, m3) + accel(r2, r1, m1)
        a3 = accel(r3, r1, m1) + accel(r3, r2, m2)
        return a1, a2, a3

    # Initial conditions
    r1, v1 = np.array(pos1_start, dtype=float), np.array(vel1_start, dtype=float)
    r2, v2 = np.array(pos2_start, dtype=float), np.array(vel2_start, dtype=float)
    r3, v3 = np.array(pos3_start, dtype=float), np.array(vel3_start, dtype=float)

    trajectory1 = [r1.copy()]
    trajectory2 = [r2.copy()]
    trajectory3 = [r3.copy()]

    # Plot setup
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_facecolor('black')
    ax.xaxis.set_pane_color((0, 0, 0, 1))
    ax.yaxis.set_pane_color((0, 0, 0, 1))
    ax.zaxis.set_pane_color((0, 0, 0, 1))

    line1, = ax.plot([], [], [], color='red', lw=0.5, alpha=0.7)
    line2, = ax.plot([], [], [], color='white', lw=0.5, alpha=0.7)
    line3, = ax.plot([], [], [], color='blue', lw=0.5, alpha=0.7)

    dot1, = ax.plot([], [], [], 'o', color='red')
    dot2, = ax.plot([], [], [], 'o', color='white')
    dot3, = ax.plot([], [], [], 'o', color='blue')

    def update(frame):
        nonlocal r1, r2, r3, v1, v2, v3
        a1, a2, a3 = compute_accelerations(r1, r2, r3, mass1, mass2, mass3)

        v1 += a1 * dt
        v2 += a2 * dt
        v3 += a3 * dt

        r1 += v1 * dt
        r2 += v2 * dt
        r3 += v3 * dt

        trajectory1.append(r1.copy())
        trajectory2.append(r2.copy())
        trajectory3.append(r3.copy())

        def update_line(traj, line, dot):
            xs, ys, zs = zip(*traj)
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            dot.set_data([xs[-1]], [ys[-1]])
            dot.set_3d_properties([zs[-1]])

        update_line(trajectory1, line1, dot1)
        update_line(trajectory2, line2, dot2)
        update_line(trajectory3, line3, dot3)

        return line1, line2, line3, dot1, dot2, dot3

    _ = FuncAnimation(fig, update, frames=steps, interval=1, blit=False)
    plt.show()


simulate_live()
