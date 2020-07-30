# HIGHER LEVLE PLANNER
import numpy as np


class Planner:
    def __init__(self):
        pass

    def create_parabola(self, start, mid_height, end, t):
        def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
            denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
            A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
            B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
            C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
            return A, B, C

        sx = start[0]
        sy = start[1]
        sz = start[2]
        ex = end[0]
        ey = end[1]
        ez = end[2]
        mx = (ex - sx) / 2
        my = (ey - sy) / 2
        mz = mid_height
        ts = t[0]
        tm = t[1]
        te = t[2]

        ax, bx, cx = calc_parabola_vertex(ts, sx, tm, mx, te, ex)
        ay, by, cy = calc_parabola_vertex(ts, sy, tm, my, te, ey)
        az, bz, cz = calc_parabola_vertex(ts, sz, tm, mz, te, ez)
        return [ax, bx, cx], [ay, by, cy], [az, bz, cz]

    def create_parabola_points(self, n, px, py, pz):
        rx = []
        ry = []
        rz = []
        for ti in np.linspace(0, n, n):
            rx.append(self.solve_parabola(ti, px))
            ry.append(self.solve_parabola(ti, py))
            rz.append(self.solve_parabola(ti, pz))
        return rx, ry, rz

    def solve_parabola(self, theta, p):
        return p[0] * theta ** 2 + p[1] * theta + p[2]
