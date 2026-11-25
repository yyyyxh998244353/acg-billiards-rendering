#!/usr/bin/env python3
"""Generate reproducible pool layouts (SI units, XY = long/short).
Line 1 is the cue ball, then balls 1..15. Frame 0 is the untouched rack.
The contact rule is the frictionless, normal-restitution rigid body impulse
model (cf. Box2D's restitution discussion, https://box2d.org/documentation/md_simulation.html).
This is an independent disk solver, not Box2D: swept contacts are resolved
sequentially, including tied contacts in ball-index order. Level cloth has
constant rolling deceleration; there is no sliding/spin coupling, pocketing,
jumping, cushion deformation or cue-stick dynamics.
"""
import argparse
import json
import math
import random
from pathlib import Path

LENGTH, WIDTH = 2.54, 1.27
R = 0.028575
D = 2.0 * R
XMAX, YMAX = LENGTH / 2.0 - R, WIDTH / 2.0 - R
MASS = 0.17                   # kg per ball
BALL_E, RAIL_E = 0.96, 0.84
DRAG = 0.19                   # m/s^2, constant rolling deceleration
SHOT_SPEED = 5.8
SPEED_EPS, GEOM_TOL = 1e-10, 1e-8


class Ball:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx = self.vy = 0.0


def rack():
    balls = [None] * 16
    balls[0] = Ball(-LENGTH / 4.0, -0.08)
    # Eight ball in the center and unlike rear corners. Adjacent centers are D apart.
    rows = ((1,), (11, 4), (3, 8, 14), (9, 5, 13, 7), (2, 15, 6, 10, 12))
    for row, numbers in enumerate(rows):
        for col, number in enumerate(numbers):
            balls[number] = Ball(LENGTH / 4.0 + math.sqrt(3.0) * R * row, (2 * col - row) * R)
    return balls


def momentum(balls):
    return MASS * sum(b.vx for b in balls), MASS * sum(b.vy for b in balls)


def energy(balls):
    return 0.5 * MASS * sum(b.vx**2 + b.vy**2 for b in balls)


def penetration(balls):
    worst = 0.0
    for i, b in enumerate(balls):
        if not all(math.isfinite(a) for a in (b.x, b.y, b.vx, b.vy)):
            raise ValueError("nonfinite ball state")
        worst = max(worst, abs(b.x) - XMAX, abs(b.y) - YMAX)
        for j in range(i):
            a = balls[j]
            worst = max(worst, D - math.hypot(b.x - a.x, b.y - a.y))
    if worst > GEOM_TOL:
        raise ValueError("ball penetration exceeds tolerance: %.12g m" % worst)
    return worst


def next_hit(balls, horizon):
    """Earliest swept center-to-cushion or disk-to-disk contact in free flight."""
    t, hit = horizon, None
    for i, b in enumerate(balls):
        for axis, p, v, limit in ((-1, b.x, b.vx, XMAX), (-2, b.y, b.vy, YMAX)):
            if abs(v) > SPEED_EPS:
                tau = max(0.0, (math.copysign(limit, v) - p) / v)
                if tau < t:
                    t, hit = tau, (i, axis)
        for j in range(i):
            a = balls[j]
            x, y = b.x - a.x, b.y - a.y
            vx, vy = b.vx - a.vx, b.vy - a.vy
            approach = -(x * vx + y * vy)
            if approach <= 0.0:
                continue
            gap = x*x + y*y - D*D
            disc = approach**2 - (vx*vx + vy*vy) * gap
            if disc <= (SPEED_EPS * D)**2:  # miss or negligible normal velocity
                continue
            # Smaller quadratic root, rationalized to avoid cancellation at contact.
            tau = max(0.0, gap / (approach + math.sqrt(disc)))
            if tau < t:
                t, hit = tau, (i, j)
    return t, hit


def simulate(balls, seed, fps, frames):
    last_frame = max(frames)
    layouts = {0: snapshot(balls)}
    cue = balls[0]
    target_y = 0.012 + random.Random(seed).uniform(-0.003, 0.003)
    dx, dy = balls[1].x - cue.x, target_y - cue.y  # jitter aim by at most 3 mm
    scale = SHOT_SPEED / math.hypot(dx, dy)
    cue.vx, cue.vy = dx * scale, dy * scale
    p0x, p0y = momentum(balls)
    ext_x = ext_y = 0.0  # integrated cloth and cushion impulses, N s
    initial_e = prev_e = energy(balls)
    worst_pen = penetration(balls)
    max_dp = max_gain = 0.0
    pair_hits = rail_hits = 0
    # Equal masses and nonincreasing KE bound every speed by SHOT_SPEED.
    # Limit free travel per drag update to R/8; sweep even within those steps.
    steps_per_frame = math.ceil(SHOT_SPEED / (fps * R / 8.0))
    dt = 1.0 / (fps * steps_per_frame)
    steps = steps_per_frame * last_frame
    for step in range(1, steps + 1):
        for b in balls:
            speed = math.hypot(b.vx, b.vy)
            if speed:
                f = min(1.0, DRAG * dt / speed)
                dvx, dvy = -f * b.vx, -f * b.vy
                b.vx += dvx; b.vy += dvy
                ext_x += MASS * dvx; ext_y += MASS * dvy
        remaining = dt
        hits = 0
        while remaining > 0.0:
            tau, hit = next_hit(balls, remaining)
            for b in balls:
                b.x += b.vx * tau; b.y += b.vy * tau
            remaining -= tau
            if hit is None:
                break
            hits += 1
            if hits > 4096:
                raise RuntimeError("contact sequence did not converge")
            i, j = hit
            b = balls[i]
            if j == -1:
                ext_x -= MASS * (1.0 + RAIL_E) * b.vx
                b.vx *= -RAIL_E
                rail_hits += 1
            elif j == -2:
                ext_y -= MASS * (1.0 + RAIL_E) * b.vy
                b.vy *= -RAIL_E
                rail_hits += 1
            else:
                a = balls[j]
                x, y = b.x - a.x, b.y - a.y
                dist = math.hypot(x, y)
                nx, ny = x / dist, y / dist
                rel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
                kick = -0.5 * (1.0 + BALL_E) * rel
                b.vx += kick * nx; b.vy += kick * ny
                a.vx -= kick * nx; a.vy -= kick * ny
                pair_hits += 1
        worst_pen = max(worst_pen, penetration(balls))
        px, py = momentum(balls)
        max_dp = max(max_dp, math.hypot(px - p0x - ext_x, py - p0y - ext_y))
        e = energy(balls)
        max_gain = max(max_gain, e - prev_e)
        if max_dp > 1e-9 or max_gain > 1e-10 or e > initial_e + 1e-10:
            raise ValueError("impulse balance or dissipative energy bound violated")
        prev_e = e
        if step % steps_per_frame == 0 and step // steps_per_frame in frames:
            layouts[step // steps_per_frame] = snapshot(balls)
    if last_frame / fps >= 0.4 and pair_hits == 0:
        raise ValueError("cue never hit the rack")
    stats = dict(seed=seed, fps=fps, frames=sorted(frames), after_frame=last_frame,
                after_s=last_frame / fps, table_m=[LENGTH, WIDTH], radius_m=R, mass_kg=MASS,
                target_y_m=target_y,
                initial_cue_velocity_m_s=[p0x / MASS, p0y / MASS], rolling_deceleration_m_s2=DRAG,
                ball_restitution=BALL_E, rail_restitution=RAIL_E, substeps=steps,
                ball_impulses=pair_hits, cushion_impulses=rail_hits, post_strike_energy_J=initial_e,
                after_energy_J=prev_e, max_step_energy_gain_J=max_gain, external_impulse_Ns=[ext_x, ext_y],
                max_momentum_residual_Ns=max_dp, max_penetration_m=worst_pen)
    return layouts, stats


def snapshot(balls):
    return ''.join('%.12f %.12f\n' % (b.x, b.y) for b in balls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--seed', type=int, default=0, help='perturbs cue aim only (default: 0)')
    parser.add_argument('--fps', type=int, default=60, help='frame rate (default: 60)')
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 90], metavar='N',
                        help='frame indices to export; after.txt equals the last (default: 0 90)')
    args = parser.parse_args()
    if args.fps < 1 or min(args.frames) < 0:
        parser.error('fps must be positive and frames must be nonnegative')
    frames = set(args.frames)
    layouts, stats = simulate(rack(), args.seed, args.fps, frames)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for frame in sorted(frames):
        (out / ('frame_%06d.txt' % frame)).write_bytes(layouts[frame].encode('ascii'))
    (out / 'before.txt').write_bytes(layouts[0].encode('ascii'))
    (out / 'after.txt').write_bytes(layouts[max(frames)].encode('ascii'))
    print(json.dumps(stats, sort_keys=True))


if __name__ == '__main__':
    main()
