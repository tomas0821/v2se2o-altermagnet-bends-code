"""S5 disorder (2026-11-14). Runs with memory safeguards (below).

Model: Cheng Table 1 KV2Se2O, E_F = -0.09 eV, orbital-resolved single-orbital Kwant systems (d_xz and d_xy decouple;
diagonal disorder keeps them decoupled). Anderson disorder: on-site, uniform in
[-U/2, U/2], one random number per site shared by both orbitals of that site; spin up (sublattice A) and spin down
(sublattice B) independent (salt contains the spin), or correlated (same salt -> same potential on the A and B site
of a cell; K-vacancy-like). Disorder only in the scattering region; leads clean.

Parts
  calib : straight ribbons, W = 20, 38; U = 0.25, 0.30, 0.35; L = 20, 40, 80, 160, 320 sites; 24 seeds;
          per spin sector: l from <T> = N0 l/(L + l), xi from the slope of <ln T> over L >= 80.
  ens   : diffusive-window ensembles, U = 0.25, 0.30, 0.35; straight / L / T with arms Ls = W; W = 20, 38;
          200 seeds; independent A/B disorder.
  corr  : correlated A/B disorder, U = 0.30, W = 20, all three geometries, 200 seeds.
  loc   : localised contrast, U = 1.2 eV, W = 20, three geometries, 100 seeds (reported via <ln T>).
  clean : U = 0 references.
Labelling rule (stated before the run): an ensemble is "diffusive" if the slow sector has <T> >= 1 and its device
length is below the calibrated xi_slow; otherwise it is reported as localised.

Memory safeguards (run on a machine capped at 6 GB):
  - 2 workers, nice 19, OMP_NUM_THREADS = 1, workers recycled every 4 tasks;
  - RLIMIT_AS = 1.2 GB per worker (measured peak: 0.52 GB virtual, 0.15 GB resident for the largest system);
  - watchdog on MemAvailable: pause workers below 700 MB, resume above 1000 MB, stop cleanly below 400 MB;
  - every finished task is appended to data/s5_results.jsonl at once; a rerun skips finished tasks (resume).
Run (project root): OMP_NUM_THREADS=1 nice -n 19 ~/venvs/crn-te/bin/python calc/s5_disorder.py
"""
import json, os, resource, signal, sys, threading, time, warnings
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import v2x2o_tb as tb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/s5_results.jsonl"
LOG = ROOT / "data/s5_run.log"
E_F = -0.09
NWORK = 2
AS_LIMIT = int(1.2 * 1024 ** 3)
PAUSE_MB, RESUME_MB, STOP_MB = 700, 1000, 400


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def mem_available_mb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 1024
    return 1e9


# ---------------------------------------------------------------- tasks
def tasks():
    T = []
    for W in (20, 38):
        for U in (0.25, 0.30, 0.35):
            for L in (20, 40, 80, 160, 320):
                for sp in ("up", "dn"):
                    for o in ("xz", "xy"):
                        T.append(dict(part="calib", geom="straight", W=W, Ls=L, U=U, spin=sp, orb=o, n=24, mode="ind"))
    for U in (0.25, 0.30, 0.35):
        for W in (20, 38):
            for g in ("straight", "L", "T"):
                for sp in ("up", "dn"):
                    for o in ("xz", "xy"):
                        T.append(dict(part="ens", geom=g, W=W, Ls=W, U=U, spin=sp, orb=o, n=200, mode="ind"))
    for g in ("straight", "L", "T"):
        for sp in ("up", "dn"):
            for o in ("xz", "xy"):
                T.append(dict(part="corr", geom=g, W=20, Ls=20, U=0.30, spin=sp, orb=o, n=200, mode="corr"))
                T.append(dict(part="loc", geom=g, W=20, Ls=20, U=1.2, spin=sp, orb=o, n=100, mode="ind"))
    for W in (20, 38):
        for g in ("straight", "L", "T"):
            for sp in ("up", "dn"):
                for o in ("xz", "xy"):
                    T.append(dict(part="clean", geom=g, W=W, Ls=W, U=0.0, spin=sp, orb=o, n=1, mode="ind"))
    for t in T:
        t["key"] = "|".join(str(t[k]) for k in ("part", "geom", "W", "Ls", "U", "spin", "orb", "mode"))
    return T


# ---------------------------------------------------------------- worker side
def init_worker():
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def build(geom, p, W, Ls):
    import kwant
    from kwant.digest import uniform
    lat = kwant.lattice.square(a=1, norbs=1)

    def onsite(site, U, salt):
        return p[0] + U * (uniform(repr(site.tag), salt) - 0.5) if U > 0 else p[0]
    if geom == "straight":
        sites = [(x, y) for x in range(-Ls, W + Ls) for y in range(W)] if Ls != 0 else []
    elif geom == "L":
        sites = [(x, y) for x in range(-Ls, W) for y in range(W)] + [(x, y) for x in range(W) for y in range(W, W + Ls)]
    else:
        sites = ([(x, y) for x in range(-Ls, W) for y in range(W)] + [(x, y) for x in range(W) for y in range(W, W + Ls)]
                 + [(x, y) for x in range(W) for y in range(-Ls, 0)])
    b = kwant.Builder()
    for s in sites:
        b[lat(*s)] = onsite
    for d, t in tb._hops(p).items():
        if t != 0.0:
            b[kwant.builder.HoppingKind(d, lat)] = t

    def lead(direction, cells):
        ld = kwant.Builder(kwant.TranslationalSymmetry(direction)); tb._fill(ld, lat, p, cells); return ld
    b.attach_lead(lead((-2, 0), [(x, y) for x in range(2) for y in range(W)]))
    if geom == "straight":
        b.attach_lead(lead((2, 0), [(x, y) for x in range(2) for y in range(W)]))
    else:
        b.attach_lead(lead((0, 2), [(x, y) for x in range(W) for y in range(2)]))
        if geom == "T":
            b.attach_lead(lead((0, -2), [(x, y) for x in range(W) for y in range(2)]))
    return b.finalized()


def run_task(t):
    import kwant
    t0 = time.time()
    p = dict(tb.orbital_params(tb.MODELS["cheng"], t["spin"]))[t["orb"]]
    # calibration: Ls is the total disordered length L; straight ribbon of length L = 2 Ls' + W -> use Ls' directly
    if t["part"] == "calib":
        f = build_straight_len(p, t["W"], t["Ls"])      # disordered length exactly Ls sites
    else:
        f = build(t["geom"], p, t["W"], t["Ls"])
    T1, T2, N = [], [], None
    for s in range(t["n"]):
        salt = f"c{s}" if t["mode"] == "corr" else f"{t['spin']}{s}"
        salt = f"{t['part']}|{t['U']}|{t['W']}|{t['Ls']}|{salt}"
        sm = kwant.smatrix(f, E_F, params=dict(U=t["U"], salt=salt))
        T1.append(float(sm.transmission(1, 0)))
        if t["geom"] == "T" and t["part"] != "calib":
            T2.append(float(sm.transmission(2, 0)))
        if N is None:
            N = [int(sm.num_propagating(i)) for i in range(len(f.leads))]
    return dict(key=t["key"], task={k: v for k, v in t.items() if k != "key"}, N=N, T1=T1, T2=T2,
                seconds=round(time.time() - t0, 2))


def build_straight_len(p, W, L):
    """straight ribbon whose disordered region is exactly L sites long."""
    import kwant
    from kwant.digest import uniform
    lat = kwant.lattice.square(a=1, norbs=1)

    def onsite(site, U, salt):
        return p[0] + U * (uniform(repr(site.tag), salt) - 0.5) if U > 0 else p[0]
    b = kwant.Builder()
    for x in range(L):
        for y in range(W):
            b[lat(x, y)] = onsite
    for d, t in tb._hops(p).items():
        if t != 0.0:
            b[kwant.builder.HoppingKind(d, lat)] = t
    for dirn in ((-2, 0), (2, 0)):
        ld = kwant.Builder(kwant.TranslationalSymmetry(dirn)); tb._fill(ld, lat, p, [(x, y) for x in range(2) for y in range(W)])
        b.attach_lead(ld)
    return b.finalized()


# ---------------------------------------------------------------- driver with watchdog
def main():
    os.nice(19)
    done = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                done.add(json.loads(l)["key"])
            except Exception:
                pass
    todo = [t for t in tasks() if t["key"] not in done]
    # heavy tasks first, but keep calibration early
    todo.sort(key=lambda t: (t["part"] != "clean", -(t["W"] * (t["Ls"] if t["part"] == "calib" else 3 * t["Ls"] + t["W"]) * t["n"])))
    log(f"start: {len(done)} tasks already done, {len(todo)} to run, MemAvailable {mem_available_mb():.0f} MB")
    if not todo:
        return 0
    if mem_available_mb() < RESUME_MB:
        log(f"not starting: MemAvailable below {RESUME_MB} MB")
        return 2
    pool = Pool(NWORK, initializer=init_worker, maxtasksperchild=4)
    state = {"paused": False, "stop": False}

    def watchdog():
        while not state["stop"]:
            m = mem_available_mb()
            pids = [w.pid for w in pool._pool if w.is_alive()]
            if m < STOP_MB:
                log(f"WATCHDOG: MemAvailable {m:.0f} MB < {STOP_MB} MB -> stopping the pool (finished tasks are saved)")
                state["stop"] = True
                for pid in pids:
                    try: os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError: pass
                pool.terminate()
                return
            if m < PAUSE_MB and not state["paused"]:
                log(f"WATCHDOG: MemAvailable {m:.0f} MB < {PAUSE_MB} MB -> pausing workers {pids}")
                state["paused"] = True
            if state["paused"] and m <= RESUME_MB:
                for pid in pids:              # idempotent; also catches a freshly recycled worker
                    try: os.kill(pid, signal.SIGSTOP)
                    except ProcessLookupError: pass
            elif m > RESUME_MB and state["paused"]:
                log(f"WATCHDOG: MemAvailable {m:.0f} MB > {RESUME_MB} MB -> resuming workers")
                for pid in pids:
                    try: os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError: pass
                state["paused"] = False
            time.sleep(5)

    th = threading.Thread(target=watchdog, daemon=True); th.start()
    n_ok = 0
    t_start = time.time()
    from multiprocessing import TimeoutError as MPTimeout
    try:
        it = pool.imap_unordered(run_task, todo)
        while True:
            try:
                res = it.next(timeout=10)
            except MPTimeout:
                if state["stop"]:
                    break
                continue
            except StopIteration:
                break
            with open(OUT, "a") as f:
                f.write(json.dumps(res) + "\n")
            n_ok += 1
            if n_ok % 10 == 0 or n_ok == len(todo):
                log(f"{n_ok}/{len(todo)} tasks done ({time.time() - t_start:.0f} s), MemAvailable {mem_available_mb():.0f} MB")
    except Exception as e:
        log(f"pool ended early: {type(e).__name__}: {e}")
        state["failed"] = True
    finally:
        stopped = state["stop"]
        state["stop"] = True
        if stopped or state.get("failed"):
            pool.terminate()          # never drain queued work whose results cannot be saved
        else:
            pool.close(); pool.join()
    log(f"end: {n_ok} new tasks written; rerun the same command to resume anything left")
    return 3 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
