"""Generate expert labels for sampled states from student flights on stepped evaluation starts."""
import json, os, sys, time
import numpy as np
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_MODULE_PATHS = [os.path.join(REPO_ROOT, "common"), os.path.join(REPO_ROOT, "expert")]
FLAMBE_ROOT = os.environ.get("FLAMBE_ROOT", REPO_ROOT)
FLAMBE_GATE2 = os.environ.get("FLAMBE_GATE2", os.path.join(FLAMBE_ROOT, "research", "02_experiments", "gate2"))
FLAMBE_OUTPUT = os.environ.get("FLAMBE_OUTPUT", os.path.join(FLAMBE_ROOT, "out"))
FLAMBE_POLICY = os.environ.get("FLAMBE_POLICY", os.path.join(FLAMBE_ROOT, "policy"))

sys.path[:0] = [*LOCAL_MODULE_PATHS, FLAMBE_POLICY]; import lead_g2_dagger as g2d

YARDS = ["syn_002", "syn_016", "syn_018", "syn_040", "syn_001", "syn_006", "syn_007", "syn_008"]
UNSEEN = {"syn_001", "syn_006", "syn_007", "syn_008"}


def main():
    M, NP = sys.argv[1], int(sys.argv[2]); S = int(sys.argv[3]) if len(sys.argv) > 3 else 10; MX = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    src = f"{FLAMBE_GATE2}/stepped_v2b/eval_{M}"; OUT = os.path.join(FLAMBE_OUTPUT, M); TD = f"{OUT}/strided"; PD = f"{OUT}/parts"
    os.makedirs(PD, exist_ok=True); t0 = time.time(); rows, Xs, Ys, Yd, IF = [], [], [], [], []
    print(f"[relabel_stepped v1] model {M} stride {S} NP {NP} max flights/yard {MX or 'all'}", flush=True)
    for y in YARDS:
        st = json.load(open(f"{FLAMBE_GATE2}/stepped_v2b/final/stepped2_{y}.json"))["starts"]
        st = [s for s in st if os.path.exists(f"{src}/tracks_{y}/{s['id']}.npz")]
        if MX: st = st[:MX]
        os.makedirs(f"{TD}/{y}", exist_ok=True)
        for s in st:
            z = np.load(f"{src}/tracks_{y}/{s['id']}.npz"); X, clr = z["X"], z["clr"][:len(z["X"])]
            np.savez(f"{TD}/{y}/{s['id']}.npz", X=X[::S], clr=clr[::S])
        todo = [s for s in st if not os.path.exists(f"{PD}/{y}__{s['id']}.npz")]
        if todo:
            with ProcessPoolExecutor(min(NP, len(todo)), mp_context=get_context("spawn"), initializer=g2d.rl_winit, initargs=(y,)) as p:
                for r in p.map(g2d.relabel_one, [(TD, y, s) for s in todo], chunksize=1):
                    Xstr = np.load(f"{TD}/{y}/{r['id']}.npz")["X"]
                    row = {k: r[k] for k in ("id", "yard", "visited", "dropped_clearance", "dropped_no_plan", "errors")} | {"kept": int(len(r["idx"])), "unseen": y in UNSEEN}
                    np.savez(f"{PD}/{y}__{r['id']}.tmp.npz", X=Xstr[r["idx"]].astype(np.float32), Y=r["Y"], I=(r["idx"] * S).astype(np.int32), R=json.dumps(row, default=int))
                    os.replace(f"{PD}/{y}__{r['id']}.tmp.npz", f"{PD}/{y}__{r['id']}.npz")
        for s in st:
            with np.load(f"{PD}/{y}__{s['id']}.npz") as z:
                Xs.append(z["X"]); Ys.append(z["Y"]); IF.append(z["I"]); rows.append(json.loads(str(z["R"])))
            Yd += [y] * len(Xs[-1])
        print(f"[relabel_stepped v1] {y}: {len(st)} flights, kept {sum(r['kept'] for r in rows if r['yard'] == y)} states ({time.time() - t0:.0f} s)", flush=True)
    X = np.concatenate(Xs).astype(np.float32); Y = np.concatenate(Ys).astype(np.float32)
    fl = np.concatenate([np.full(len(a), i, np.int32) for i, a in enumerate(Xs)])
    np.savez_compressed(f"{OUT}/dag_own.tmp.npz", X=X, Y=Y, yard=np.array(Yd), flight=fl, idx_full=np.concatenate(IF))
    os.replace(f"{OUT}/dag_own.tmp.npz", f"{OUT}/dag_own.npz")
    log = {"model": M, "stride": S, "flights": len(rows), "visited_strided": sum(r["visited"] for r in rows), "kept": int(len(X)),
           "dropped_clearance_lt_0.30": sum(r["dropped_clearance"] for r in rows), "dropped_no_plan": sum(r["dropped_no_plan"] for r in rows),
           "wall_s": round(time.time() - t0), "per_flight": rows}
    json.dump(log, open(f"{OUT}/dag_own.json", "w"))
    print("RELABEL_STEPPED_DONE", json.dumps({k: v for k, v in log.items() if k != "per_flight"}), flush=True)


if __name__ == "__main__":
    main()
