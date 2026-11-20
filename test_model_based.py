"""Differential test: InventoryService vs. a deliberately naive reference model.

Heap validation, precisely:
  * live expiry entries and purge entries: exact (deadline, id) match with the model
  * EVERY expiry entry, live or stale: deadline equals the one recorded when the
    reservation was created, and a stale entry's reservation is non-ACTIVE or purged
  * stale count: svc._stale equals the number of such stale entries
  * NOT checked: *which* stale entries exist at a given moment or when compaction
    runs. That is an implementation detail; the model has no heap by design.
"""
import json, os, random, sys, unittest
from datetime import datetime, timedelta, timezone
from inventory_service import *


class Ref:
    """O(n), no heaps, no tricks. Same documented semantics."""
    def __init__(self, retention):
        self.retention = retention
        self.avail, self.res, self.idem = {}, {}, {}   # res: id -> dict
        self.n = 0

    def sweep(self, now):
        out = []
        for rid, r in list(self.res.items()):
            if r["st"] == "ACTIVE" and r["exp"] <= now:
                r["st"], r["term"] = "EXPIRED", now
                self.avail[r["p"]] += r["q"]
                out.append(rid)
        if self.retention is not None:
            for rid, r in list(self.res.items()):
                if r["st"] != "ACTIVE" and r["term"] + timedelta(seconds=self.retention) <= now:
                    del self.res[rid]
                    self.idem.pop((r["c"], r["k"]), None)
        return out


def run(seed, ops=400):
    trace = []
    try:
        _run_inner(seed, ops, trace)
    except AssertionError as e:
        raise AssertionError(f"{e}\n--- reproduce: python3 -c 'import test_model_based as t; t.run({seed})' ---\n"
                             f"--- last operations (seed={seed}) ---\n" + "\n".join(trace[-25:])) from None
    except Exception as e:   # service crashed (KeyError etc.): report it as a differential failure too
        raise AssertionError(
            f"seed={seed}: unexpected {type(e).__name__}: {e}\n"
            f"--- reproduce: python3 -c 'import test_model_based as t; t.run({seed})' ---\n"
            f"--- last operations ---\n" + "\n".join(trace[-25:])
        ) from e


def _run_inner(seed, ops, trace):
    rnd = random.Random(seed)
    retention = rnd.choice([1, 3, 10, 40, None])
    clk = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    svc = InventoryService(clock=lambda: clk[0], retention_seconds=retention)
    svc._COMPACT_MIN = rnd.choice([0, 2, 8])       # exercise compaction constantly
    ref = Ref(retention)
    for p in ("p1", "p2"):
        q = rnd.randint(5, 30)
        svc.add_product(p, q); ref.avail[p] = q
    real2ref, ids = {}, []
    ever = {}   # model id -> its deadline, for every reservation ever created (incl. purged)
    trace.append(f"setup: retention={retention} compact_min={svc._COMPACT_MIN} stock={dict(ref.avail)}")
    T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def fmt(got):
        if got[0] == "ok" and hasattr(got[1], "reservation_id"):
            r = got[1]
            return f"ok(id={r.reservation_id[:8]} {r.customer_id} {r.product_id} qty={r.quantity} ttl={r.ttl_seconds} {r.status.value})"
        return str(got)

    def outcome(fn):
        try:
            r = fn(); return ("ok", r)
        except InventoryError as e:
            return ("err", type(e).__name__)

    for step in range(ops):
        now = clk[0]
        op = rnd.choices(["reserve", "confirm", "cancel", "get", "adv", "sweep", "avail"],
                         [30, 12, 12, 6, 20, 6, 6])[0]
        ctx = f"seed={seed} step={step} op={op}"
        t = f"t={(now - T0).total_seconds():.0f}s"
        if op == "adv":
            d = rnd.choice([0, 1, 2, 5, 15, 60])
            trace.append(f"{step}: advance clock +{d}s ({t} -> t={(now - T0).total_seconds() + d:.0f}s)")
            clk[0] += timedelta(seconds=d); continue
        if op == "reserve":
            c, k = f"c{rnd.randint(0,2)}", f"k{rnd.randint(0,8)}"
            p = rnd.choice(["p1", "p1", "p2", "nope"]); q = rnd.randint(1, 6)
            ttl = rnd.choice([1, 5, 20, 50])
            trace.append(f"{step}: reserve customer={c} product={p} qty={q} key={k} ttl={ttl}s @ {t}")
            got = outcome(lambda: svc.reserve(c, p, q, k, ttl_seconds=ttl))
            ref.sweep(now)
            exp_id = ref.idem.get((c, k))
            if exp_id is not None:
                e = ref.res[exp_id]
                exp = ("err", "IdempotencyConflict") if (e["p"], e["q"], e["ttl"]) != (p, q, ttl) else ("ok", exp_id)
            elif p not in ref.avail: exp = ("err", "ProductNotFound")
            elif ref.avail[p] < q: exp = ("err", "InsufficientInventory")
            else:
                ref.n += 1; rid = f"r{ref.n}"
                ref.res[rid] = dict(c=c, k=k, p=p, q=q, ttl=ttl, st="ACTIVE", exp=now + timedelta(seconds=ttl))
                ever[rid] = now + timedelta(seconds=ttl)
                ref.idem[(c, k)] = rid; ref.avail[p] -= q; exp = ("ok", rid)
            trace.append(f"     service={fmt(got)} model={exp}")
            if got[0] == "ok":
                assert exp[0] == "ok", (
                    f"{ctx}: service SUCCEEDED (reservation {got[1].reservation_id}) "
                    f"but the model expected {exp}")
                real2ref.setdefault(got[1].reservation_id, exp[1])
                ids.append(got[1].reservation_id)
                got = ("ok", real2ref[got[1].reservation_id])
            assert got == exp, f"{ctx}: got {got} expected {exp}"
        elif op in ("confirm", "cancel", "get"):
            real_id = rnd.choice(ids) if ids and rnd.random() < .9 else "unknown"
            trace.append(f"{step}: {op} reservation={real_id[:8]} (model {real2ref.get(real_id)}) @ {t}")
            got = outcome(lambda: getattr(svc, "get_reservation" if op == "get" else op)(real_id))
            ref.sweep(now)
            rid = real2ref.get(real_id)
            r = ref.res.get(rid)
            if r is None: exp = ("err", "ReservationNotFound")
            elif op == "get": exp = ("ok", r["st"])
            elif op == "confirm":
                if r["st"] == "CONFIRMED": exp = ("ok", "CONFIRMED")
                elif r["st"] != "ACTIVE": exp = ("err", "InvalidStateTransition")
                else: r["st"], r["term"] = "CONFIRMED", now; exp = ("ok", "CONFIRMED")
            else:
                if r["st"] == "CANCELLED": exp = ("ok", "CANCELLED")
                elif r["st"] != "ACTIVE": exp = ("err", "InvalidStateTransition")
                else:
                    r["st"], r["term"] = "CANCELLED", now; ref.avail[r["p"]] += r["q"]; exp = ("ok", "CANCELLED")
            trace.append(f"     service={fmt(got)} model={exp}")
            if got[0] == "ok": got = ("ok", got[1].status.value)
            assert got == exp, f"{ctx}: got {got} expected {exp}"
        elif op == "sweep":
            got = sorted(real2ref[i] for i in svc.expire_reservations())
            want = sorted(ref.sweep(now))
            trace.append(f"{step}: expire_reservations @ {t}\n     service={got} model={want}")
            assert got == want, f"{ctx}: expired sets differ: service {got} model {want}"
        else:
            p = rnd.choice(["p1", "p2", "nope"])
            got = svc.get_available_inventory(p); ref.sweep(now)
            trace.append(f"{step}: get_available_inventory product={p} @ {t}\n     service={got} model={ref.avail.get(p, 0)}")
            assert got == ref.avail.get(p, 0), f"{ctx}: avail {got} != {ref.avail.get(p, 0)}"
        # ---- structural checks the old suite never made ----
        try:
            svc.check_invariants()
        except AssertionError as e:
            raise AssertionError(f"{ctx}: {e}") from e
        assert {k: v for k, v in svc._available.items()} == ref.avail, f"{ctx}: stock drift"
        assert len(svc._reservations) == len(ref.res), f"{ctx}: retention drift {len(svc._reservations)} vs {len(ref.res)}"
        # Exact contents of both heaps, translated into the model's ids.
        live = {r.reservation_id: r for r in svc._reservations.values() if r.status is Status.ACTIVE}
        heap_ids = [rid for _, rid in svc._expiry_heap]
        assert set(live) <= set(heap_ids), f"{ctx}: ACTIVE reservation missing from expiry heap (would never expire)"
        assert len(heap_ids) == len(set(heap_ids)), f"{ctx}: duplicate heap entries"
        dead = sum(1 for rid in heap_ids if rid not in live)
        assert svc._stale == dead, f"{ctx}: _stale={svc._stale} but {dead} dead entries"
        for d, rid in svc._expiry_heap:          # live AND stale entries
            mid = real2ref[rid]
            assert d == ever[mid], f"{ctx}: heap entry for {mid} has deadline {d}, reservation was created with {ever[mid]}"
            if rid not in live:
                m = ref.res.get(mid)
                assert m is None or m["st"] != "ACTIVE", f"{ctx}: model says {mid} is ACTIVE but service treats its entry as dead"
        got_exp = sorted((d, real2ref[rid]) for d, rid in svc._expiry_heap if rid in live)
        want_exp = sorted((r["exp"], rid) for rid, r in ref.res.items() if r["st"] == "ACTIVE")
        assert got_exp == want_exp, f"{ctx}: expiry deadlines differ\n  service {got_exp}\n  model   {want_exp}"
        if retention is None:
            assert svc._purge_heap == [], f"{ctx}: purge heap should be empty when retention is None"
        else:
            got_purge = sorted((d, real2ref[rid]) for d, rid in svc._purge_heap)
            want_purge = sorted((r["term"] + timedelta(seconds=retention), rid)
                                for rid, r in ref.res.items() if r["st"] != "ACTIVE")
            assert got_purge == want_purge, f"{ctx}: purge entries differ\n  service {got_purge}\n  model   {want_purge}"

REGRESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression_seeds.json")


def regression_seeds():
    try:
        return json.load(open(REGRESSION_FILE))
    except FileNotFoundError:
        return []


class ModelBasedTests(unittest.TestCase):
    """Permanent part of the suite. Every seed that ever failed is replayed forever."""

    def test_regression_seeds(self):
        for seed in regression_seeds():
            with self.subTest(seed=seed):
                run(seed)

    def test_random_seeds(self):
        for seed in range(300):
            with self.subTest(seed=seed):
                run(seed)


if __name__ == "__main__":
    bad = {}
    for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 500):
        try: run(seed)
        except AssertionError as e:
            known = regression_seeds()
            if seed not in known:   # preserve the failing seed as a permanent regression test
                json.dump(sorted(known + [seed]), open(REGRESSION_FILE, "w"))
            bad.setdefault(str(e).split(": ", 1)[-1][:70], (seed, str(e)))
    print(f"{len(bad)} distinct failure kinds")
    for k, (seed, msg) in bad.items(): print(" -", msg[:200])
