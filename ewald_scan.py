#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

D = 8
V_RE = re.compile(rb"<v>([^<]+)</v>")
TERNARY = np.array(list(itertools.product((-1, 0, 1), repeat=D)), dtype=np.int16)
UNIT_INDEX: list[int] = []
for j in range(D):
    e = np.zeros(D, dtype=np.int16)
    e[j] = 1
    UNIT_INDEX.append(int(np.flatnonzero(np.all(TERNARY == e, axis=1))[0]))


def det_bareiss(rows: list[list[int]] | np.ndarray) -> int:
    A = [[int(x) for x in row] for row in rows]
    n = len(A)
    if n == 0:
        return 1
    sign = 1
    prev = 1
    for k in range(n - 1):
        piv = next((i for i in range(k, n) if A[i][k] != 0), None)
        if piv is None:
            return 0
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            sign = -sign
        pivot = A[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * pivot - A[i][k] * A[k][j]) // prev
            A[i][k] = 0
        prev = pivot
    return sign * A[n - 1][n - 1]


def parse_vertices(raw: bytes) -> np.ndarray:
    rows: list[list[int]] = []
    for m in V_RE.finditer(raw):
        vals = [int(x) for x in m.group(1).split()]
        if len(vals) != D + 1 or vals[0] != 1:
            raise ValueError(f"unexpected homogeneous vertex row: {vals}")
        rows.append(vals[1:])
    if not rows:
        raise ValueError("no vertices")
    return np.asarray(rows, dtype=np.int64)


def primitive_normal_from_simplex(V: np.ndarray, simplex: np.ndarray) -> tuple[int, ...]:
    # Exact integer kernel of the 7x8 matrix of within-facet differences.
    W = V[simplex[1:]] - V[simplex[0]]
    cof: list[int] = []
    for j in range(D):
        minor = np.delete(W, j, axis=1)
        cof.append(((-1) ** j) * det_bareiss(minor))
    g = 0
    for x in cof:
        g = math.gcd(g, abs(x))
    if g == 0:
        raise ValueError("degenerate facet simplex")
    u = np.asarray([x // g for x in cof], dtype=np.int64)
    vals = V @ u
    mx, mn = int(vals.max()), int(vals.min())
    if mx == 1 and np.all(vals <= 1):
        return tuple(int(x) for x in u)
    if mn == -1 and np.all(vals >= -1):
        return tuple(int(-x) for x in u)
    raise ValueError(f"failed exact facet normalization: range=({mn},{mx}), u={u.tolist()}")


def facet_normals(V: np.ndarray) -> list[tuple[int, ...]]:
    hull = ConvexHull(V.astype(np.float64), qhull_options="Qx")
    out: set[tuple[int, ...]] = set()
    for eq, simplex in zip(hull.equations, hull.simplices):
        off = float(eq[-1])
        if abs(off) < 1e-12:
            raise ValueError("facet through origin")
        q = eq[:-1] / (-off)
        u = tuple(int(x) for x in np.rint(q))
        ua = np.asarray(u, dtype=np.int64)
        vals = V @ ua
        if np.max(np.abs(q - ua)) > 1e-6 or int(vals.max()) != 1 or np.any(vals > 1):
            u = primitive_normal_from_simplex(V, simplex)
        out.add(u)
    U = sorted(out)
    for u in U:
        vals = V @ np.asarray(u, dtype=np.int64)
        if int(vals.max()) != 1 or np.any(vals > 1):
            raise ValueError(f"invalid facet {u}")
    for j in range(D):
        e = tuple(1 if i == j else 0 for i in range(D))
        if e not in out:
            raise ValueError(f"missing normalized coordinate facet e_{j}")
    return U


def rank_mod_rows(E: np.ndarray, p: int) -> int:
    pivots: list[np.ndarray | None] = [None] * D
    rank = 0
    for row in E:
        a = np.mod(row, p).astype(np.int16, copy=True)
        for c in range(D):
            x = int(a[c]) % p
            if x == 0:
                continue
            b = pivots[c]
            if b is None:
                a = (a * pow(x, -1, p)) % p
                pivots[c] = a
                rank += 1
                if rank == D:
                    return D
                break
            a = (a - x * b) % p
    return rank


def canonical_half(E: np.ndarray) -> np.ndarray:
    keep: list[tuple[int, ...]] = []
    for row in E:
        nz = np.flatnonzero(row)
        if len(nz) and int(row[int(nz[0])]) > 0:
            keep.append(tuple(int(x) for x in row))
    keep.sort(key=lambda v: (sum(x != 0 for x in v), v))
    return np.asarray(keep, dtype=np.int16) if keep else np.empty((0, D), dtype=np.int16)


def quick_unimodular_basis(E: np.ndarray, combo_cap: int = 10000) -> tuple[bool, list[list[int]] | None]:
    eset = {tuple(int(x) for x in row) for row in E}
    units = np.eye(D, dtype=np.int16)
    if all(tuple(int(x) for x in units[j]) in eset for j in range(D)):
        return True, units.astype(int).tolist()

    H = canonical_half(E)
    if len(H) < D:
        return False, None

    # A few deterministic greedy orderings; exact determinant at full rank.
    rng = np.random.default_rng(0xEAA1D)
    orderings = [np.arange(len(H))]
    orderings.extend(rng.permutation(len(H)) for _ in range(12))
    for order in orderings:
        chosen: list[np.ndarray] = []
        rank2 = 0
        for idx in order:
            cand = H[int(idx)]
            trial = np.asarray(chosen + [cand], dtype=np.float64)
            new_rank = int(np.linalg.matrix_rank(trial, tol=1e-9))
            if new_rank > rank2:
                chosen.append(cand)
                rank2 = new_rank
                if len(chosen) == D:
                    if abs(det_bareiss(np.asarray(chosen, dtype=np.int16))) == 1:
                        return True, np.asarray(chosen, dtype=int).tolist()
                    break

    # Deterministic bounded search among lowest-support points.
    k = min(len(H), 22)
    for tested, inds in enumerate(itertools.combinations(range(k), D), start=1):
        if abs(det_bareiss(H[list(inds)])) == 1:
            return True, H[list(inds)].astype(int).tolist()
        if tested >= combo_cap:
            break
    return False, None


def scan(archive: Path, limit: int | None, output: Path, do_quick_basis: bool) -> None:
    t0 = time.perf_counter()
    mask_cache: dict[tuple[int, ...], np.ndarray] = {}
    records: list[dict] = []
    failures: list[dict] = []
    rank_bad: list[dict] = []
    heuristic_bad: list[dict] = []
    e_hist: Counter[int] = Counter()
    processed = 0

    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".poly")]
        members.sort(key=lambda m: m.name)
        if limit is not None:
            members = members[:limit]
        for member in members:
            try:
                fh = tf.extractfile(member)
                if fh is None:
                    raise ValueError("cannot extract member")
                V = parse_vertices(fh.read())
                U = facet_normals(V)
                valid = np.ones(len(TERNARY), dtype=bool)
                for u in U:
                    mm = mask_cache.get(u)
                    if mm is None:
                        mm = np.abs(TERNARY.astype(np.int64) @ np.asarray(u, dtype=np.int64)) <= 1
                        mask_cache[u] = mm
                    valid &= mm
                E = TERNARY[valid]
                r2 = rank_mod_rows(E, 2)
                r3 = rank_mod_rows(E, 3)
                r5 = rank_mod_rows(E, 5)
                units = sum(bool(valid[idx]) for idx in UNIT_INDEX)
                has_quick = None
                basis = None
                if do_quick_basis:
                    has_quick, basis = quick_unimodular_basis(E)
                rec = {
                    "file": member.name,
                    "vertices": int(len(V)),
                    "facets": int(len(U)),
                    "e_count": int(len(E)),
                    "half_count": int((len(E) - 1) // 2),
                    "unit_count": int(units),
                    "rank_mod_2": int(r2),
                    "rank_mod_3": int(r3),
                    "rank_mod_5": int(r5),
                    "quick_basis": has_quick,
                    "basis": basis,
                }
                records.append(rec)
                e_hist[int(len(E))] += 1
                detailed = {
                    **rec,
                    "vertices_data": V.astype(int).tolist(),
                    "facet_normals": [list(u) for u in U],
                    "ewald_points": E.astype(int).tolist(),
                }
                if min(r2, r3, r5) < D:
                    rank_bad.append(detailed)
                elif do_quick_basis and not has_quick:
                    heuristic_bad.append(detailed)
            except (ValueError, QhullError, ArithmeticError) as exc:
                failures.append({"file": member.name, "error": repr(exc)})
            processed += 1
            if processed % 500 == 0:
                print(
                    f"PROGRESS {processed} elapsed={time.perf_counter()-t0:.2f}s "
                    f"rank_bad={len(rank_bad)} heuristic_bad={len(heuristic_bad)} failures={len(failures)}",
                    flush=True,
                )

    records.sort(key=lambda r: (r["e_count"], r["unit_count"], r["file"]))
    summary = {
        "archive": str(archive),
        "processed": processed,
        "elapsed_seconds": time.perf_counter() - t0,
        "mask_cache_size": len(mask_cache),
        "failure_count": len(failures),
        "failures": failures[:100],
        "rank_bad_count": len(rank_bad),
        "rank_bad": rank_bad[:100],
        "heuristic_bad_count": len(heuristic_bad),
        "heuristic_bad": heuristic_bad[:200],
        "smallest": records[:200],
        "e_count_histogram": dict(sorted(e_hist.items())),
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "SUMMARY "
        + json.dumps(
            {k: summary[k] for k in (
                "processed",
                "elapsed_seconds",
                "mask_cache_size",
                "failure_count",
                "rank_bad_count",
                "heuristic_bad_count",
            )}
        ),
        flush=True,
    )
    for rec in records[:20]:
        print("SMALL " + json.dumps(rec), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output", type=Path, default=Path("scan-result.json"))
    ap.add_argument("--quick-basis", action="store_true")
    args = ap.parse_args()
    scan(args.archive, args.limit, args.output, args.quick_basis)


if __name__ == "__main__":
    main()
