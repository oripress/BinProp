"""Explain saved JGDB continuations with Opus 5 motifs; game details: docs/go.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sgfmill import sgf, sgf_moves

from binary_properties.inference import infer
from binary_properties.io import position_from_dict
from binary_properties.libraries import load_builtin_library

LIBRARY_ID = "go/claude-opus-5"
GAME_ID = "go_test_00002616_move149"
SGF_MEMBER = "jgdb/sgf/test/0002/00002616.sgf"
SGF_SHA256 = "f52d378492b497abadb421ce1b484ce46a49613866d182ce7a632681ccd03c4d"
ROOT_PLY = 149
COLOR = "W"
# These are saved engine variations from the same root, not actual game moves.
POSITIVE_LINE = "N7 T7 S6 T5 T4 Q7 R5 S4".split()
NEGATIVE_LINE = "P3 Q4 N7 O6 M6 P6 P5 P7".split()
# Original record: Black 7d, White 8d, komi 0.5, result W+4.50.
# The source does not identify the players or date.
SGF = (
    b"(;\nSZ[19]\nWR[8d]\nBR[7d]\nKM[0.50]\nRE[W+4.50]\n;B[pp];W[dd];B[pc];W[pe];B"
    b"[cp];W[ep];B[eq];W[fq];B[dq];W[qc];B[qb];W[qd];B[mc];W[gp];B[cn];W[nq]"
    b";B[kp];W[no];B[pn];W[kn];B[hq];W[hp];B[fr];W[iq];B[gq];W[jp];B[oq];W[n"
    b"r];B[ph];W[ql];B[qf];W[ob];B[pb];W[pf];B[rb];W[qg];B[qh];W[rg];B[rh];W"
    b"[qo];B[nn];W[qn];B[mo];W[np];B[po];W[mn];B[ln];W[mm];B[pm];W[qp];B[qm]"
    b";W[rm];B[qq];W[rk];B[rp];W[ro];B[so];W[rn];B[rr];W[ok];B[ko];W[lm];B[j"
    b"o];W[kq];B[lq];W[lr];B[io];W[ip];B[kr];W[jq];B[lo];W[mq];B[im];W[nh];B"
    b"[ni];W[mi];B[nj];W[nm];B[mh];W[ng];B[pj];W[nk];B[mj];W[pr];B[or];W[os]"
    b";B[qr];W[sn];B[pq];W[ps];B[sq];W[sp];B[ol];W[rq];B[nl];W[mk];B[rp];W[p"
    b"k];B[so];W[nd];B[od];W[sp];B[ml];W[lk];B[li];W[ne];B[nc];W[rq];B[ll];W"
    b"[kl];B[rp];W[so];B[kk];W[jl];B[jk];W[rq];B[il];W[on];B[rp];W[kj];B[jj]"
    b";W[rq];B[km];W[jm];B[rp];W[sl];B[qs];W[rq];B[om];W[nn];B[rp];W[ns];B[s"
    b"s];W[rq];B[jn];W[km];B[rp];W[qj];B[oi];W[rq];B[ls];W[lp];B[rp];W[sh];B"
    b"[rq];W[ki];B[si];W[kh];B[mf];W[mg];B[lg];W[lf];B[pg];W[rf];B[of];W[oe]"
    b";B[nf];W[kg];B[me];W[lh];B[og];W[md];B[ld];W[sg];B[le];W[ik];B[hk];W[i"
    b"j];B[hj];W[ji];B[gl];W[cf];B[db];W[dc];B[eb];W[cb];B[bb];W[cc];B[ca];W"
    b"[gc];B[fc];W[gb];B[fd];W[gd];B[fe];W[ci];B[ge];W[he];B[hf];W[ie];B[gg]"
    b";W[cm];B[cg];W[dg];B[ch];W[dh];B[bi];W[cj];B[bf];W[be];B[ce];W[df];B[b"
    b"d];W[fb];B[ae];W[ec];B[de];W[ea];B[bj];W[ee];B[ef];W[ed];B[fg];W[ei];B"
    b"[fj];W[ej];B[eh];W[fi];B[gi];W[dm];B[fk];W[dn];B[bn];W[bc];B[fp];W[fo]"
    b";B[en];W[do];B[eo];W[dp];B[fq];W[co];B[bo];W[bp];B[cq];W[bm];B[bq];W[d"
    b"i];B[el];W[eg];B[fh];W[ck];B[ff];W[dl];B[bk];W[bl];B[rj];W[rc];B[jc];W"
    b"[an];B[ap];W[ac];B[ir];W[jr];B[ib];W[sb];B[ak];W[hn];B[in];W[hh];B[ih]"
    b";W[hg];B[gh];W[ig];B[fn];W[gn];B[go];W[ho];B[fo];W[cd];B[be];W[hr];B[h"
    b"i];W[ii];B[if];W[jg];B[jf];W[ic];B[id];W[hd];B[hc];W[hb];B[ic];W[je];B"
    b"[kf];W[ke];B[lg];W[ia];B[oh];W[kc];B[kb];W[jd];B[jb];W[lb];B[lc];W[ka]"
    b";B[kd];W[ja];B[kc];W[mb];B[la];W[ha];B[nb];W[ma];B[na];W[la];B[ek];W[d"
    b"k];B[qi];W[qk];B[hs];W[is];B[gs];W[mp];B[gm];W[hm];B[hl];W[em];B[fm];W"
    b"[oj];B[lj];W[pd];B[oc];W[re];B[ra];W[am];B[ao];W[ir];B[gr];W[pi];B[sc]"
    b";W[sd];B[pj];W[aq];B[ar];W[pi];B[sa];W[sc];B[pj];W[mg];B[ng];W[pi];B[m"
    b"r];W[ks];B[pj];W[bg];B[bh];W[pi];B[ag];W[pj];B[ri];W[sk];B[ad];W[da];B"
    b"[tt];W[tt])\n"
)


def _play(board, color, point, ko):
    if point is None:
        return None
    if point == ko:
        raise ValueError("Illegal simple-ko recapture in the saved variation")
    next_ko = board.play(*point, color)
    if board.get(*point) != color:
        raise ValueError("Illegal self-capture in the saved variation")
    return next_ko


def root_position():
    """Replay the original SGF to the paper's root after 149 moves."""
    board, moves = sgf_moves.get_setup_and_moves(sgf.Sgf_game.from_bytes(SGF))
    ko = None
    for index, (color, point) in enumerate(moves[:ROOT_PLY]):
        if color != ("b" if index % 2 == 0 else "w"):
            raise ValueError("Unexpected player order in the game record")
        ko = _play(board, color, point, ko)
    return board, ko


def board_rows(board):
    """Preserve the paper builder's row order (reflected from SGF display)."""
    return [
        "".join((board.get(row, col) or ".").upper() for col in range(19))
        for row in range(19)
    ]


def comparison() -> dict:
    """Replay the two archived eight-ply paths, keeping White as evaluated color."""
    root, root_ko = root_position()
    groups = {}
    for group, line in (
        ("positive", POSITIVE_LINE),
        ("negative", NEGATIVE_LINE),
    ):
        board, ko = root.copy(), root_ko
        for index, move in enumerate(line):
            # The canonical paper builder uses row 0 as GTP rank 19.
            point = (19 - int(move[1:]), "ABCDEFGHJKLMNOPQRST".index(move[0]))
            ko = _play(board, "w" if index % 2 == 0 else "b", point, ko)
        groups[group] = [
            {
                "board_rows": board_rows(board),
                "color": COLOR,
                "id": GAME_ID + "-" + group,
            }
        ]
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-comparison", type=Path)
    args = parser.parse_args()
    groups = comparison()
    if args.write_comparison:
        with args.write_comparison.open("x", encoding="utf-8") as handle:
            json.dump(groups, handle, indent=2)
            handle.write("\n")
        print(f"Saved comparison to {args.write_comparison}")
    library = load_builtin_library(LIBRARY_ID)
    positive = [position_from_dict(row, "go") for row in groups["positive"]]
    negative = [position_from_dict(row, "go") for row in groups["negative"]]
    result = infer(library, positive, negative)
    print(f"Library: {LIBRARY_ID}")
    print(f"Game: JGDB {SGF_MEMBER}; Black 7d, White 8d; result W+4.50")
    print("Saved KataGo continuations after move 149: N7 versus P3")
    for motif in result["selected"]:
        print(f"\n{motif['name']}\n  {motif['description']}")
        print(
            f"  Support fit: {motif['support_fit']:.0%}; discovery delta: {motif['delta']:.4f}"
        )
    if result["errors"] or not result["selected"] or result["used_fallback"]:
        raise RuntimeError(f"Example could not explain the game comparison: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
