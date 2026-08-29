# -*- coding: utf-8 -*-
"""
문서 감사 — 결과서에 적힌 숫자·경로가 실제 파일과 맞는지 검사한다.

왜 필요한가
    숫자 하나가 바뀌면 문서 여러 곳을 손으로 따라 고쳐야 하는데 꼭 하나씩 빠진다.
    실제로 겪은 것들:
      배포 모델을 다시 저장할 때마다 임계값이 바뀐다
        (0.3463 -> 0.3735 -> 0.2899). 한 번은 문서 5곳 중 4곳만
        고쳐서 1곳이 옛 값으로 남았다
      main 병합으로 결과 4줄 추가 -> 문서의 "61줄" 이 옛 값이 됐다
      explain.py -> explain_dl.py 개명 -> 문서의 경로가 죽은 링크가 됐다

    사람이 눈으로 잡기 어려운 종류다. 3초면 도는 검사로 막는다.

    uv run python -m src.verify_docs                    # docs/*.md 전부
    uv run python -m src.verify_docs docs/02_학습결과서.md

무엇을 보는가
    1 경로    문서의 `path/to/file` 이 실재하는가
    2 파일값  `models/x.json` 옆에 적힌 숫자가 그 파일에 실제로 있는가
    3 숫자    0.xxxx 값이 원본 파일에 있거나 두 값의 차·합으로 설명되는가
    4 줄 수   "N줄" 이 results/*.csv 의 실제 행수와 맞는가
    5 표 검산 "A − B" 열이 실제 A−B 와 맞는가
    6 구조    표 열 개수 · 코드블록 짝 · 굵게 표시 짝

한계
    3번은 완전하지 않다. 우연히 다른 값과 일치하면 통과한다.
    "틀린 걸 반드시 잡는" 검사가 아니라 "명백히 어긋난 걸 싸게 거르는" 검사다.
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.config import ROOT

DOCS = ROOT / "docs"
소수 = re.compile(r"(?<![\w.])(\d\.\d{3,4})(?![\d])")          # 0.7232 꼴
경로 = re.compile(r"`((?:src|models|results|docs|reports|data|app|notebooks)/[\w./\-가-힣*]+)`")
줄수 = re.compile(r"\*{0,2}(\d{2,4})줄\*{0,2}")


# ── 원본에서 값 모으기 ──────────────────────────────────────────
def 원본값():
    """models/*.json 과 results/*.csv 에 실제로 있는 숫자를 전부 모은다."""
    vals = set()
    for p in sorted((ROOT / "models").glob("*.json")):
        def 훑기(o):
            if isinstance(o, dict):
                for v in o.values():
                    훑기(v)
            elif isinstance(o, list):
                for v in o:
                    훑기(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                vals.add(round(float(o), 4))
        try:
            훑기(json.load(open(p, encoding="utf-8")))
        except Exception:
            pass
    for p in sorted((ROOT / "results").rglob("*.csv")):
        try:
            df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        except Exception:
            continue
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            vals.update(round(float(v), 4) for v in s)
    return vals


def 설명가능(x, vals, tol=6e-5):
    """값 자체이거나, 두 값의 차·합이면 설명된 것으로 본다."""
    if any(abs(x - v) <= tol for v in vals):
        return True
    큰값 = [v for v in vals if 0 < v <= 1.2]                    # 지표끼리의 차만 본다
    for a, b in itertools.combinations(큰값, 2):
        if abs(abs(a - b) - x) <= tol:
            return True
    return False


# ── 검사들 ──────────────────────────────────────────────────────
def 검사_경로(doc):
    나쁨 = []
    for p in sorted(set(경로.findall(doc))):
        있음 = (bool(list((ROOT / p).parent.glob((ROOT / p).name)))
                if "*" in p else (ROOT / p).exists())
        if not 있음:
            나쁨.append(f"없는 경로: {p}")
    return 나쁨


def 검사_숫자(doc, vals):
    나쁨 = []
    for s in sorted(set(소수.findall(doc))):
        # 지표·임계값 범위만 본다. 배수(3.6) · 시간 같은 값은 원본에 없는 게 정상이다.
        if float(s) >= 1.0:
            continue
        if not 설명가능(float(s), vals):
            나쁨.append(f"원본에 없는 값: {s}")
    return 나쁨


def 검사_파일값(doc):
    """
    한 줄에 `models/xxx.json` 경로와 숫자가 같이 나오면, 그 숫자가 그 파일 안에
    실제로 있는지 본다.

    ★ 이게 제일 잘 잡는다. "숫자" 검사는 원본 값이 1,500개나 돼서
      옛 값도 우연히 '두 값의 차' 로 설명돼 통과해버린다.
      실제로 임계값 0.3463 이 그렇게 빠져나갔다. 이 검사는 정확히 그걸 잡는다.
    """
    캐시 = {}

    def 값들(p):
        if p not in 캐시:
            s = set()

            def 훑기(o):
                if isinstance(o, dict):
                    for v in o.values():
                        훑기(v)
                elif isinstance(o, list):
                    for v in o:
                        훑기(v)
                elif isinstance(o, (int, float)) and not isinstance(o, bool):
                    s.add(round(float(o), 4))
            try:
                훑기(json.load(open(ROOT / p, encoding="utf-8")))
            except Exception:
                s = None
            캐시[p] = s
        return 캐시[p]

    나쁨 = []
    for 줄 in doc.split("\n"):
        for p in re.findall(r"`((?:models|results|data)/[\w./\-]+\.json)`", 줄):
            실제 = 값들(p)
            if not 실제:
                continue
            뒤 = 줄[줄.index(p) + len(p):]                      # 경로 뒤쪽 숫자만
            for s in 소수.findall(뒤):
                if not any(abs(float(s) - v) <= 6e-5 for v in 실제):
                    나쁨.append(f"{p} 에 없는 값 {s} 가 적혀 있음 "
                              f"(파일에는 {sorted(v for v in 실제 if v < 1)[:6]}…)")
    return 나쁨


def 검사_줄수(doc):
    """
    합계를 뜻하는 표현만 본다 — "표 N줄", "N줄 전체".
    "기본값 32줄" 처럼 구성요소를 가리키는 것은 건너뛴다.
    "N줄 — a + b + c" 꼴은 합이 맞는지도 검산한다.
    """
    실제 = set()
    for p in sorted((ROOT / "results").glob("*.csv")):
        try:
            실제.add(len(pd.read_csv(p, encoding="utf-8-sig")))
        except Exception:
            pass
    나쁨 = []
    if 실제:
        for m in re.finditer(r"표\s*\*{0,2}(\d+)줄|\*{0,2}(\d+)줄\*{0,2}\s*전체", doc):
            n = int(m.group(1) or m.group(2))
            if n not in 실제:
                나쁨.append(f'합계 "{n}줄" 이 results/*.csv 행수 {sorted(실제)} 와 안 맞음')
    한줄 = r"[^\n|]*?"
    분해 = (r"(\d+)줄\*{0,2}\s*[—-]\s*" + 한줄 + r"(\d+)" + 한줄 + r"\+" + 한줄
           + r"(\d+)" + 한줄 + r"\+" + 한줄 + r"(\d+)")
    for m in re.finditer(분해, doc):
        총, *부분 = (int(x) for x in m.groups())
        if sum(부분) != 총:
            나쁨.append(f'분해 안 맞음: {" + ".join(map(str, 부분))} = {sum(부분)} 인데 {총}줄 로 적힘')
    return 나쁨


def 검사_표검산(doc):
    """머리행이 'A − B' 인 열이 실제 A−B 와 맞는지."""
    나쁨, ls = [], doc.split("\n")
    i = 0
    while i < len(ls):
        if ls[i].startswith("|") and i + 1 < len(ls) and \
           re.fullmatch(r"\|[-: |]+\|", ls[i + 1].strip()):
            head = [c.strip() for c in ls[i].strip().strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(ls) and ls[j].startswith("|"):
                rows.append([c.strip() for c in ls[j].strip().strip("|").split("|")])
                j += 1
            for ci, h in enumerate(head):
                # ★ 공백을 반드시 요구한다. 안 그러면 "PR-AUC" 를
                #   "PR 빼기 AUC" 로 읽어 표 전체를 오탐한다.
                m = re.match(r"(.+?)\s+[−–-]\s+(.+)", h)
                if not m:
                    continue
                왼, 오 = (x.strip() for x in m.groups())
                cand = [(k, x) for k, x in enumerate(head)
                        if 왼 in x or x in 왼]
                cand2 = [(k, x) for k, x in enumerate(head)
                         if (오 in x or x in 오) and k != ci]
                if not cand or not cand2:
                    continue
                a, b = cand[0][0], cand2[0][0]
                for r in rows:
                    try:
                        av = float(re.sub(r"[^\d.\-]", "", r[a]))
                        bv = float(re.sub(r"[^\d.\-]", "", r[b]))
                        cv = float(re.sub(r"[^\d.\-]", "", r[ci]))
                    except (ValueError, IndexError):
                        continue
                    if abs((av - bv) - cv) > 6e-5:
                        나쁨.append(f'"{h}" 행 {r[0][:20]}: {av}−{bv}={av-bv:.4f} '
                                  f'인데 {cv} 로 적힘')
            i = j
        else:
            i += 1
    return 나쁨


def 검사_구조(doc):
    나쁨, ls = [], doc.split("\n")
    i = 0
    while i < len(ls):
        if ls[i].startswith("|") and i + 1 < len(ls) and \
           re.fullmatch(r"\|[-: |]+\|", ls[i + 1].strip()):
            n, j = ls[i].count("|"), i
            while j < len(ls) and ls[j].startswith("|"):
                if ls[j].count("|") != n:
                    나쁨.append(f"표 열 개수 불일치 {j+1}줄: {ls[j][:44]}")
                j += 1
            i = j
        else:
            i += 1
    if sum(1 for l in ls if l.startswith("```")) % 2:
        나쁨.append("코드블록 ``` 짝이 안 맞음")
    for k, l in enumerate(ls, 1):
        if l.count("**") % 2:
            나쁨.append(f"굵게 ** 짝이 안 맞음 {k}줄: {l[:44]}")
    return 나쁨


# ── 실행 ────────────────────────────────────────────────────────
def 감사(path, vals):
    doc = Path(path).read_text(encoding="utf-8")
    결과 = {"경로": 검사_경로(doc), "파일값": 검사_파일값(doc),
           "숫자": 검사_숫자(doc, vals), "줄 수": 검사_줄수(doc),
           "표 검산": 검사_표검산(doc), "구조": 검사_구조(doc)}
    총 = sum(len(v) for v in 결과.values())
    print(f"\n{'=' * 70}\n{Path(path).name}\n{'=' * 70}")
    for 이름, 목록 in 결과.items():
        print(f"  {'OK ' if not 목록 else '★XX'} {이름:<8s} {len(목록)}건")
        for x in 목록:
            print(f"        - {x}")
    return 총


def main(argv=None):
    # 윈도우 기본 콘솔(cp949)은 '—' 같은 글자를 못 찍고 죽는다.
    # 검사는 다 끝났는데 마지막 줄에서 터져서 결과를 못 본다.
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="결과서 숫자·경로 감사")
    ap.add_argument("문서", nargs="*", help="생략하면 docs/*.md 전부")
    a = ap.parse_args(argv)

    대상 = [Path(x) for x in a.문서] or sorted(DOCS.glob("*.md"))
    if not 대상:
        print("검사할 문서가 없습니다."); return

    print("원본 값 모으는 중 (models/*.json · results/**/*.csv)…")
    vals = 원본값()
    print(f"  {len(vals):,}개")

    총 = sum(감사(p, vals) for p in 대상)
    print(f"\n{'=' * 70}")
    print(f"문서 {len(대상)}개 · 지적 {총}건" if 총 else
          f"문서 {len(대상)}개 — 문제 없음")


if __name__ == "__main__":
    main()
