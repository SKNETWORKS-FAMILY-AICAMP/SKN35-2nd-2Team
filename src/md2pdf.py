# -*- coding: utf-8 -*-
"""
마크다운 문서를 PDF 로. 산출물 제출용.

이 레포에는 pandoc · wkhtmltopdf · weasyprint 가 없다. 설치하려면 팀 전원이
같은 걸 깔아야 하므로, 파이썬만으로 되게 reportlab 으로 직접 조판한다.

    uv run python -m src.md2pdf docs/학습결과서_머신러닝.md
    uv run python -m src.md2pdf docs/학습결과서_딥러닝.md docs/딥러닝.pdf

reportlab 은 pyproject 에 넣지 않았다. 문서 만들 때만 쓰므로
`uv run --with reportlab python -m src.md2pdf ...` 로 그때만 끌어다 쓴다
(위 명령이 안 되면 --with reportlab 을 붙여라).

★ 한글 폰트
  Courier 에는 한글 글리프가 없어서 `코드` 표기 안에 한글이 있으면 ■ 로 깨진다.
  그래서 한글이 섞인 코드는 한글 폰트로 조판한다. 실제로 한 번 당했다.

지원하는 마크다운
    # ## ###  ·  표  ·  ```코드블록```  ·  > 인용  ·  - 목록  ·  **굵게**  ·  `코드`
    그 외 문법은 그냥 본문으로 나온다. 결과서에 필요한 만큼만 만들었다.
"""
import argparse
import re
import sys
from pathlib import Path

# ── 한글 폰트 찾기 (맥·윈도우·리눅스) ────────────────────────────
FONT_후보 = [
    ("KR", "KR-B", "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"),
    ("KR", "KR-B", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
     "/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    ("KR", "KR-B", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
     "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
]


def _폰트등록():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for 보통, 굵게, p1, p2 in FONT_후보:
        if Path(p1).exists():
            pdfmetrics.registerFont(TTFont(보통, p1))
            pdfmetrics.registerFont(TTFont(굵게, p2 if Path(p2).exists() else p1))
            pdfmetrics.registerFontFamily(보통, normal=보통, bold=굵게,
                                          italic=보통, boldItalic=굵게)
            return 보통, 굵게
    raise SystemExit(
        "한글 폰트를 못 찾았습니다.\n"
        "  윈도우: 맑은 고딕이 기본으로 있어야 합니다\n"
        "  맥    : /System/Library/Fonts/AppleSDGothicNeo.ttc\n"
        "  리눅스: sudo apt install fonts-nanum")


def _스타일(보통, 굵게):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle

    C = dict(INK=colors.HexColor("#151A21"), INK2=colors.HexColor("#3D4654"),
             INK3=colors.HexColor("#5D6675"), RULE=colors.HexColor("#DEE3EA"),
             RULE2=colors.HexColor("#F1F3F6"), ACC=colors.HexColor("#2F4BC4"),
             ACCS=colors.HexColor("#E8ECFA"))
    S = {
        "h1": ParagraphStyle("h1", fontName=굵게, fontSize=20, leading=28,
                             textColor=C["INK"], spaceBefore=0, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName=굵게, fontSize=14, leading=20,
                             textColor=C["INK"], spaceBefore=20, spaceAfter=8),
        "h3": ParagraphStyle("h3", fontName=굵게, fontSize=11.5, leading=17,
                             textColor=C["INK"], spaceBefore=13, spaceAfter=5),
        "p": ParagraphStyle("p", fontName=보통, fontSize=9.3, leading=15.2,
                            textColor=C["INK2"], alignment=TA_LEFT, spaceAfter=6),
        "li": ParagraphStyle("li", fontName=보통, fontSize=9.3, leading=15.2,
                             textColor=C["INK2"], leftIndent=11, bulletIndent=2,
                             spaceAfter=3),
        "quote": ParagraphStyle("quote", fontName=보통, fontSize=9.1, leading=15,
                                textColor=C["INK3"], leftIndent=10,
                                spaceBefore=4, spaceAfter=8),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8.2,
                               leading=12.6, textColor=C["INK2"], leftIndent=8,
                               spaceBefore=4, spaceAfter=8),
        # 한글이 섞인 코드블록용 — Courier 로 조판하면 ■ 로 깨진다
        "code_kr": ParagraphStyle("code_kr", fontName=보통, fontSize=8.2,
                                  leading=13.4, textColor=C["INK2"], leftIndent=8,
                                  spaceBefore=4, spaceAfter=8),
        "meta": ParagraphStyle("meta", fontName=보통, fontSize=8.6, leading=14,
                               textColor=C["INK3"], spaceAfter=10),
        "th": ParagraphStyle("th", fontName=굵게, fontSize=7.6, leading=10.5,
                             textColor=C["INK3"]),
        "td": ParagraphStyle("td", fontName=보통, fontSize=7.8, leading=10.8,
                             textColor=C["INK2"]),
        "tdb": ParagraphStyle("tdb", fontName=굵게, fontSize=7.8, leading=10.8,
                              textColor=C["INK"]),
    }
    return S, C


def _인라인(t, 보통, 굵게):
    """**굵게** 와 `코드` 를 reportlab 태그로."""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", rf'<font name="{굵게}">\1</font>', t)

    def _코드(m):
        c = m.group(1)
        # ★ Courier 에는 한글이 없다. 섞여 있으면 한글 폰트로.
        return f'<font name="{"Courier" if c.isascii() else 보통}" size="8.4">{c}</font>'

    return re.sub(r"`(.+?)`", _코드, t)


def 만들기(md_path, pdf_path=None, 머리말=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate,
                                    Paragraph, Spacer, Table, TableStyle)

    md_path = Path(md_path)
    pdf_path = Path(pdf_path) if pdf_path else md_path.with_suffix(".pdf")
    보통, 굵게 = _폰트등록()
    S, C = _스타일(보통, 굵게)
    폭 = 168 * mm

    lines = md_path.read_text(encoding="utf-8").split("\n")
    flow, i = [], 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue

        if ln.startswith("```"):                                   # 코드블록
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i].replace("&", "&amp;")
                           .replace("<", "&lt;").replace(">", "&gt;"))
                i += 1
            i += 1
            본문 = "<br/>".join(buf) or " "
            st = S["code"] if 본문.isascii() else S["code_kr"]
            flow.append(Table([[Paragraph(본문, st)]], colWidths=[폭],
                              style=TableStyle([
                                  ("BACKGROUND", (0, 0), (-1, -1), C["RULE2"]),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                  ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                  ("TOPPADDING", (0, 0), (-1, -1), 7),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 7)])))
            flow.append(Spacer(1, 5))
            continue

        if ln.startswith("|"):                                     # 표
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.fullmatch(r"[-: ]+", "".join(cells)):
                    rows.append(cells)
                i += 1
            if rows:
                n = max(len(r) for r in rows)
                rows = [r + [""] * (n - len(r)) for r in rows]
                data = [[Paragraph(_인라인(c, 보통, 굵게), S["th"]) for c in rows[0]]]
                st = [("GRID", (0, 0), (-1, -1), 0.4, C["RULE"]),
                      ("BACKGROUND", (0, 0), (-1, 0), C["RULE2"]),
                      ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                      ("LEFTPADDING", (0, 0), (-1, -1), 5),
                      ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                      ("TOPPADDING", (0, 0), (-1, -1), 4),
                      ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
                for ri, r in enumerate(rows[1:], start=1):
                    굵은행 = r[0].startswith("**")
                    data.append([Paragraph(_인라인(c, 보통, 굵게),
                                           S["tdb"] if 굵은행 else S["td"]) for c in r])
                    if 굵은행:
                        st.append(("BACKGROUND", (0, ri), (-1, ri), C["ACCS"]))
                첫칸 = max(폭 * 0.20, 폭 / n)
                widths = ([첫칸] + [(폭 - 첫칸) / (n - 1)] * (n - 1)) if n > 1 else [폭]
                flow.append(Table(data, colWidths=widths, style=TableStyle(st),
                                  repeatRows=1))
                flow.append(Spacer(1, 9))
            continue

        if ln.startswith("---"):                                   # 구분선
            flow.append(Spacer(1, 4))
            flow.append(Table([[""]], colWidths=[폭], style=TableStyle(
                [("LINEABOVE", (0, 0), (-1, 0), 0.6, C["RULE"])])))
            flow.append(Spacer(1, 4))
            i += 1
            continue

        for 표시, 키 in (("### ", "h3"), ("## ", "h2"), ("# ", "h1")):
            if ln.startswith(표시):
                flow.append(Paragraph(_인라인(ln[len(표시):], 보통, 굵게), S[키]))
                break
        else:
            if ln.startswith("> "):                                # 인용
                buf = []
                while i < len(lines) and lines[i].startswith(">"):
                    buf.append(lines[i].lstrip("> ").rstrip())
                    i += 1
                flow.append(Table([[Paragraph(_인라인(" ".join(buf), 보통, 굵게),
                                              S["quote"])]], colWidths=[폭],
                                  style=TableStyle([
                                      ("LINEBEFORE", (0, 0), (0, 0), 2, C["ACC"]),
                                      ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                      ("TOPPADDING", (0, 0), (-1, -1), 5),
                                      ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])))
                flow.append(Spacer(1, 6))
                continue
            if re.match(r"^\s*[-*] ", ln):                          # 목록
                flow.append(Paragraph(_인라인(re.sub(r"^\s*[-*] ", "", ln), 보통, 굵게),
                                      S["li"], bulletText="·"))
            elif ln.startswith("**담당**"):
                flow.append(Paragraph(_인라인(ln, 보통, 굵게), S["meta"]))
            else:
                flow.append(Paragraph(_인라인(ln, 보통, 굵게), S["p"]))
        i += 1

    꼬리 = 머리말 or f"SKN35 2팀 · {md_path.stem}"

    def _장식(canvas, doc):
        canvas.saveState()
        canvas.setFont(보통, 7.5)
        canvas.setFillColor(C["INK3"])
        canvas.drawString(21 * mm, 12 * mm, 꼬리)
        canvas.drawRightString(189 * mm, 12 * mm, str(canvas.getPageNumber()))
        canvas.setStrokeColor(C["RULE"])
        canvas.line(21 * mm, 15.5 * mm, 189 * mm, 15.5 * mm)
        canvas.restoreState()

    doc = BaseDocTemplate(str(pdf_path), pagesize=A4,
                          leftMargin=21 * mm, rightMargin=21 * mm,
                          topMargin=18 * mm, bottomMargin=20 * mm,
                          title=md_path.stem, author="SKN35 2팀")
    doc.addPageTemplates([PageTemplate(id="p", frames=[
        Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")],
        onPage=_장식)])
    doc.build(flow)
    print(f"만듦: {pdf_path}  ({pdf_path.stat().st_size:,} bytes)")
    return pdf_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="마크다운 -> PDF (한글 지원)")
    ap.add_argument("md", help="입력 .md")
    ap.add_argument("pdf", nargs="?", help="출력 .pdf (생략하면 같은 이름)")
    ap.add_argument("--머리말", help="쪽 하단에 넣을 문구")
    a = ap.parse_args(argv)
    try:
        import reportlab  # noqa: F401
    except ImportError:
        sys.exit("reportlab 이 없습니다. 앞에 --with reportlab 을 붙이세요:\n"
                 "  uv run --with reportlab python -m src.md2pdf " + a.md)
    만들기(a.md, a.pdf, a.머리말)


if __name__ == "__main__":
    main()
