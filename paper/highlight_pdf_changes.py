"""Highlight current PDF text against an explicit original PDF; preserve layout."""
import argparse
import difflib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import fitz


def tokens(doc):
    result = []
    for pi, page in enumerate(doc):
        # PDF content order retains the article's columns and full-width floats.
        for block in page.get_text('rawdict')['blocks']:
            if 'lines' not in block:
                continue
            chars = []
            for line in block['lines']:
                for span in line['spans']:
                    for c in span['chars']:
                        if c['bbox'][1] < 30:  # running page number
                            continue
                        for letter in unicodedata.normalize('NFKC', c['c']):
                            chars.append((letter, pi, c['bbox']))
                # Remove discretionary line-end hyphens and join the fragments.
                if chars and chars[-1][0] == '-':
                    chars.pop()
                else:
                    chars.append((' ', pi, (0, 0, 0, 0)))
            text = ''.join(c[0] for c in chars)
            for m in re.finditer(r'\w+(?:[-–—]\w+)*|[^\w\s]', text):
                raw = m.group()
                key = re.sub('[-–—]', '', raw)
                if not key:
                    key = raw
                result.append((key, raw, chars[m.start():m.end()]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original')
    parser.add_argument('current')
    parser.add_argument('output')
    args = parser.parse_args()
    original, current = fitz.open(args.original), fitz.open(args.current)
    before, after = tokens(original), tokens(current)
    matcher = difflib.SequenceMatcher(None, [t[0] for t in before], [t[0] for t in after], autojunk=False)
    changed = set()
    changes = []
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag in ('insert', 'replace'):
            changed.update(range(c, d))
            changes.append((tag, a, b, c, d))
        elif tag == 'equal':
            for oi, ni in zip(range(a, b), range(c, d)):
                if before[oi][1] != after[ni][1]:
                    changed.add(ni)
    # Exact text moved to another location is not new wording.
    old_keys = [t[0] for t in before]
    for tag, a, b, c, d in changes:
        if d-c < 6:
            continue
        sub = difflib.SequenceMatcher(None, old_keys, [t[0] for t in after[c:d]], autojunk=False)
        for match in sub.get_matching_blocks():
            if match.size >= 6:
                for offset in range(match.size):
                    oi, ni = match.a+offset, c+match.b+offset
                    if before[oi][1] == after[ni][1]:
                        changed.discard(ni)
    lines = defaultdict(list)
    for index in sorted(changed):
        for letter, pi, box in after[index][2]:
            if not letter.isspace():
                lines[(pi, round(box[1], 1), round(box[3], 1))].append(fitz.Rect(box))
    for pi in range(len(current)):
        page = current[pi]
        for (p, _, _), rects in lines.items():
            if p != pi:
                continue
            rects.sort(key=lambda r:r.x0)
            groups = []
            for rect in rects:
                if groups and rect.x0 - groups[-1].x1 < 3:
                    groups[-1] |= rect
                else:
                    groups.append(rect)
            for rect in groups:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1, .93, 0))
                annot.set_opacity(.4)
                annot.update()
    target = Path(args.output)
    temporary = target.with_name(target.stem + '_building.pdf')
    current.save(str(temporary), garbage=4, deflate=True)
    temporary.replace(target)
    verify = fitz.open(str(target))
    assert len(verify) == len(current)
    source_check = fitz.open(args.current)
    assert all(a.get_text() == b.get_text() for a,b in zip(source_check, verify))
    print(f'Baseline: {Path(args.original).resolve()}')
    print(f'{len(changed)} changed text tokens; {len(current)} pages; text/layout preserved.')


if __name__ == '__main__':
    main()
