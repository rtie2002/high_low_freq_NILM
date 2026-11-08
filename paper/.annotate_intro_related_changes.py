import os
import re
import sys

import fitz


def normalise(value):
    return re.sub(r"[^a-z0-9]+", "", value.lower().replace("–", "-").replace("—", "-"))


def extract_balanced(text, start):
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    raise ValueError("Unclosed DIFadd block")


def change_phrases(tex):
    start = tex.index(r"\begin{document}")
    end = tex.index(r"\end{document}")
    section = tex[start:end]
    phrases = []
    cursor = 0
    token = r"\DIFadd{"
    while True:
        position = section.find(token, cursor)
        if position < 0:
            break
        content_start = position + len(token) - 1
        content = extract_balanced(section, content_start)
        content = re.sub(r"\\cite\{[^}]*\}", "", content)
        content = re.sub(r"\\[A-Za-z]+", "", content)
        words = [normalise(item) for item in re.findall(r"[A-Za-z0-9]+", content)]
        words = [word for word in words if word]
        if len(words) >= 4:
            if len(words) == 4:
                phrases.append(words)
            else:
                phrases.extend(words[index:index + 4] for index in range(len(words) - 3))
        cursor = content_start + len(content) + 1
    hyphen_terms = set()
    for match in re.finditer(r"\\DIFdel\{[-–—]+\}", section):
        window = section[max(0, match.start() - 60):match.end() + 60]
        window = re.sub(r"\\DIFdel(?:begin|end)", "", window)
        window = re.sub(r"\\DIF(?:add|del)\{[^{}]*\}", "", window)
        window = re.sub(r"\\DIF(?:add|del)(?:begin|end)", "", window)
        window = re.sub(r"-\s+", "-", window)
        hyphen_terms.update(
            normalise(term)
            for term in re.findall(r"[A-Za-z]+-[A-Za-z]+", window)
        )
    return phrases, hyphen_terms


def column_words(page):
    midpoint = page.rect.width / 2
    columns = [[], []]
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        value = normalise(text)
        if value:
            columns[x0 >= midpoint].append((value, fitz.Rect(x0, y0, x1, y1)))
    return [sorted(column, key=lambda item: (round(item[1].y0, 1), item[1].x0)) for column in columns]


def locate(phrase, words):
    tokens = [item[0] for item in words]
    size = len(phrase)
    return [index for index in range(len(tokens) - size + 1) if tokens[index:index + size] == phrase]


def main(current_pdf, diff_tex, output_pdf):
    tex = open(diff_tex, encoding="utf-8-sig").read()
    phrases, hyphen_terms = change_phrases(tex)
    document = fitz.open(current_pdf)
    marked = set()
    matched_phrases = 0
    for page_index in range(len(document)):
        page = document[page_index]
        for column in column_words(page):
            for phrase in phrases:
                for location in locate(phrase, column):
                    matched_phrases += 1
                    for _, rect in column[location:location + len(phrase)]:
                        marked.add((page_index, round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)))
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            if normalise(text) in hyphen_terms:
                marked.add((page_index, round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)))
    for page_index in sorted({item[0] for item in marked}):
        page = document[page_index]
        for _, x0, y0, x1, y1 in [item for item in marked if item[0] == page_index]:
            annotation = page.add_highlight_annot(fitz.Rect(x0, y0, x1, y1))
            annotation.set_colors(stroke=(1, 0.93, 0))
            annotation.set_opacity(0.42)
            annotation.update()
    if os.path.exists(output_pdf):
        os.remove(output_pdf)
    document.save(output_pdf, garbage=4, deflate=True)
    print(f"Highlighted {len(marked)} words from {matched_phrases} source-diff phrase matches.")


if __name__ == "__main__":
    main(*sys.argv[1:])
