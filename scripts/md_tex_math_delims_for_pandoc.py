#!/usr/bin/env python3
"""
Convert \\[, \\], \\(, \\) to pandoc-safe $$ / $ OUTSIDE fenced ``` blocks only.

Markdown readers parse \\( as literal '(' unless using incompatible extensions.
Bugfix: flushing plain text cannot use an inner-generator `yield`; emit chunks from the outer generator only.
"""

from __future__ import annotations

import re
import sys

FENCE_START = re.compile(r"^\s*`{3,}")


def escape_tex_dollars(tex: str) -> str:
    return tex.replace("$", r"\$")


def convert_plain(body: str) -> str:
    out: list[str] = []
    i = 0
    n = len(body)

    while i < n:
        if body.startswith("\\[", i):
            j = body.find("\\]", i + 2)
            if j == -1:
                out.append(body[i:])
                break
            tex = escape_tex_dollars(body[i + 2 : j].strip())
            out.append(f"\n$${tex}$$\n")
            i = j + 2
            continue

        if body.startswith("\\(", i):
            j = i + 2
            paren = bracket = brace = 0
            ln = len(body)
            matched = False
            while j < ln:
                if j + 2 <= ln and body[j : j + 2] == "\\)":
                    if paren <= 0 and bracket <= 0 and brace <= 0:
                        tex = escape_tex_dollars(body[i + 2 : j].strip())
                        out.append(f"${tex}$")
                        i = j + 2
                        matched = True
                        break
                    j += 2
                    continue
                ch = body[j]
                if ch == "(":
                    paren += 1
                elif ch == ")" and paren > 0:
                    paren -= 1
                elif ch == "[":
                    bracket += 1
                elif ch == "]" and bracket > 0:
                    bracket -= 1
                elif ch == "{":
                    brace += 1
                elif ch == "}" and brace > 0:
                    brace -= 1
                j += 1
            if not matched:
                out.append(body[i])
                i += 1
            continue

        out.append(body[i])
        i += 1

    return "".join(out)


def chunks_md(text: str):
    lines = text.splitlines(keepends=True)
    fence_active = False
    buf_plain: list[str] = []

    def emit_plain():
        nonlocal buf_plain
        if not buf_plain:
            return
        merged = "".join(buf_plain)
        buf_plain = []
        return ("plain", merged)

    for line in lines:
        if FENCE_START.match(line):
            pack = emit_plain()
            if pack:
                yield pack
            fence_active = not fence_active
            yield ("fence", line)
            continue
        if fence_active:
            yield ("fence", line)
        else:
            buf_plain.append(line)
    pack = emit_plain()
    if pack:
        yield pack


def main() -> None:
    text = sys.stdin.read()
    if not text.endswith("\n"):
        text += "\n"
    out_parts: list[str] = []
    for kind, chunk in chunks_md(text):
        if kind == "fence":
            out_parts.append(chunk)
        else:
            out_parts.append(convert_plain(chunk))
    sys.stdout.write("".join(out_parts))


if __name__ == "__main__":
    main()
