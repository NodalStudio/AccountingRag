_TERMINATORS = (".", ";", ":", "!", "?")


def join_lines(lines: list[str]) -> str:
    paras: list[str] = []
    current = ""
    for raw in lines:
        line = raw.replace(" ", " ").strip()
        if not line:
            continue
        starts_bullet = line.startswith(("- ", "• "))
        if current and (starts_bullet or current.endswith(_TERMINATORS)):
            paras.append(current)
            current = line
        elif current.endswith("-") and not current.endswith(" -"):
            current = current[:-1] + line          # césure
        elif current:
            current = current + " " + line
        else:
            current = line
    if current:
        paras.append(current)
    return "\n".join(paras)
