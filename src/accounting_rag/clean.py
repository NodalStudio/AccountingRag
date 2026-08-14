_TERMINATORS = (".", ";", ":", "!", "?")


def join_lines(lines: list[str]) -> str:
    paras: list[str] = []
    current = ""
    pending_bullet = False
    for raw in lines:
        # Normalise U+00A0 (NBSP) et U+202F (NNBSP) en espace normal
        line = raw.replace(" ", " ").replace(" ", " ").replace(" ", " ").strip()
        if not line:
            continue

        # Cas spécial: ligne vide "- " ou "• " (puce seule)
        if line in ("-", "•"):
            pending_bullet = True
            continue

        # Si une puce était en attente, on la préfixe
        if pending_bullet:
            line = "- " + line
            pending_bullet = False

        starts_bullet = line.startswith(("- ", "• "))
        if current and (starts_bullet or current.endswith(_TERMINATORS)):
            paras.append(current)
            current = line
        elif current:
            # Ne pas ajouter d'espace après un tiret de fin
            if current.endswith("-"):
                current = current + line
            else:
                current = current + " " + line
        else:
            current = line

    if pending_bullet and current:
        current = current + " -"
    if current:
        paras.append(current)
    return "\n".join(paras)
