_TERMINATORS = (".", ";", ":", "!", "?")


def join_lines(lines: list[str]) -> str:
    paras: list[str] = []
    current = ""
    pending_bullet = False
    
    for raw in lines:
        # Normalise U+00A0 (NBSP) et U+202F (NNBSP) en espace normal (F9-bis)
        line = raw.replace(" ", " ").replace(" ", " ").strip()
        if not line:
            continue

        # Cas spécial: ligne vide "- " ou "• " (puce seule) (F7-bis)
        if line in ("-", "•"):
            if pending_bullet:
                # Une autre puce arrive: émettre la précédente comme paragraphe
                paras.append("-")
            else:
                # Première puce: la mettre en attente
                if current:
                    paras.append(current)
                    current = ""
            pending_bullet = True
            continue

        # Si une puce était en attente, on la préfixe à la ligne courante
        if pending_bullet:
            line = "- " + line
            pending_bullet = False

        # Si la ligne courante commence par une puce, c'est un nouveau paragraphe
        starts_bullet = line.startswith(("- ", "• "))
        if current and (starts_bullet or current.endswith(_TERMINATORS)):
            paras.append(current)
            current = line
        elif current:
            # F3-bis: Deux cas distincts en fin de ligne
            if current.endswith("-") and not current.endswith(" -"):
                # Tiret attaché au mot (tirets 214-): joindre SANS espace
                current = current + line
            else:
                # Tiret séparateur (espace+tiret): joindre AVEC espace
                current = current + " " + line
        else:
            current = line

    # En fin de liste, émettre la puce en attente comme paragraphe
    if pending_bullet:
        paras.append("-")
    
    if current:
        paras.append(current)
    return "\n".join(paras)
