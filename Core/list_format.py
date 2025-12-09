# Core/list_format.py

import re

def format_ciiperf_lists(text: str) -> str:
    """
    Formatte les listes produites par le prompt CII concurrence :
    - lignes niveau 1 : "- ..."
    - lignes niveau 2 : "/ ..."

    Transforme en :
    • ...          (niveau 1)
        • ...;     (niveau 2, sauf dernier -> '.')
    """

    lines = text.splitlines()
    out = []
    buffer_lvl2 = []  # stocke les contenus des lignes "/ ..." d'un bloc contigu

    def flush_lvl2():
        """Vide le buffer de niveau 2 en appliquant ';' / '.'."""
        nonlocal buffer_lvl2
        if not buffer_lvl2:
            return
        n = len(buffer_lvl2)
        for i, content in enumerate(buffer_lvl2, start=1):
            # on nettoie la ponctuation finale existante
            c = content.rstrip(" ;.")
            if i < n:
                out.append("    • " + c + ";")
            else:
                out.append("    • " + c + ".")
        buffer_lvl2 = []

    for line in lines:
        # on enlève les espaces (et NBSP) en début de ligne
        stripped = line.lstrip(" \t\u00A0")

        m_lvl1 = re.match(r"^-\s+(?P<txt>.+)$", stripped)
        m_lvl2 = re.match(r"^/\s+(?P<txt>.+)$", stripped)

        if m_lvl1:
            # Avant de commencer un nouveau bloc, on vide les éventuelles lignes lvl2 accumulées
            flush_lvl2()
            content = m_lvl1.group("txt").strip()
            out.append("• " + content)

        elif m_lvl2:
            content = m_lvl2.group("txt").strip()
            buffer_lvl2.append(content)

        else:
            # Ligne normale : on vide d'abord un éventuel bloc niveau 2, puis on recopie la ligne telle quelle
            flush_lvl2()
            out.append(line)

    # Fin du texte : on vide un éventuel dernier bloc niveau 2
    flush_lvl2()

    return "\n".join(out)
