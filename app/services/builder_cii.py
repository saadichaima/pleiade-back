# app/services/builder_cii.py
from typing import Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from Core import rag_cii
from Core import rag
from app.services.web_scraper import scrape_website, validate_url


def _truncate_content(text: str, max_words: int = 500) -> str:
    """Tronque le contenu scrapé à max_words pour contrôler la taille du prompt."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


def _scrape_competitors_parallel(
    concurrents_simpl: list,
    max_words_per_competitor: int = 500,
    timeout_per_site: int = 10,
) -> list:
    """
    Scrape les sites web des concurrents en parallèle.
    Ajoute le champ 'website_content' à chaque dict concurrent.
    """
    tasks = []
    for idx, cpt in enumerate(concurrents_simpl):
        url = (cpt.get("website") or "").strip()
        if url and validate_url(url):
            tasks.append((idx, url))
        else:
            cpt["website_content"] = ""

    if not tasks:
        print("[SCRAPE COMPETITORS] Aucune URL valide à scraper")
        return concurrents_simpl

    print(f"[SCRAPE COMPETITORS] Scraping de {len(tasks)} site(s) concurrent(s) en parallèle...")

    results = {}
    with ThreadPoolExecutor(max_workers=min(len(tasks), 5)) as executor:
        future_to_idx = {
            executor.submit(scrape_website, url, 1, timeout_per_site): idx
            for idx, url in tasks
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                raw_content = future.result()
                results[idx] = _truncate_content(raw_content, max_words_per_competitor)
            except Exception as e:
                print(f"[SCRAPE COMPETITORS] Erreur scraping concurrent {idx}: {e}")
                results[idx] = ""

    for idx, url in tasks:
        concurrents_simpl[idx]["website_content"] = results.get(idx, "")

    scraped_count = sum(1 for v in results.values() if v)
    print(f"[SCRAPE COMPETITORS] {scraped_count}/{len(tasks)} site(s) scrapé(s) avec succès")

    return concurrents_simpl


def build_sections_cii(
    index_client_pack,   # (index, chunks, vectors) pour docs techniques
    index_mix_pack,      # (index, chunks, vectors) mix client+admin
    index_admin_pack,    # (index, chunks, vectors) pour docs admin
    info,
    axes_cibles,
    concurrents,
    total_heures: str = "",
    contexte_societe: str = "",
    secteur: str = "",
    visee_generale: str = "",
    performance_type: str = "",
    doc_complete: bool = False,
    emit: Optional[Callable] = None,
) -> Dict[str, Any]:
    (i,  c,  v)  = index_client_pack
    (im, cm, vm) = index_mix_pack
    (ia, ca, va) = index_admin_pack

    period = (
        f"{info.date_debut or ''} — {info.date_fin or ''}".strip(" —")
        if (info.date_debut or info.date_fin)
        else str(info.annee)
    )

    # Présentation prioritairement basée sur les docs administratifs
    src_index  = ia or im
    src_chunks = ca or cm
    src_vecs   = va or vm
    presentation = rag_cii.gen_presentation(
        src_index,
        src_chunks,
        src_vecs,
        societe=info.societe,
        annee=info.annee,
        referent_nom=info.responsable_innovation,
        referent_titre=info.titre_resp,
        referent_tel=info.telephone,
        referent_email=info.email,
        site_web=info.site_web or "",
        contexte_societe=contexte_societe,
        secteur=secteur,
        performance_type=performance_type,
    )

    contexte = rag_cii.gen_contexte(
        im, cm, vm,
        annee=info.annee,
        visee_generale=visee_generale,
        performance_type=performance_type,
    )

    # --- Concurrents : simplification + liste lisible pour le prompt ----------
    concurrents_simpl = [
        {
            "name": x.get("name"),
            "website": x.get("site") or "",
            "axes": x.get("axes") or [],
            "weakness": x.get("weakness") or "",
            "client_advantage": x.get("client_advantage") or "",
        }
        for x in (concurrents or [])
    ]

    # --- Scraper les sites web des concurrents en parallèle ---
    if emit:
        emit("scraping_competitors", "Analyse des sites web concurrents", 42)
    concurrents_simpl = _scrape_competitors_parallel(concurrents_simpl)

    lines = []
    for cpt in concurrents_simpl:
        name = (cpt.get("name") or "").strip()
        if not name:
            continue
        website = (cpt.get("website") or "").strip()
        axes_str = ", ".join(cpt.get("axes") or [])
        weakness = (cpt.get("weakness") or "").strip()
        adv = (cpt.get("client_advantage") or "").strip()
        web_content = (cpt.get("website_content") or "").strip()

        line = f"- {name}"
        if website:
            line += f" ({website})"
        if axes_str:
            line += f" — axes visés: {axes_str}"
        if weakness:
            line += f"\n  Limite du concurrent: {weakness}"
        if adv:
            line += f"\n  Avantage du projet client: {adv}"
        if web_content:
            line += f"\n  Contenu extrait du site web:\n  {web_content}"
        lines.append(line)

    liste_concurrents = "\n".join(lines) if lines else "- Aucun concurrent renseigné."

    analyse = rag_cii.gen_analyse(
        im, cm, vm,
        concurrents=concurrents_simpl,
        axes_cibles=axes_cibles,
        performance_type=performance_type,
        liste_concurrents=liste_concurrents,
    )



    performances = rag_cii.gen_performances(
        i, c, v,
        projet=info.projet_name,
        performance_type=performance_type,
    )
    demarche = rag_cii.gen_demarche_annee(
        i, c, v,
        annee=info.annee,
        performance_type=performance_type,
    )

    if not doc_complete:
        demarche = rag_cii.evaluateur_travaux(demarche)
        demarche = rag.wrap_questions_rouge(demarche)


    resultats = rag_cii.gen_resultats_annee(
        im, cm, vm,
        annee=info.annee,
        performance_type=performance_type,
    )

    rh_intro = rag_cii.gen_rh_intro(
        im, cm, vm,
        annee=info.annee,
        total_heures=total_heures,
        performance_type=performance_type,
    )

    biblio_intro = rag_cii.get_biblio_intro()

    # Générer le résumé en se basant sur les sections déjà générées
    resume = rag_cii.gen_resume_from_sections(
        sections={
            "presentation": presentation,
            "contexte": contexte,
            "analyse": analyse,
            "performances": performances,
            "demarche": demarche,
            "resultats": resultats,
            "rh_intro": rh_intro,
        },
        projet=info.projet_name,
        annee=info.annee,
        period=period,
        performance_type=performance_type,
    )

    # Générer les données du tableau comparatif
    # Basé sur les sections travaux (demarche) et concurrence (analyse)
    tableau_comparatif = rag_cii.generate_tableau_comparatif(
        i, c, v,
        societe=info.societe,
        projet=info.projet_name,
        competitors=concurrents_simpl,
        section_travaux=demarche,
        section_concurrence=analyse,
    )

    return {
        "presentation": presentation,    # 👉 d.cii.presentation
        "resume": resume,
        "contexte": contexte,
        "analyse": analyse,
        "performances": performances,
        "demarche": demarche,
        "resultats": resultats,
        "rh_intro": rh_intro,
        "biblio_intro": biblio_intro,
        "tableau_comparatif": tableau_comparatif,  # 👉 Données pour le tableau
    }
