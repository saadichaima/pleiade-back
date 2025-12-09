# app/services/builder_cii.py
from typing import Dict, Any
from Core import rag_cii
from Core import rag
from Core.list_format import format_ciiperf_lists  

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


    resume = rag_cii.gen_resume(
        i, c, v,
        projet=info.projet_name,
        annee=info.annee,
        period=period,
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

    lines = []
    for cpt in concurrents_simpl:
        name = (cpt.get("name") or "").strip()
        if not name:
            continue
        website = (cpt.get("website") or "").strip()
        axes_str = ", ".join(cpt.get("axes") or [])
        weakness = (cpt.get("weakness") or "").strip()
        adv = (cpt.get("client_advantage") or "").strip()

        line = f"- {name}"
        if website:
            line += f" ({website})"
        if axes_str:
            line += f" — axes visés: {axes_str}"
        if weakness:
            line += f"\n  Limite du concurrent: {weakness}"
        if adv:
            line += f"\n  Avantage du projet client: {adv}"
        lines.append(line)

    liste_concurrents = "\n".join(lines) if lines else "- Aucun concurrent renseigné."

    analyse = rag_cii.gen_analyse(
        im, cm, vm,
        concurrents=concurrents_simpl,
        axes_cibles=axes_cibles,
        performance_type=performance_type,
        liste_concurrents=liste_concurrents,
    )
    analyse = format_ciiperf_lists(analyse)


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
    }
