"""Flag-first Streamlit experience. Advanced reports remain in the legacy app."""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys

import pycountry
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from .flag_matcher import FlagMatch, build_gallery, rank_reference_flags
from .quick_facts import CountryFacts, fetch_country_facts
from .taxonomy import country_name_from_code

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "artifacts/models/worldwide_mobilenet_v3_small.pt"
CODES = tuple(sorted({country.alpha_2.lower() for country in pycountry.countries} | {"xk"}))

EXAMPLES = {
    "France": "fr",
    "Côte d'Ivoire": "ci",
    "Irlande": "ie",
    "Japon": "jp",
    "Brésil": "br",
    "Allemagne": "de",
    "Maroc": "ma",
}


@st.cache_resource(ttl=86400, show_spinner=False)
def _gallery() -> dict:
    return build_gallery(CODES, max_workers=20)


@st.cache_resource(show_spinner=False)
def _classifier():
    from .inference import load_inference_bundle
    return load_inference_bundle(MODEL_FILE, device="cpu")


@st.cache_data(ttl=43200, show_spinner=False)
def _facts(country_code: str) -> CountryFacts:
    return fetch_country_facts(country_code)


def _read_image(data: bytes) -> Image.Image:
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("Image trop volumineuse (maximum 10 Mo).")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > 20_000_000:
                raise ValueError("Dimensions trop grandes (maximum 20 mégapixels).")
            image = ImageOps.exif_transpose(source)
            image.load()
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Fichier image illisible. Utilisez PNG, JPEG ou WebP.") from exc


def _predict(image: Image.Image) -> dict:
    match = FlagMatch(None, None, None, (), False)
    try:
        gallery = _gallery()
        match = rank_reference_flags(image, gallery)
    except Exception:
        gallery = {}

    if match.accepted:
        return {
            "code": match.code,
            "mode": "reference",
            "candidates": match.matches,
            "score": None,
            "gallery_count": len(gallery),
            "warning": None,
        }

    try:
        from .inference import predict_robust
        prediction = predict_robust(image, _classifier())
        top5 = prediction.top5
        confidence = prediction.top1_confidence
        gap = confidence - top5[1][1] if len(top5) > 1 else 0.0
        is_plausible = prediction.is_known and gap >= 0.08

        # Network-dependent references can be incomplete; do not convert their
        # distance to a probability or silently override a model prediction.
        return {
            "code": prediction.top1_country if is_plausible else None,
            "mode": "model" if is_plausible else "uncertain",
            "candidates": top5,
            "score": confidence,
            "gallery_count": len(gallery),
            "warning": (
                "Cette image reste difficile à distinguer automatiquement. "
                "Choisissez le pays avant de consulter la fiche."
                if not is_plausible else
                "Résultat du classificateur : à confirmer, surtout pour une photo "
                "prise dans un environnement complexe."
            ),
        }
    except Exception:
        return {
            "code": None,
            "mode": "unavailable",
            "candidates": match.matches,
            "score": None,
            "gallery_count": len(gallery),
            "warning": (
                "Le classificateur n'a pas pu démarrer. "
                "Vous pouvez choisir le pays manuellement."
            ),
        }


def _format_int(value: int | float | None) -> str:
    if value is None:
        return "Indisponible"
    return f"{value:,.0f}".replace(",", " ")


def _render_fact_sheet(code: str) -> None:
    try:
        facts = _facts(code.lower())
    except Exception as exc:
        st.error("Impossible de charger les informations du pays.")
        st.caption("Les services de données sont peut-être momentanément indisponibles.")
        return

    st.divider()
    col_header, col_flag = st.columns([4, 1], vertical_alignment="center")
    with col_header:
        st.subheader(facts.name)
        st.caption(f"Code ISO : {facts.code}  ·  Fiche récupérée le {facts.retrieved_at}")
    with col_flag:
        if facts.flag_url:
            st.image(facts.flag_url, width=120)

    col1, col2, col3 = st.columns(3)
    col1.metric("Capitale", facts.capital or "Indisponible")
    col2.metric("Superficie", f"{_format_int(facts.area_km2)} km²" if facts.area_km2 is not None else "Indisponible")
    col3.metric("Population", _format_int(facts.population))
    if facts.population is not None:
        year = facts.population_year or "année non précisée"
        st.caption(f"Population — source : {facts.population_source}, référence : {year}.")
        if facts.population_source_url:
            st.markdown(f"[Consulter la source de population]({facts.population_source_url})")
    if facts.warning:
        st.info(facts.warning)
    st.caption(
        "Capitale, superficie et drapeau : REST Countries. "
        "Les nombres de population ne sont pas des estimations en direct."
    )

    st.download_button(
        "Exporter la fiche simple (JSON)",
        data=__import__("json").dumps(
            {
                "code": facts.code,
                "country": facts.name,
                "capital": facts.capital,
                "area_km2": facts.area_km2,
                "population": facts.population,
                "population_reference_year": facts.population_year,
                "population_source": facts.population_source,
                "population_source_url": facts.population_source_url,
                "retrieved_at": facts.retrieved_at,
            },
            ensure_ascii=False,
            indent=2,
        ),
        file_name=f"flag_intelligence_{facts.code.lower()}.json",
        mime="application/json",
        use_container_width=False,
    )


def _render_prediction(result: dict) -> None:
    mode = result.get("mode")
    code = result.get("code")
    if mode == "reference":
        st.success(f"Drapeau reconnu : {country_name_from_code(code)}")
        st.caption(
            "Correspondance visuelle avec les drapeaux de référence. "
            "Il ne s'agit pas d'une probabilité statistique."
        )
    elif mode == "model":
        st.warning(f"Pays probable : {country_name_from_code(code)} · à confirmer")
        if result.get("score") is not None:
            st.caption(f"Score interne du modèle : {result['score']:.1%} (non calibré pour les photos réelles).")
    else:
        st.warning("Identification incertaine : aucune réponse forcée.")

    if result.get("warning"):
        st.caption(result["warning"])

    candidates = result.get("candidates", ())
    if candidates:
        with st.expander("Voir les autres pays possibles"):
            for candidate, score in candidates[:5]:
                name = country_name_from_code(candidate)
                if mode == "reference":
                    st.write(f"{name} · écart visuel {score:.3f} (plus petit = meilleur)")
                else:
                    st.write(f"{name} · score modèle {score:.1%}")

    options = [""] + list(CODES)
    # Index is updated when the image changes; no stale country from another upload.
    selected = st.selectbox(
        "Confirmer ou corriger le pays",
        options=options,
        index=options.index(code) if code in options else 0,
        format_func=lambda c: "Sélectionner un pays..." if not c else f"{country_name_from_code(c)} ({c.upper()})",
        key=f"fi-confirm-{st.session_state.get('fi_current_digest', 'empty')}",
    )
    if selected:
        _render_fact_sheet(selected)


def run() -> None:
    st.set_page_config(
        page_title="Flag Intelligence | Reconnaissance de drapeaux",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #f7f9fc; color: #17233d; }
        .block-container { max-width: 1120px; padding-top: 2rem; }
        .fi-eyebrow { font-weight: 700; letter-spacing: .14em; color: #3660a9;
                      text-transform: uppercase; font-size: .78rem; }
        .fi-intro { color: #56657a; font-size: 1.06rem; }
        div[data-testid="stMetric"] {background: white; padding: 1rem;
            border: 1px solid #e7ecf5; border-radius: 14px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fi-eyebrow">Flag Intelligence · Vision par ordinateur</div>', unsafe_allow_html=True)
    st.title("Reconnaître un drapeau. Découvrir son pays.")
    st.markdown(
        '<div class="fi-intro">Importez une image : le système identifie le drapeau, '
        'puis affiche une fiche factuelle courte et sourcée. '
        'Les rapports avancés ne font pas partie de ce parcours.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.subheader("1 · Votre drapeau")
        tab_upload, tab_photo, tab_example = st.tabs(
            ["Importer", "Appareil photo", "Exemple de démo"]
        )
        uploaded_bytes = None
        with tab_upload:
            uploaded = st.file_uploader(
                "Photo ou image du drapeau",
                type=["png", "jpg", "jpeg", "webp"],
                help="Maximum 10 Mo. Les images nettes et centrées donnent les meilleurs résultats.",
            )
            if uploaded is not None:
                uploaded_bytes = uploaded.getvalue()
        with tab_photo:
            camera = st.camera_input("Photographier le drapeau")
            if camera is not None and uploaded_bytes is None:
                uploaded_bytes = camera.getvalue()
        with tab_example:
            example_name = st.selectbox("Drapeau de démonstration", list(EXAMPLES))
            if st.button("Charger ce drapeau", type="secondary"):
                try:
                    import requests
                    response = requests.get(
                        f"https://flagcdn.com/w320/{EXAMPLES[example_name]}.png",
                        timeout=7,
                    )
                    response.raise_for_status()
                    st.session_state.fi_example_bytes = response.content
                except Exception:
                    st.error("L'image de démonstration n'est pas accessible.")
            if uploaded_bytes is None:
                uploaded_bytes = st.session_state.get("fi_example_bytes")

        if uploaded_bytes:
            try:
                image = _read_image(uploaded_bytes)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.image(image, caption="Image à analyser", use_container_width=True)
        else:
            st.info("Ajoutez une image pour démarrer la reconnaissance.")
            image = None

    with right:
        st.subheader("2 · Reconnaissance")
        if image is None:
            st.write("L'identification et la fiche du pays apparaîtront ici.")
            st.caption("La reconnaissance automatique peut être incertaine pour les drapeaux très similaires ou les scènes complexes.")
        else:
            digest = sha256(uploaded_bytes).hexdigest()[:20]
            st.session_state.fi_current_digest = digest
            if st.button("Identifier le drapeau", type="primary", use_container_width=True):
                with st.spinner("Analyse de l'image..."):
                    st.session_state.fi_result = {"hash": digest, "result": _predict(image)}
            payload = st.session_state.get("fi_result", {})
            if payload.get("hash") == digest:
                _render_prediction(payload["result"])
            else:
                st.caption("Cliquez sur « Identifier le drapeau » pour lancer l'analyse.")

    st.divider()
    with st.expander("À propos des performances et des données"):
        st.write(
            "Deux méthodes complémentaires sont utilisées : une comparaison visuelle "
            "prudente avec des drapeaux de référence et, en secours, un classificateur "
            "MobileNetV3-Small. Les résultats incertains doivent être confirmés. "
            "La population provient de la Banque mondiale (dernière année observée "
            "disponible) ou, à défaut, de REST Countries avec une réserve explicite. "
            "Les drapeaux de la démo servent à vérifier le fonctionnement, "
            "pas à mesurer les performances sur des photographies indépendantes."
        )
    st.caption("Flag Intelligence · Application de démonstration · Reconnaissance prioritaire")
