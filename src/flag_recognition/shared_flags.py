"""Resolve visual flag-design duplicates in the classifier taxonomy.

The training labels represent jurisdictions, but multiple jurisdictions use
exactly the same national flag. An image alone cannot determine which
administrative territory is depicted. Only combine scores for known
same-design labels, never arbitrary countries with similar-looking flags.
"""
from __future__ import annotations

from dataclasses import dataclass

# Verified duplicate *French tricolor* labels in the current deployed model:
# FR and French overseas territories that fly the national French flag.
# Restrict to well-established cases; some territories have separate local
# flags, so their images should not be silently remapped.
FRENCH_TRICOLOR_CLASSES = frozenset({
    "fr",  # France
    "yt",  # Mayotte
    "mf",  # Saint Martin (French part)
    "bl",  # Saint Barthelemy
    "re",  # Reunion
    "gp",  # Guadeloupe
    "gf",  # French Guiana
})


@dataclass(frozen=True)
class FlagDesignResolution:
    """One shared visual design: score sum is NOT calibrated confidence."""
    code: str
    design_name: str
    combined_score: float
    matched_labels: tuple[str, ...]
    evidence: str


def resolve_french_flag_design(
    predictions: tuple[tuple[str, float], ...] | list[tuple[str, float]],
) -> FlagDesignResolution | None:
    """Canonicalize a near-unanimous French-tricolor prediction cluster.

    This intentionally requires several separately predicted duplicates.
    A single strong prediction of Mayotte is not automatically renamed France;
    the image could show a local flag. A mixed or low-score list remains
    uncertain. This is a *flag design* decision, not a jurisdiction decision.
    """
    if not predictions:
        return None

    valid: list[tuple[str, float]] = []
    for raw_code, raw_score in predictions:
        code = str(raw_code).strip().lower()
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return None
        if not 0.0 <= score <= 1.0:
            return None
        valid.append((code, score))

    # The duplicate pattern must dominate across separate classes, rather than
    # turn a weak top-1 guess into a supposedly certain answer.
    related = [(code, score) for code, score in valid if code in FRENCH_TRICOLOR_CLASSES]
    unrelated = [score for code, score in valid if code not in FRENCH_TRICOLOR_CLASSES]
    related_mass = sum(score for _, score in related)
    if (
        len(related) < 3
        or related_mass < 0.90
        or (unrelated and max(unrelated) >= 0.07)
        or valid[0][0] not in FRENCH_TRICOLOR_CLASSES
    ):
        return None

    return FlagDesignResolution(
        code="fr",
        design_name="tricolore français",
        combined_score=related_mass,
        matched_labels=tuple(code for code, _ in related),
        evidence=(
            "Le modèle a réparti son score entre plusieurs territoires qui "
            "utilisent le même drapeau tricolore français. Une image ne permet "
            "pas de déterminer le territoire exact : la France est affichée "
            "par défaut et peut être corrigée."
        ),
    )
