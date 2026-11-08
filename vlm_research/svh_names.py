"""Public LLaVA variant names for new experiments; preserve frozen runner labels.

No imports from the inference stack, no GPU dependencies, and no mutation of
historical result/checkpoint condition strings.
"""

LLAVA_SVH = "llava-SVH"
LLAVA_SVH_R = "llava-SVH-R"

_CANONICAL = (LLAVA_SVH, LLAVA_SVH_R)

# These aliases denote the *physical* frozen variants, not dense controls or
# separate Middle9/Middle10 ablations.
_ALIASES = {
    "C3-V2": LLAVA_SVH,
    "Physical-C3": LLAVA_SVH,
    "C3-V2+L11-L24": LLAVA_SVH_R,
    "Physical-L11-L24": LLAVA_SVH_R,
}

# Different historical runners used different internal condition strings.
_FROZEN = {
    LLAVA_SVH: {
        "gqa": "C3-V2",
        "physical": "Physical-C3",
    },
    LLAVA_SVH_R: {
        "gqa": "C3-V2+L11-L24",
        "physical": "Physical-L11-L24",
    },
}

_LAYERS = {
    LLAVA_SVH: {"writing": tuple(range(20)), "exit_before": 20, "reaccess": (), "answer_attention_skip": (2, 3, 4, 5)},
    LLAVA_SVH_R: {"writing": tuple(range(20)), "exit_before": 25, "reaccess": tuple(range(11, 25)), "answer_attention_skip": (2, 3, 4, 5)},
}


def canonical_name(name: str) -> str:
    """Return the public name; reject unknown and non-equivalent ablations."""
    if name in _CANONICAL:
        return name
    try:
        return _ALIASES[name]
    except KeyError as exc:
        raise ValueError(f"Not a frozen LLaVA SVH variant: {name!r}") from exc


def frozen_name(name: str, *, runner: str = "gqa") -> str:
    """Translate a public name only at a *new* runner entry point."""
    variant = canonical_name(name)
    try:
        return _FROZEN[variant][runner]
    except KeyError as exc:
        raise ValueError(f"Unknown frozen runner naming scheme: {runner!r}") from exc


def layer_policy(name: str) -> dict:
    """Descriptive 0-based layer specification; does not execute inference."""
    return dict(_LAYERS[canonical_name(name)])
