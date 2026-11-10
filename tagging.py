"""Explain champion mechanics with readable rules and matched text."""

import argparse
import re

from data_io import ABILITIES, load_abilities, resolve_champion

# Rules read the natural descriptions so verbs and their targets stay visible.
REFLEXIVE = r"(?:himself|herself|themself|themselves)"
PULL = (
    r"\b(?:pulls?|pulling|grabs?|grabbing|drags?|dragging)"
    rf"\s+(?!{REFLEXIVE}\b)(?:\w+\s+){{0,3}}(?:enem\w+|opponent\w*)\b|\bpulls? them\b"
)
HOOK = (
    r"\b(?:pulls?|pulling|grabs?|grabbing)\s+(?:(?:an?|the|first)\s+){0,2}"
    r"(?:enemy|opponent)\b|\bpulls? them\b"
)
DISPLACEMENT = (
    r"\b(?:knocks?|knocking|knocked|throws?|throwing)\s+(?:\w+\s+){0,5}"
    r"(?:up|back|away|aside)\b|\b(?:up in|into) the air\b|\bairborne\b"
)
SHIELD_GAIN = (
    r"gains?|gaining|grants?|granting|gives?|giving|receives?|receiving"
    r"|gets?|getting|creates?|creating|provides?|providing|summons?|summoning"
)
RULES = {
    "hook": HOOK,
    "crowd control": (
        r"\b(?:stuns?|stunning|roots?|rooting|rooted|binds?|binding|taunts?|taunting|silences?|silencing|"
        r"slows?|slowing|suppresses?|charms?|immobiliz\w*|fears?|flee)\b"
        rf"|{DISPLACEMENT}|{PULL}"
    ),
    "mobility": (
        r"\b(?:dash(?:es|ing)?|blink(?:s|ing)?|teleport(?:s|ing)?|leap(?:s|ing)?|"
        r"jump(?:s|ing)?|lunge(?:s|ing)?)\b|\bpropels? \w+ to\b"
        rf"|\b(?:launch(?:es|ing)?|pulls?) {REFLEXIVE}\b"
    ),
    "shield": (
        rf"\b(?:{SHIELD_GAIN})(?:\s+[\w'-]+){{0,5}}\s+(?:shield|protective barrier)\b"
        r"|\bshield(?:s|ing)?\s+(?:an? ally|nearby \w+|target \w+|all \w+|them\b|"
        r"himself\b|herself\b|the champion\b|and hastes nearby allies\b)"
        r"|\b(?:protective barrier|magical barrier that blocks)\b"
    ),
    "shield break": (
        r"\b(?:breaks?|breaking|removes?|removing|destroys?|destroying)"
        r"(?:\s+[\w'’-]+){0,4}\s+shields?\b"
    ),
    "heal": (
        r"\bheal(?:s|ing)?\b(?! and shield power)"
        r"|\b(?:restores?|restoring|recovers?|regains?|receives?)\s+"
        r"(?:(?:his|her|their|some|missing|maximum|own)\s+){0,3}health\b"
    ),
    "stealth": r"\b(?:stealth|camouflage|camouflaged|invisib(?:le|ility))\b",
}
TAG_LABELS = tuple(RULES)
PATTERNS = {tag: re.compile(pattern, re.IGNORECASE) for tag, pattern in RULES.items()}
CC_PREVENTION = re.compile(
    r"\b(?:break(?:s|ing)? free of|remov(?:es?|ing)|cleans(?:es?|ing)|immune to)"
    r"(?:\s+(?!(?:and|then)\b)\w+){0,3}\s+$", re.IGNORECASE
)
SELF_TARGET = re.compile(rf"^\s+{REFLEXIVE}\b", re.IGNORECASE)


def tag_evidence(abilities: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """Return the ability slot and matched phrase behind each tag."""
    evidence = {}
    for slot in ABILITIES:
        text = abilities.get(slot, "")
        for tag, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if tag == "crowd control" and (
                    CC_PREVENTION.search(text[:match.start()]) or SELF_TARGET.match(text[match.end():])
                ):
                    continue
                evidence.setdefault(tag, []).append((slot, match.group().strip()))
                break
    return evidence


def get_tags(abilities: dict[str, str], labels: list[str]) -> list[str]:
    evidence = tag_evidence(abilities)
    return [label for label in dict.fromkeys(labels) if label in evidence]


def shared_tags(tags_a: list[str], tags_b: list[str]) -> list[str]:
    common = set(tags_b)
    return [tag for tag in dict.fromkeys(tags_a) if tag in common]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("champion")
    args = parser.parse_args()
    abilities = load_abilities()
    try:
        champion = resolve_champion(args.champion, abilities)
    except ValueError as error:
        parser.error(str(error))
    for tag, matches in tag_evidence(abilities[champion]).items():
        print(f"{tag}: " + ", ".join(f"{slot} ({phrase})" for slot, phrase in matches))
