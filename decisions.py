"""
decisions.py — Protokoły wyboru ostatecznej decyzji po debacie.

Inspirowane: Kaesberg et al. 2025 "Voting or Consensus? Decision-Making in
Multi-Agent Debate" (arXiv:2502.19130).

Trzy protokoły:
    judge     — niezależny sędzia podsumowuje i wybiera (oryginalny baseline)
    voting    — każdy agent proponuje finalną odpowiedź, potem głosują;
                wygrywa odpowiedź z największą liczbą głosów
    consensus — agenci iteracyjnie zbliżają się do wspólnej odpowiedzi;
                kończymy gdy próg zgody zostanie osiągnięty

Wnioski z paperu (dla zadań QA z Llama 3 8B):
    • voting wygrywa w zadaniach reasoningowych (+13.2%)
    • consensus wygrywa w zadaniach wiedzowych (+2.8%)
"""

from collections import Counter

from agents import _generate


# =============================================================================
# 1. JUDGE — pojedynczy sędzia podsumowuje debatę
# =============================================================================
def judge_decision(agents, judge, debate_log, topic, config):
    """Klasyczny baseline — sędzia czyta debatę i wydaje werdykt."""
    print(f"\n{'='*60}")
    print("  PROTOKÓŁ: JUDGE — sędzia podsumowuje")
    print(f"{'='*60}")

    verdict = judge.summarize(debate_log, topic, config)
    print(f"\n[Sędzia]:\n{verdict}")
    return verdict


# =============================================================================
# 2. VOTING — agenci proponują finalne odpowiedzi i głosują
# =============================================================================
def voting_decision(agents, judge, debate_log, topic, config):
    """Każdy agent: (1) formułuje swoją finalną odpowiedź, (2) głosuje.

    Wygrywa odpowiedź z największą liczbą głosów. Remis → wygrywa pierwsza.
    """
    print(f"\n{'='*60}")
    print("  PROTOKÓŁ: VOTING — każdy agent głosuje")
    print(f"{'='*60}")

    # Krok 1: każdy agent formułuje swoją finalną propozycję
    print("\n--- Krok 1: finalne propozycje agentów ---")
    proposals = {}
    transcript = _format_transcript(debate_log, topic)

    for agent in agents:
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {
                "role": "user",
                "content": (
                    f"{transcript}\n\n"
                    "Na podstawie powyższej debaty sformułuj swoją FINALNĄ odpowiedź "
                    "w jednym zdaniu. Odpowiedz tylko tym zdaniem, bez wyjaśnień."
                ),
            },
        ]
        proposal = _generate(agent.model, agent.tokenizer, messages, config)
        proposals[agent.name] = proposal
        print(f"\n[{agent.name}]: {proposal}")

    # Krok 2: każdy agent głosuje na jedną z propozycji
    print("\n--- Krok 2: głosowanie ---")
    options_str = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(proposals.values()))
    votes = []

    for agent in agents:
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Temat: {topic}\n\n"
                    f"Propozycje finalne:\n{options_str}\n\n"
                    "Zagłosuj na NAJLEPSZĄ odpowiedź. Odpowiedz wyłącznie liczbą "
                    f"(1-{len(proposals)})."
                ),
            },
        ]
        vote_text = _generate(agent.model, agent.tokenizer, messages, config)
        # Parsujemy pierwszą cyfrę z odpowiedzi
        vote_idx = _parse_vote(vote_text, num_options=len(proposals))
        chosen = list(proposals.values())[vote_idx]
        votes.append(chosen)
        print(f"  [{agent.name}] głosuje na: opcja {vote_idx + 1}")

    # Krok 3: zliczanie głosów
    tally = Counter(votes)
    print(f"\n--- Wyniki głosowania ---")
    for option, count in tally.most_common():
        print(f"  {count} głos(y): {option[:80]}...")

    winner = tally.most_common(1)[0][0]
    print(f"\n[ZWYCIĘZCA]:\n{winner}")
    return winner


def _parse_vote(text, num_options):
    """Wyciąga numer głosu z odpowiedzi modelu (1-indexed → 0-indexed).

    Jeśli nie znajdzie poprawnego numeru, zwraca 0.
    """
    for char in text.strip():
        if char.isdigit():
            num = int(char)
            if 1 <= num <= num_options:
                return num - 1
    return 0


# =============================================================================
# 3. CONSENSUS — agenci iteracyjnie zbliżają się do zgody
# =============================================================================
def consensus_decision(agents, judge, debate_log, topic, config):
    """Agenci formułują wspólne stanowisko. Sprawdzamy próg zgody.

    Param z config: consensus_threshold (0.5 = majority, 0.66 = supermajority, 1.0 = unanimity)
    Param z config: max_consensus_rounds (ile dodatkowych rund gdy brak zgody)
    """
    threshold = config.get("consensus_threshold", 0.66)
    max_rounds = config.get("max_consensus_rounds", 3)

    print(f"\n{'='*60}")
    print(f"  PROTOKÓŁ: CONSENSUS — próg zgody: {threshold:.0%}")
    print(f"{'='*60}")

    transcript = _format_transcript(debate_log, topic)
    current_proposal = None

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Runda konsensusu {round_num}/{max_rounds} ---")

        # Krok A: pierwszy agent (lub aktualne stanowisko) jako punkt wyjścia
        if current_proposal is None:
            messages = [
                {"role": "system", "content": agents[0].system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{transcript}\n\n"
                        "Sformułuj propozycję wspólnego stanowiska wszystkich agentów "
                        "w jednym zdaniu, które mogłoby ich pogodzić."
                    ),
                },
            ]
            current_proposal = _generate(
                agents[0].model, agents[0].tokenizer, messages, config
            )
            print(f"\n[Propozycja od {agents[0].name}]: {current_proposal}")

        # Krok B: każdy agent ocenia czy się zgadza
        agreements = []
        for agent in agents:
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Temat: {topic}\n\n"
                        f"Proponowane wspólne stanowisko:\n\"{current_proposal}\"\n\n"
                        "Czy ZGADZASZ się z tym stanowiskiem? Odpowiedz tylko TAK lub NIE."
                    ),
                },
            ]
            response = _generate(agent.model, agent.tokenizer, messages, config)
            agrees = _parse_yes_no(response)
            agreements.append(agrees)
            print(f"  [{agent.name}]: {'TAK' if agrees else 'NIE'}")

        agreement_ratio = sum(agreements) / len(agreements)
        print(f"  → Zgoda: {sum(agreements)}/{len(agreements)} ({agreement_ratio:.0%})")

        if agreement_ratio >= threshold:
            print(f"\n[KONSENSUS OSIĄGNIĘTY]:\n{current_proposal}")
            return current_proposal

        # Krok C: brak konsensusu → agenci, którzy się nie zgadzają, modyfikują
        print("  Brak konsensusu — modyfikujemy stanowisko...")
        dissenters = [a for a, ok in zip(agents, agreements) if not ok]
        if dissenters:
            modifier = dissenters[0]
            messages = [
                {"role": "system", "content": modifier.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Temat: {topic}\n\n"
                        f"Aktualne stanowisko:\n\"{current_proposal}\"\n\n"
                        "Zmodyfikuj je tak, aby było bardziej akceptowalne dla wszystkich. "
                        "Odpowiedz jednym zdaniem."
                    ),
                },
            ]
            current_proposal = _generate(
                modifier.model, modifier.tokenizer, messages, config
            )
            print(f"  [{modifier.name} proponuje]: {current_proposal}")

    # Brak konsensusu po wszystkich rundach — zwracamy ostatnią propozycję
    print(f"\n[BRAK KONSENSUSU po {max_rounds} rundach — zwracamy ostatnią propozycję]:")
    print(current_proposal)
    return current_proposal


def _parse_yes_no(text):
    """Wyciąga TAK/NIE z odpowiedzi modelu (PL+EN). Domyślnie False."""
    t = text.strip().lower()
    if t.startswith("tak") or t.startswith("yes"):
        return True
    if t.startswith("nie") or t.startswith("no"):
        return False
    # Fallback — szukamy w pierwszych 30 znakach
    head = t[:30]
    if "tak" in head or "yes" in head:
        return True
    return False


# =============================================================================
# Helpery
# =============================================================================
def _format_transcript(debate_log, topic):
    """Formatuje log debaty do tekstu dla promptów."""
    out = f"Temat debaty: {topic}\n\n"
    for entry in debate_log:
        out += f"[Runda {entry['round']}] {entry['agent']}: {entry['text']}\n\n"
    return out


# Mapowanie nazw z configu na funkcje
DECISIONS = {
    "judge": judge_decision,
    "voting": voting_decision,
    "consensus": consensus_decision,
}
