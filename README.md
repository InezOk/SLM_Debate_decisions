# Multi-Agent Debate — Decision Protocols

Rozszerzenie repozytorium [SLM_Debate](https://github.com/InezOk/SLM_Debate) o **protokoły decyzyjne**:
po debacie agenci nie tylko są oceniani przez sędziego, ale mogą też **głosować** lub **dążyć do konsensusu**.

Inspiracja:
> Kaesberg et al. 2025. *Voting or Consensus? Decision-Making in Multi-Agent Debate.*
> [arXiv:2502.19130](https://arxiv.org/abs/2502.19130)

## Szybki start

```bash
pip install -r requirements.txt
python main.py
```

## Struktura projektu

```
config.yaml        ← Wszystkie parametry (prompty, model, architektura, protokół decyzyjny)
main.py            ← Punkt wejścia
agents.py          ← Klasa Agent i Judge
architectures.py   ← 3 architektury wymiany: round_robin / relay / free_for_all
decisions.py       ← 3 protokoły decyzyjne: judge / voting / consensus  ← NOWE
```

## Trzy protokoły decyzyjne

Po zakończeniu debaty trzeba wybrać **ostateczną odpowiedź**. Zmień w `config.yaml`:

```yaml
decision_protocol: "judge"      # baseline
decision_protocol: "voting"     # głosowanie
decision_protocol: "consensus"  # konsensus
```

### 1. `judge` — niezależny sędzia (baseline)
Pojedynczy LLM (sędzia) czyta całą debatę i wydaje werdykt. Szybkie, ale kapryśne — sędzia może
mieć własne preferencje, halucynować lub przeskakiwać na inny język.

### 2. `voting` — głosowanie
1. Każdy agent formułuje swoją FINALNĄ odpowiedź (jedno zdanie).
2. Każdy agent głosuje na najlepszą z propozycji (jedna z N).
3. Wygrywa odpowiedź z największą liczbą głosów.

Według Kaesberg et al.: **lepsze dla zadań reasoningowych** (+13.2%).

### 3. `consensus` — konsensus
1. Pierwszy agent proponuje wspólne stanowisko.
2. Każdy agent ocenia: TAK/NIE.
3. Jeśli próg zgody (`consensus_threshold`) osiągnięty → koniec.
4. Jeśli nie → agent niezgadzający się modyfikuje stanowisko, wracamy do kroku 2.
5. Po `max_consensus_rounds` rundach kończymy z ostatnią propozycją.

Parametry:
```yaml
consensus_threshold: 0.66   # 0.5 = majority, 0.66 = supermajority, 1.0 = unanimity
max_consensus_rounds: 3
```

Według Kaesberg et al.: **lepsze dla zadań wiedzowych** (+2.8%).

## Co zmieniać w `config.yaml`

### Model
```yaml
model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# lub:
model_name: "speakleash/Bielik-1B-Instruct-v0.1"
```

### Urządzenie
```yaml
device: "cpu"     # cpu | cuda | mps
```

### Parametry generowania
```yaml
temperature: 0.7     # niższa = bardziej deterministyczny
do_sample: true      # false = greedy (zawsze ten sam wynik)
max_new_tokens: 256  # maks. długość odpowiedzi
```

### Architektura debaty
```yaml
architecture: "round_robin"   # wszyscy widzą całą historię
architecture: "relay"         # każdy widzi TYLKO poprzedniego (głuchy telefon)
architecture: "free_for_all"  # jak round_robin, ale losowa kolejność
```

### Protokół decyzyjny
```yaml
decision_protocol: "judge" | "voting" | "consensus"
```

### Agenci
Dodaj, usuń, zmień. Każdy ma `name` i `system_prompt`.

## Jak działa cały pipeline — krok po kroku

1. `main.py` ładuje config + model (z Hugging Face)
2. Tworzy agentów + sędziego (wszyscy współdzielą jeden model)
3. **Faza debaty** — `architectures.py` steruje kto widzi czyje wypowiedzi:
   - **round_robin**: pełna historia
   - **relay**: tylko ostatnia wypowiedź
   - **free_for_all**: pełna historia, losowa kolejność
4. **Faza decyzji** — `decisions.py` wybiera ostateczną odpowiedź:
   - **judge**: sędzia podsumowuje
   - **voting**: agenci proponują → głosują
   - **consensus**: agenci iterują aż do zgody

## Eksperymenty do przeprowadzenia

1. **Protokół decyzyjny**: dla tego samego tematu odpal wszystkie trzy (`judge`, `voting`, `consensus`).
   Czy odpowiedzi się różnią? Który jest najbardziej rozsądny?
2. **Próg konsensusu**: ustaw `1.0` (unanimity) vs `0.5` (majority). Czy unanimity da bardziej
   ostrożną/zachowawczą odpowiedź?
3. **Temperatura w głosowaniu**: niska temperatura = agenci podobnie myślą, brak różnorodności.
   Wysoka = chaos. Co działa lepiej dla `voting`?
4. **Architektura × protokół**: czy `relay + voting` różni się od `round_robin + voting`?
5. **Liczba agentów**: dodaj 4-tego, 5-tego agenta. Jak zmienia się dynamika głosowania?
6. **Model**: TinyLlama vs Bielik — który lepiej parsuje "TAK/NIE" i numery głosów?

## Wymagania

- Python 3.10+
- GPU z ~4 GB VRAM (lub CPU — wolniej, ale działa)

## Czym się różni od [SLM_Debate](https://github.com/InezOk/SLM_Debate)?

To repo dodaje plik `decisions.py` i parametr `decision_protocol` w configu. Cała reszta
(agents.py, architectures.py) jest taka sama. Możesz porównywać oba podejścia.
