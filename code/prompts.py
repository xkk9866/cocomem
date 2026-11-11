"""Prompts of CoCoMem. Baseline prompts live next to their implementations
(and are taken from the official repositories where available)."""

EXTRACT = """You are building the long-term memory of a personal chat assistant. Read the conversation session below and extract atomic, self-contained factual claims that could matter in future conversations with this user.

Extract:
1. Facts about the user: personal details, possessions, relationships, jobs, places, events and when they happened, plans, habits, preferences, quantities, amounts, names.
2. Specific information the assistant gave that the user might ask about later (named recommendations, numbers, list items, facts).

Rules:
- One fact per line, written as a full sentence that is understandable on its own (name the subject: "The user ..." or "The assistant ...").
- Keep exact values: numbers, names, titles, prices, dates. Resolve relative times ("yesterday", "last week") against the session date when possible.
- Skip greetings, generic advice, puzzles and chit-chat. For generic Q&A sessions with no personal information, keep at most 5 lines (the topic and the most specific named items). At most 15 lines in total.
- Line format: speaker|turn|claim, where speaker is U (user) or A (assistant) and turn is the turn index, e.g.
U|0|The user adopted a beagle named Max on 2023/05/14.
A|1|The assistant recommended the book "Atomic Habits" by James Clear.
- If the user corrects or updates something said earlier, append " || REVISES: <short description of the earlier information>" to that line.
- If nothing is worth remembering, output NONE.

Session:
{session}

Claims:"""

NOTE = """Below are facts extracted from one conversation session (session date: {date}). Write a concise memory note (at most {words} words) that a personal assistant can later read to recall this session. Keep every specific value (numbers, names, dates, titles). After each sentence, cite in square brackets the ids of the facts it is based on, e.g. "The user adopted a dog named Max [c2, c5]."

Facts:
{facts}

Memory note:"""

DIGEST = """Below are facts about one topic, gathered from several past conversation sessions with the user (in chronological order). Write a consolidated memory digest (at most {words} words) that integrates them: state the current situation, compute any totals or counts the facts imply, give the timeline of events with dates, and keep every specific value. After each sentence, cite in square brackets the ids of the facts it is based on.

Facts:
{facts}

Digest:"""

GROUND = """A user has just corrected or updated something they told the assistant earlier.

New statement (from the session of {date}):
"{claim}"
{revises_hint}
What the user said in that session:
<<{context}>>

Earlier stored facts (candidates):
{cands}

Which of the earlier facts does the new statement correct, replace or make outdated? Only select facts whose content is contradicted or superseded by the new statement; do not select facts that are merely related.

Return only JSON: {{"revised": [list of fact ids], "reason": "..."}}"""

RESOLVE = """You maintain the long-term memory of {owner}. The memory stores facts from earlier conversations. Below is what was said in a new conversation, followed by stored facts that may be related to it.

New conversation ({date}):
{context}

Stored facts (id, date recorded, speaker):
{cands}

Does the new conversation correct, replace or update any stored fact? For each stored fact whose content is contradicted or made outdated by the new conversation, write the fact as it should read now: one full sentence that keeps the wording and the details of the stored fact and changes only what was corrected. Leave out stored facts that are merely related and stored facts that stay true.

Return only JSON: {{"revised": [{{"id": "<id of the stored fact>", "now": "<the fact as it reads now>"}}]}}. Return {{"revised": []}} if no stored fact is corrected."""

REPAIR = """You maintain a memory {kind} of a personal assistant. One or more facts it was derived from have been corrected by the user. Revise the {kind} so that it is consistent with the corrected facts.

Current {kind}:
\"\"\"{unit}\"\"\"

Corrections (old -> new, corrected on {date}):
{diffs}

All facts the {kind} is derived from (current versions):
{facts}

Instructions:
- Change only what the corrections affect, including anything derived from the corrected facts (totals, counts, durations, orders, dates, conclusions). Keep everything else verbatim.
- Where a value changed, write the new value and briefly note the earlier value, e.g. "3 plants (earlier recorded as 2; corrected on {date})".
- Keep the [fact id] citations, replacing ids of corrected facts with the ids of the new facts.

Output only the full revised {kind} text (no JSON, no preamble). If nothing needs to change, output exactly: UNCHANGED"""

READER = """You are a personal assistant with a long-term memory of past conversations with the user. Use the retrieved memory below to answer the user's question.

Memory entries are listed with their dates. A fact marked [corrected on ...; previously recorded ...] was corrected by the user: its first part is the current, valid information, and the "previously recorded" part is what the user had said before (use it only for questions about what was said originally). Entries marked [SUPERSEDED] are outdated. When information was corrected or updated, the most recent version is the current one.

Retrieved memory:
{memory}

Current date: {date}
Question: {question}

Think step by step about the relevant memory entries, then give a short, direct answer. Format:
Reasoning: <brief>
Answer: <answer>"""

REPAIR_SPAN = """You maintain a memory {kind} of a personal assistant. One or more facts it was derived from have been corrected by the user. The {kind} is shown sentence by sentence; the sentences marked <<k>> depend on the corrected facts, the others (k) do not and stay unchanged.

{kind}:
{unit}

Corrections (old -> new, corrected on {date}):
{diffs}

Facts the {kind} is derived from (current versions):
{facts}

Rewrite only the marked sentences ({ids}) so that they are consistent with the corrected facts, including anything derived from them (totals, counts, durations, orders, dates, conclusions). Where a value changed, write the new value and briefly note the earlier value, e.g. "3 plants (earlier recorded as 2; corrected on {date})". Keep [fact id] citations, replacing ids of corrected facts with the ids of the new facts. A marked sentence that is no longer true and has no corrected counterpart may be returned as an empty string.

Output one line per marked sentence, in the form "<k>: <rewritten sentence>", e.g.
3: The user owns 3 plants (earlier recorded as 2; corrected on {date}) [c12, c40].
Output nothing else."""

LEDGER_REPAIR = """You maintain a memory {kind} of a personal assistant. It was written from the facts below. One of those facts has been corrected by the user.

Correction: {old} => {new}

Facts the {kind} is derived from (after the correction):
{facts}

The {kind}, sentence by sentence:
{unit}

For each sentence numbered {ids}, write it as it should read after the correction: update the corrected value and anything derived from it (totals, counts, maxima, durations, orders, dates, conclusions), using the facts; keep everything else the sentence says. A sentence that needs no change is repeated unchanged.

Output one line per listed sentence, in the form "<k>: <sentence>", and nothing else."""

CF_PERTURB = """Below are facts from a personal assistant's memory. For each fact, write a counterfactual version: the same sentence with its key specific detail (a number, name, date, place, item, or yes/no status) changed to a different plausible value, as if the user had originally said something else. Keep everything else identical.

Facts:
{facts}

Output one line per fact, in the form "<k>|<counterfactual fact>", and nothing else."""

EXTRACT_DIALOG = """You are building the long-term memory of an assistant that follows the conversations between two friends, {a} and {b}. Read the conversation session below and extract atomic, self-contained factual claims about {a} and {b} that could matter later.

Extract facts about each person: personal details, family, jobs, places, possessions, pets, hobbies, events and when they happened, plans, opinions, quantities, names, and anything they share or show (including described photos).

Rules:
- One fact per line, written as a full sentence that names the person (e.g. "{a} ..." or "{b} ...").
- Keep exact values: numbers, names, titles, dates. Resolve relative times ("yesterday", "last week") against the session date when possible.
- Skip greetings and chit-chat. At most 15 lines.
- Line format: speaker|turn|claim, where speaker is A ({a}) or B ({b}), i.e. the person who said it, and turn is the turn index, e.g.
A|0|{a} adopted a beagle named Max on 2023/05/14.
- If someone corrects or updates something said earlier, append " || REVISES: <short description of the earlier information>" to that line.
- If nothing is worth remembering, output NONE.

Session:
{session}

Claims:"""

PROFILE_PERSON = """Below are facts about {person} gathered from many past conversations (chronological). Write {person}'s profile (at most {words} words): who they are, family, work, where they live, pets, possessions, hobbies, ongoing projects, important events with dates, and the current state of anything that changed over time. Keep specific values (names, numbers, dates). After each sentence, cite in square brackets the ids of the facts it is based on.

Facts:
{facts}

Profile of {person}:"""

READER_DIALOG = """You are an assistant with a long-term memory of the conversations between {a} and {b}. Use the retrieved memory below to answer the question.

Memory entries are listed with their dates. A fact marked [corrected on ...; previously recorded ...] was corrected: its first part is the current, valid information, and the "previously recorded" part is what had been said before (use it only for questions about what was said originally). Entries marked [SUPERSEDED] are outdated. When information was corrected or updated, the most recent version is the current one.

Retrieved memory:
{memory}

Question: {question}

Think step by step about the relevant memory entries, then give a short, direct answer. Format:
Reasoning: <brief>
Answer: <answer>"""


# ---------------------------------------------------------------------------------------------------------------
# Record-first (self-maintainable) compression, v22.  The compressor writes every sentence as a triple: what the
# sentence will state (T), the derivation record it is stated from (D: operands as "<fact id> = <value taken>", every
# fact that is added or compared, and the computation), and the sentence (S).  The record is the materialized
# auxiliary view of the sentence (stored as metadata); the sentences form the unit text; the T line is not stored.
# Nothing is attributed after the fact.
#
# Why the T line: a record written before anything says what its sentence is about is conditioned on the fact list
# alone, and greedy decoding then restates the list -- in the first build (D/S pairs, 11-10) 41% of the note sentences
# in units with six or more inputs carried a record naming at least 80% of the unit's inputs, and 119 of 274 profiles
# ran into the output window inside their first record, before any sentence was written.  Naming the statement first
# makes the record a function of that statement.  rebuild_units.py stats reports the share of such records
# ("listing") for the memories in use.
PLAN_FORMAT = """Write the {kind} as a sequence of short sentences, one statement per sentence. For EVERY sentence write three lines: what the sentence will state, the record it is stated from, and the sentence itself. A record holds only the facts that its own sentence states or computes with, never the other facts of the {kind}. A {kind} of n sentences therefore has n T/D/S triples; never put two sentences on one S line.
T: <what the sentence will state, in a few words>
D: <the facts it is stated from, each as one entry "<fact id> = <the exact value or detail the sentence takes from that fact>", entries separated by semicolons, one fact id per entry. For a total, a count, a maximum or minimum, a second-highest, a difference, a first or last event, an order, a duration or a conclusion, list EVERY fact that is added or compared, including the facts that do not win the comparison, and end with the computation and its result. Write "none" if the sentence uses no fact.>
S: <exactly one sentence, stating what its record holds, keeping every specific value, with the ids of its facts cited in square brackets>
Example of the form:
T: number of plants
D: c3 = 3 plants; c7 = 2 plants; total 3 + 2 = 5
S: The user owns five plants in total [c3, c7].
T: longest ride
D: c12 = Monday 52 km; c13 = Tuesday 30 km; c15 = Friday 41 km; max = 52 km (Monday)
S: The longest ride was 52 km, on Monday [c12, c13, c15].
T: age group
D: c4 = age 17; 17 < 18, so a minor
S: The user is a minor [c4].
Output only T:, D: and S: lines, nothing else."""

PLAN_NOTE = """Below are facts extracted from one conversation session (session date: {date}). Write a concise memory note (at most {sents} sentences, together at most {words} words) that a personal assistant can later read to recall this session. Keep every specific value (numbers, names, dates, titles).

Facts:
{facts}

""" + PLAN_FORMAT.replace("{kind}", "note") + """

Memory note:"""

PLAN_DIGEST = """Below are facts about one topic, gathered from several past conversation sessions with the user (in chronological order). Write a consolidated memory digest (at most {sents} sentences, together at most {words} words) that integrates them: state the current situation, compute any totals or counts the facts imply, give the timeline of events with dates, and keep every specific value.

Facts:
{facts}

""" + PLAN_FORMAT.replace("{kind}", "digest") + """

Digest:"""

PLAN_PROFILE = """Below are facts the user shared across many past conversations (chronological). Write the user's profile (at most {sents} sentences, together at most {words} words): who they are, where they live and work, family, pets, possessions, ongoing projects, routines, preferences and the current state of anything that changed over time. Keep specific values (names, numbers, dates).

Facts:
{facts}

""" + PLAN_FORMAT.replace("{kind}", "profile") + """

User profile:"""

PLAN_PROFILE_PERSON = """Below are facts about {person} gathered from many past conversations (chronological). Write {person}'s profile (at most {sents} sentences, together at most {words} words): who they are, family, work, where they live, pets, possessions, hobbies, ongoing projects, important events with dates, and the current state of anything that changed over time. Keep specific values (names, numbers, dates).

Facts:
{facts}

""" + PLAN_FORMAT.replace("{kind}", "profile") + """

Profile of {person}:"""


SAME = """Do the two sentences below state the same information? Ignore differences in wording, order and citation markers; answer "no" only if a fact, value, name, number, date or conclusion differs.

A: {a}
B: {b}

Answer yes or no only."""
