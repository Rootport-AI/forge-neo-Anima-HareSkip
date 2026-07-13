"""Manual Skip mode: parse and validate an explicit step-skip list.

Pure stdlib module — no Forge / gradio / torch imports, so it can be unit
tested and imported by tooling in isolation (mirrors ``constants.py`` and the
Forge-independent parts of ``state.py``).

Manual Skip mode is the third exclusive skip-strategy mode (alongside HareSkip
and TeaCache). Instead of a stochastic pattern or an accumulator decision, the
experimenter types a comma-separated list of 1-based step numbers to skip. The
list is parsed here and validated against the concrete ``p.steps`` of the run.

Design intent (see ``docs/ManualSkip-spec.md`` §2): this is experiment
apparatus, not a creative convenience. Ambiguous input must raise an error and
abort generation rather than silently falling back to full compute — the
opposite of the "degrade safely to full compute" rule that governs the other
modes. All error messages are human-readable because they surface directly in
the Forge UI (raised as ``RuntimeError`` from ``Script.before_process``).

Division of responsibility between the two functions:
  * ``parse_manual_steps`` — syntax only. Splits on commas, tolerates
    whitespace, empty tokens and trailing commas, deduplicates, and returns a
    sorted list. Raises ``ManualSkipError`` for any non-numeric token. It does
    NOT know the step count, so it cannot range-check; it also does not reject
    step < 1 here — that is left to ``validate_manual_steps`` so that ALL
    numeric-range rejections (step < 1, step == 1, step > num_steps) live in
    one place with consistent messages.
  * ``validate_manual_steps`` — semantics against a concrete ``num_steps``:
    rejects step < 1, step == 1 (physically unskippable — step 1 has no prior
    residual, the same technical constraint as patcher's ``first_call``
    force-full), and step > num_steps.
"""

from __future__ import annotations


class ManualSkipError(ValueError):
    """Raised for invalid Manual Skip input.

    Carries a human-readable message (via the standard ``ValueError`` args)
    because it is surfaced verbatim in the Forge UI. ``script.before_process``
    catches it and re-raises as ``RuntimeError`` to abort generation, mirroring
    the Auto Tea CSV error path.
    """


def parse_manual_steps(text: str) -> list[int]:
    """Parse a comma-separated list of 1-based step numbers.

    Tolerant of surrounding whitespace, empty tokens and trailing commas, so
    ``"10, , 12,"`` parses the same as ``"10, 12"``. Duplicate step numbers are
    silently removed. The result is sorted ascending so it is deterministic and
    unit-testable. An empty or whitespace-only ``text`` yields ``[]`` (a valid
    "skip nothing" / baseline request).

    Raises ``ManualSkipError`` for any token that is not an integer. Range
    checks (step < 1, step == 1, step > num_steps) are deliberately NOT done
    here — see ``validate_manual_steps``.
    """
    if text is None:
        return []
    steps: set[int] = set()
    for raw_token in str(text).split(","):
        token = raw_token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except (ValueError, TypeError):
            raise ManualSkipError(
                f"'{token}' is not a valid step number (expected a "
                "comma-separated list of whole numbers, e.g. 10, 12)."
            )
        steps.add(value)
    return sorted(steps)


def validate_manual_steps(steps: list[int], num_steps: int) -> None:
    """Validate parsed step numbers against the run's concrete step count.

    ``steps`` must be the output of ``parse_manual_steps`` (deduplicated
    integers). ``num_steps`` is the live ``p.steps`` of the generation, so the
    upper bound tracks the actual step count of each run (no hard-coded limit).

    Rejects, with a human-readable ``ManualSkipError``:
      * step < 1               — steps are 1-based; 0 and negatives are invalid.
      * step == 1              — step 1 is physically unskippable (no prior
                                 residual exists; same constraint as patcher's
                                 ``first_call`` force-full). Rejected at input
                                 rather than silently forced to full compute.
      * step > num_steps       — out of range; also a fool-proof against
                                 accidentally lowering the step count before
                                 Generate.

    Returns ``None`` on success. Does not mutate ``steps``.
    """
    try:
        max_step = int(num_steps)
    except (ValueError, TypeError):
        max_step = 0
    for step in steps:
        if step < 1:
            raise ManualSkipError(
                f"Step {step} is invalid: step numbers are 1-based and must "
                "be 1 or greater."
            )
        if step == 1:
            raise ManualSkipError(
                "Step 1 cannot be skipped: the first step has no prior "
                "residual to reuse. Remove 1 from the skip list."
            )
        if step > max_step:
            raise ManualSkipError(
                f"Step {step} is out of range: this generation has "
                f"{max_step} step(s). Specify steps between 2 and {max_step}."
            )


def parse_manual_lines(text: str) -> list[tuple[int, list[int]]]:
    """Parse a multiline Manual Skip input: one non-blank line = one job.

    Each non-blank line is a comma-separated skip list with the SAME grammar as
    a single ``parse_manual_steps`` input (§10.2). Line splitting happens FIRST,
    before per-line parsing, so a multiline paste never collapses into one
    comma-separated list (the trailing-comma silent-merge hazard, §10.2). Blank
    and whitespace-only lines are ignored (they do not produce a job).

    Returns a list of ``(physical_line_no, steps)`` tuples where
    ``physical_line_no`` is the 1-based line number as it appears in the
    textbox (blank lines are counted, so the number matches what the user
    sees). ``text is None`` and empty / all-blank input yield ``[]``.

    Raises ``ManualSkipError`` for any line containing a non-numeric token; the
    message is prefixed with ``Line {physical_line_no}:`` so the offending line
    is named. Range checks (step < 1, step == 1, step > num_steps) are NOT done
    here — see ``validate_manual_lines``.
    """
    if text is None:
        return []
    parsed: list[tuple[int, list[int]]] = []
    for index, raw_line in enumerate(str(text).split("\n")):
        physical_line_no = index + 1
        if not raw_line.strip():
            continue
        try:
            steps = parse_manual_steps(raw_line)
        except ManualSkipError as exc:
            raise ManualSkipError(f"Line {physical_line_no}: {exc}")
        parsed.append((physical_line_no, steps))
    return parsed


def validate_manual_lines(
    parsed: list[tuple[int, list[int]]], num_steps: int
) -> None:
    """Validate every parsed Manual Skip line against the run's step count.

    ``parsed`` is the output of ``parse_manual_lines`` — a list of
    ``(physical_line_no, steps)`` tuples. Each line's ``steps`` is validated
    with the existing ``validate_manual_steps``; any failure is re-raised with a
    ``Line {physical_line_no}:`` prefix so the offending line is named.

    Raises on the FIRST invalid line (mirrors the Auto Tea CSV parser, which
    raises on the first bad row). Because validation runs in ``before_process``
    before generation starts, a single bad line aborts the whole job — no
    partial execution (§10.2). Returns ``None`` on success.
    """
    for physical_line_no, steps in parsed:
        try:
            validate_manual_steps(steps, num_steps)
        except ManualSkipError as exc:
            raise ManualSkipError(f"Line {physical_line_no}: {exc}")
