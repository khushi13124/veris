# visits/templatetags/ai_result_tags.py
# Place at: visits/templatetags/ai_result_tags.py
# Ensure visits/templatetags/__init__.py also exists (empty file)

from django import template

register = template.Library()


@register.simple_tag
def parse_ai_result(ai_text):
    """
    Parses the structured AI result text stored in Visit.ai_differential.

    Format stored by symptom_disease_page.html buildResultText():
      SYMPTOMS_ENTERED:symptom1|symptom2|...
      CONDITION_1:disease name|CONF:xx.x|MATCHED:s1|s2|MISSING:s3|s4|COMPLETENESS:yy
      CONDITION_2:...

    Returns a dict:
      symptoms   — list[str]
      conditions — list[{name, confidence, completeness, matched, missing}]

    Returns None for legacy plain-text format (no SYMPTOMS_ENTERED: marker).
    """
    if not ai_text or 'SYMPTOMS_ENTERED:' not in ai_text:
        return None

    result = {'symptoms': [], 'conditions': []}

    for line in ai_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # ── Symptoms line ──────────────────────────────────────────────────
        if line.startswith('SYMPTOMS_ENTERED:'):
            raw = line[len('SYMPTOMS_ENTERED:'):]
            result['symptoms'] = [s.strip() for s in raw.split('|') if s.strip()]
            continue

        # ── Condition lines ───────────────────────────────────────────────
        if not line.startswith('CONDITION_'):
            continue

        try:
            # Strip the "CONDITION_N:" prefix
            colon_idx = line.index(':')
            rest = line[colon_idx + 1:]   # everything after "CONDITION_N:"

            # Locate keyword boundaries (all are prefixed with "|KEY:")
            # so we can split safely even if disease names contain pipes.
            CONF_KEY        = '|CONF:'
            MATCHED_KEY     = '|MATCHED:'
            MISSING_KEY     = '|MISSING:'
            COMPLETENESS_KEY= '|COMPLETENESS:'

            conf_pos        = rest.find(CONF_KEY)
            matched_pos     = rest.find(MATCHED_KEY)
            missing_pos     = rest.find(MISSING_KEY)
            completeness_pos= rest.find(COMPLETENESS_KEY)

            # Disease name is everything before |CONF:
            name = rest[:conf_pos].strip() if conf_pos != -1 else rest.strip()

            def _between(s, start_key, end_pos):
                """Extract substring between start_key and end_pos (or end of string)."""
                start = s.find(start_key)
                if start == -1:
                    return ''
                start += len(start_key)
                if end_pos != -1 and end_pos > start:
                    return s[start:end_pos]
                return s[start:]

            # CONF value: between |CONF: and |MATCHED: (or |MISSING: or |COMPLETENESS:)
            conf_end = matched_pos if matched_pos != -1 else (
                       missing_pos if missing_pos != -1 else completeness_pos)
            conf_raw = _between(rest, CONF_KEY, conf_end).strip()

            # MATCHED value: between |MATCHED: and |MISSING: (or |COMPLETENESS:)
            matched_end = missing_pos if missing_pos != -1 else completeness_pos
            matched_raw = _between(rest, MATCHED_KEY, matched_end).strip()

            # MISSING value: between |MISSING: and |COMPLETENESS:
            missing_raw = _between(rest, MISSING_KEY, completeness_pos).strip()

            # COMPLETENESS value: everything after |COMPLETENESS:
            completeness_raw = _between(rest, COMPLETENESS_KEY, -1).strip()

            matched_list = [s.strip() for s in matched_raw.split('|') if s.strip()] if matched_raw else []
            missing_list = [s.strip() for s in missing_raw.split('|') if s.strip()] if missing_raw else []

            result['conditions'].append({
                'name':         name,
                'confidence':   conf_raw,
                'completeness': completeness_raw,
                'matched':      matched_list,
                'missing':      missing_list,
            })

        except Exception as e:
            # Skip malformed lines silently
            continue

    return result if (result['symptoms'] or result['conditions']) else None