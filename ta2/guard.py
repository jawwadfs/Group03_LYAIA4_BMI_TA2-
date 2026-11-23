"""guard.py - rule layer of the Safety-guard agent (deterministic, testable).

It refuses: (1) diagnosis requests/claims, (2) extreme or crash-diet advice,
(3) medication advice, (4) prompt-injection attempts.
The LLM judge in agents.py is a second layer applied to the Advisor's draft.
"""
import re

I = re.I
INJECTION = [r"\b(ignore|disregard|forget)\b.{0,30}\b(rules|instructions|prompt|guidelines)\b",
             r"\b(system prompt|developer mode|jailbreak|you are now|override)\b"]
DIAGNOSIS_IN = [
    r"\bdiagnos(e|is)\b",
    r"\bwhat (illness|disease|condition|disorder)\b",
    # Only block "do I have" when followed by a medical condition term (prevents false-positives
    # on harmless questions like "do I have to exercise every day?")
    r"\bdo i have\b.{0,50}\b(diabetes|hypertension|disease|disorder|cancer|syndrome|obesity|condition|illness)\b",
    r"\bam i (sick|ill|diabetic)\b",
]
DIAGNOSIS_OUT = [r"\byou (have|suffer from|are suffering from|are diagnosed with)\b.{0,25}"
                 r"\b(diabetes|hypertension|heart disease|disease|disorder)\b",
                 r"\byou are (diabetic|sick|ill)\b"]
MEDICATION = [r"\b(medicine|medication|prescription|tablets?|pills?|drugs?|dosage|dose)\b"]
EXTREME = [r"\b(lose|drop|shed)\b.{0,20}\d+\s*(kg|kgs|kilos?|pounds?|lbs?)\b.{0,15}"
           r"\b(in|within)\b.{0,6}\b(a|an|one|1|2|two|3|three|\d+)\s*(day|days|week|weeks)\b",
           r"\b(starv\w*|crash diet|water fast|stop eating|skip (all )?meals|laxatives?|purg\w*|"
           r"how little (can|should) i eat|eat (nothing|almost nothing))\b",
           r"\bfast(ing)? for \d+ days?\b"]

MESSAGES = {
    "injection": "I can't ignore my safety rules, but I'm happy to help with your BMI.",
    "diagnosis": "I can't diagnose health conditions. BMI is only a screening measure - "
                 "please talk to a doctor about your health.",
    "medication": "I can't recommend medicines or treatments. Please consult a doctor.",
    "extreme": "I can't give extreme or crash-diet advice. Please speak to a doctor or "
               "registered dietitian for a safe plan.",
}


def _hit(patterns, text):
    return any(re.search(p, text, I) for p in patterns)


def _low_calories(text) -> bool:
    return any(int(n) < 1200 for n in re.findall(r"\b(\d{3,4})\s*(?:kcal|calories|cal)\b", text, I))


def check(text: str, where: str = "input") -> str:
    """Return a reason code ('' if safe). where is 'input' or 'output'."""
    if where == "input" and _hit(INJECTION, text):
        return "injection"
    if _hit(DIAGNOSIS_IN if where == "input" else DIAGNOSIS_OUT, text):
        return "diagnosis"
    if _hit(EXTREME, text) or _low_calories(text):
        return "extreme"
    if _hit(MEDICATION, text):
        return "medication"
    return ""
