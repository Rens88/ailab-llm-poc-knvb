from __future__ import annotations


STYLE_INSTRUCTIONS = """You are a football analytics assistant for knowledgeable data engineers.
Use only the supplied table profile and optional notes. Distinguish measured evidence from interpretation.
When the sample is a single match or otherwise narrow, state that limitation. Keep the answer concrete:
playing-style traits, evidence, likely caveats, and suggested follow-up data checks."""


def build_style_prompt(
    team_name: str,
    table_context: str,
    analyst_question: str,
    unstructured_context: str = "",
    steering: str = "",
) -> str:
    context = unstructured_context.strip()
    notes_block = context[:6000] if context else "No supplemental text was provided."
    steering_block = steering.strip()[:2000] if steering.strip() else "No steering request was provided."
    return f"""Question
{analyst_question.strip()}

Team to analyze
{team_name}

Structured table profile
{table_context}

Supplemental unstructured context
{notes_block}

Steering request
{steering_block}

Task
Characterize the playing style of the selected team from the evidence above. If a steering request is present, use it as the emphasis for this characterization. Do not invent facts that are not supported by the profile or notes."""


def generate_style_analysis(api_key: str, model: str, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=STYLE_INSTRUCTIONS,
        input=prompt,
        max_output_tokens=1400,
        store=False,
    )
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    return str(response)
