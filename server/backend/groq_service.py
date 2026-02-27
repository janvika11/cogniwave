"""
Groq API integration for report generation.
Supports both legacy medical diagnostics and the new
multimodal emotion intelligence reports.
"""

from groq import Groq
import os
import logging
from dotenv import load_dotenv
import re

logger = logging.getLogger(__name__)

# Ensure .env is loaded from this backend directory, regardless of cwd
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Initialize Groq client (robust to config/network issues)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Set via environment variable
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set. Report generation will fail.")
    groq_client = None
else:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        groq_client = None


def _require_client():
    if not groq_client:
        raise Exception("Groq API not configured. Set GROQ_API_KEY environment variable.")


def generate_medical_report(analysis_data: dict) -> str:
    """
    Legacy medical report generator (still available for compatibility).
    """
    _require_client()

    patient_info = analysis_data.get("patient_metadata", {})
    physiological = analysis_data.get("physiological_data", {})
    ecg_results = analysis_data.get("ecg_analysis", {})
    emotion_results = analysis_data.get("emotion_analysis", {})
    summary = analysis_data.get("summary", {})

    prompt = f"""You are an expert medical AI assistant. Generate a comprehensive, professional medical diagnostic report based on the following patient analysis data:

PATIENT INFORMATION:
- Patient ID: {patient_info.get('patient_id', 'N/A')}
- Age: {patient_info.get('age', 'N/A')}
- Gender: {patient_info.get('gender', 'N/A')}

VITAL SIGNS:
- Heart Rate: {physiological.get('heart_rate', 'N/A')} BPM
- Blood Pressure: {physiological.get('blood_pressure_systolic', 'N/A')}/{physiological.get('blood_pressure_diastolic', 'N/A')} mmHg
- Temperature: {physiological.get('temperature', 'N/A')}°C
- Oxygen Saturation: {physiological.get('oxygen_saturation', 'N/A')}%

ECG ANALYSIS FINDINGS:
{format_findings(emc_results=ecg_results)}

EMOTIONAL/BEHAVIORAL ANALYSIS:
{format_findings(emc_results=emotion_results)}

OVERALL ASSESSMENT:
- Priority Level: {summary.get('priority_level', 'N/A').upper()}
- Overall Status: {summary.get('overall_status', 'N/A')}
- Requires Immediate Attention: {'Yes' if summary.get('requires_immediate_attention') else 'No'}

Please generate a structured medical report with the following sections:
1. CLINICAL SUMMARY - Brief overview of patient condition
2. VITAL SIGNS ASSESSMENT - Analysis of physiological parameters
3. ECG FINDINGS - Detailed cardiac assessment
4. BEHAVIORAL ASSESSMENT - Emotional and stress indicators
5. RISK ANALYSIS - Identified risk factors and severity
6. RECOMMENDATIONS - Medical recommendations and follow-up actions
7. CONCLUSION - Final assessment and urgency level

Format the report professionally as it would be used by healthcare professionals. Be specific, concise, and clinically accurate."""

    try:
        logger.info("Generating medical report with Groq API...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert medical AI that generates professional diagnostic reports for healthcare providers.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000,
        )

        report = chat_completion.choices[0].message.content
        logger.info("Medical report generated successfully")
        return report

    except Exception as e:
        logger.error(f"Failed to generate report with Groq: {str(e)}")
        raise Exception(f"Report generation failed: {str(e)}")


def generate_emotion_intelligence_report(analysis_data: dict) -> str:
    """
    Generate a multimodal emotion intelligence report using Groq API.
    Focuses on voice + facial cues and fused emotional state.
    """
    _require_client()

    patient_info = analysis_data.get("patient_metadata", {})
    emotion_results = analysis_data.get("emotion_analysis", {})
    summary = analysis_data.get("summary", {})

    findings = emotion_results.get("findings", {})
    voice = findings.get("voice_emotion", {})
    face = findings.get("face_emotion", {})
    fused = findings.get("fused_emotion", {})
    context = findings.get("session_context")
    text_input = findings.get("text_input")

    prompt = f"""You are an expert in affective computing and multimodal emotion intelligence.
Generate a clear, structured EMOTION REPORT based on the following data.

PARTICIPANT INFORMATION:
- ID: {patient_info.get('patient_id', 'N/A')}
- Age: {patient_info.get('age', 'N/A')}
- Gender: {patient_info.get('gender', 'N/A')}

SESSION CONTEXT:
- Context: {context or 'N/A'}
- User text (if any): {text_input or 'N/A'}

VOICE PARALINGUISTIC EMOTION:
- Dominant: {voice.get('dominant', 'N/A')}
- Scores: {voice.get('scores', {})}

FACIAL EXPRESSION / ACTION-UNIT EMOTION:
- Dominant: {face.get('dominant', 'N/A')}
- Scores: {face.get('scores', {})}

FUSED MULTIMODAL EMOTION:
- Dominant fused emotion: {fused.get('dominant', 'N/A')}
- Scores: {fused.get('scores', {})}
- Confidence: {fused.get('confidence', 'N/A')}

OVERALL ASSESSMENT:
- Overall Status: {summary.get('overall_status', 'N/A')}
- Priority Level: {summary.get('priority_level', 'N/A').upper()}
- Requires Immediate Attention: {'Yes' if summary.get('requires_immediate_attention') else 'No'}

Please generate a structured report with the following sections:
1. EMOTIONAL SUMMARY – brief overview of the participant's emotional state.
2. VOICE-BASED ANALYSIS – interpretation of paralinguistic cues (tone, intensity, speaking style).
3. FACIAL-EXPRESSION ANALYSIS – interpretation of facial cues and likely emotions.
4. MULTIMODAL FUSION INSIGHTS – how voice and face signals combine; where they agree/disagree.
5. RISK & DISTRESS ASSESSMENT – level of stress, frustration, or distress.
6. RECOMMENDATIONS – how an AI/coach/therapist should respond (tone, strategies, next steps).
7. PRIVACY & LIMITATIONS – short note that this is probabilistic, not a clinical diagnosis.

Use clear headings and bullet points. Keep the language empathetic but analytical."""

    try:
        logger.info("Generating multimodal emotion intelligence report with Groq API...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in multimodal emotion intelligence and affective computing.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=1800,
        )

        report = chat_completion.choices[0].message.content or ""

        # Basic formatting cleanup so PDF/UI look neat:
        # - convert markdown headings (## ...) into blank-line-separated sections
        # - convert markdown bullets (* ...) into bullet lines
        cleaned = report
        cleaned = re.sub(r"\s*#+\s*", "\n\n", cleaned)          # drop '#' and start new section
        cleaned = re.sub(r"\s*\*\s+", "\n• ", cleaned)          # markdown '*' -> bullet
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)            # collapse too many blank lines

        logger.info("Emotion intelligence report generated successfully")
        return cleaned

    except Exception as e:
        logger.error(f"Failed to generate emotion intelligence report with Groq: {str(e)}")
        raise Exception(f"Emotion intelligence report generation failed: {str(e)}")


def format_findings(emc_results: dict) -> str:
    """Format analysis findings for prompt"""
    if not emc_results:
        return "No data available"

    findings = emc_results.get("findings", {})
    if not findings:
        return "No specific findings"

    output = []
    for key, value in findings.items():
        if isinstance(value, dict):
            output.append(f"- {key.replace('_', ' ').title()}: {value}")
        elif isinstance(value, list):
            output.append(f"- {key.replace('_', ' ').title()}: {', '.join(str(v) for v in value)}")
        else:
            output.append(f"- {key.replace('_', ' ').title()}: {value}")

    return "\n".join(output) if output else "No specific findings"
