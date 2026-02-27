"""
CogniWave Complete Backend - Database + Auth + Groq + PDF
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import traceback
import json
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import numpy as np

try:
    from fer import FER
    FER_AVAILABLE = True
except Exception as e:
    FER_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Import modules
from simple_agents import ECGAgent, EmotionAgent
from database import init_db, get_db, Doctor, Patient, AnalysisReport, REPORTS_DIR
from auth import get_password_hash, authenticate_doctor, create_access_token, get_current_doctor
from groq_service import generate_medical_report, generate_emotion_intelligence_report
from pdf_generator import generate_pdf_report
# Chatbot will use Gemini API directly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CogniWave API v2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    init_db()
    logger.info("✓ Database initialized")

ecg_agent = ECGAgent()
emotion_agent = EmotionAgent()

fer_detector = None
if FER_AVAILABLE:
    try:
        fer_detector = FER(mtcnn=True)
        logger.info("FER facial emotion detector initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize FER detector: {e}")
        fer_detector = None

# ===== Models =====
class DoctorRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    specialization: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    doctor: dict

class PatientMetadata(BaseModel):
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    full_name: Optional[str] = None

class PhysiologicalData(BaseModel):
    ecg_data: Optional[List[float]] = None
    heart_rate: Optional[float] = None
    blood_pressure_systolic: Optional[float] = None
    blood_pressure_diastolic: Optional[float] = None
    temperature: Optional[float] = None
    oxygen_saturation: Optional[float] = None

class BehavioralData(BaseModel):
    image_data: Optional[str] = None
    audio_data: Optional[str] = None
    text_data: Optional[str] = None

class AnalysisRequest(BaseModel):
    patient_metadata: PatientMetadata
    physiological_data: Optional[PhysiologicalData] = None
    behavioral_data: Optional[BehavioralData] = None
    symptoms: Optional[List[str]] = []
    notes: Optional[str] = None


# ===== Multimodal Emotion Session Models =====

class VoiceFeatures(BaseModel):
    """Paralinguistic voice features used for emotion estimation"""
    avg_pitch: Optional[float] = None
    avg_intensity: Optional[float] = None
    speaking_rate: Optional[float] = None
    pause_ratio: Optional[float] = None


class FaceFeatures(BaseModel):
    """Facial expression / action-unit style features"""
    smile_score: Optional[float] = None
    frown_score: Optional[float] = None
    brow_tension: Optional[float] = None
    eye_openness: Optional[float] = None


class EmotionSessionRequest(BaseModel):
    """Request body for multimodal emotion intelligence session"""
    patient_metadata: PatientMetadata
    voice_features: Optional[VoiceFeatures] = None
    face_features: Optional[FaceFeatures] = None
    image_data: Optional[str] = None  # base64 frame from camera
    audio_data: Optional[str] = None  # base64 audio from microphone
    text_input: Optional[str] = None
    session_context: Optional[str] = None


# ===== Multimodal Emotion Session Endpoint =====

@app.post("/api/v1/analysis/emotion-session", tags=["Analysis"])
async def analyze_emotion_session(req: EmotionSessionRequest, db: Session = Depends(get_db)):
    """
    Multimodal emotion intelligence endpoint.
    Uses heuristic voice + face models, fuses them, and generates an
    emotion-centric report via Groq and PDF.
    """
    try:
        start = datetime.utcnow()
        analysis_id = f"EMOTION-{int(start.timestamp() * 1000)}"
        logger.info(f"→ Emotion session {analysis_id}")

        # Get/Create Patient / Participant
        patient = db.query(Patient).filter(Patient.patient_id == req.patient_metadata.patient_id).first()
        if not patient:
            # New participant
            patient = Patient(
                patient_id=req.patient_metadata.patient_id,
                full_name=req.patient_metadata.full_name,
                age=req.patient_metadata.age,
                gender=req.patient_metadata.gender,
            )
            db.add(patient)
            db.commit()
            db.refresh(patient)
        else:
            # Update basic demographics if user provided new info
            updated = False
            if req.patient_metadata.full_name and req.patient_metadata.full_name != patient.full_name:
                patient.full_name = req.patient_metadata.full_name
                updated = True
            if req.patient_metadata.age is not None and req.patient_metadata.age != patient.age:
                patient.age = req.patient_metadata.age
                updated = True
            if req.patient_metadata.gender and req.patient_metadata.gender != patient.gender:
                patient.gender = req.patient_metadata.gender
                updated = True
            if updated:
                db.add(patient)
                db.commit()
                db.refresh(patient)

        # Run voice analyzer
        # If no microphone/audio was provided, assume a mostly calm baseline
        # so that facial expressions drive the result.
        if not req.audio_data:
            voice_probs: Dict[str, float] = {
                "calm": 0.8,
                "stressed": 0.1,
                "frustrated": 0.05,
                "engaged": 0.05,
            }
        else:
            voice_probs = analyze_audio_with_librosa(req.audio_data) or {}
            if not voice_probs:
                voice_probs = analyze_voice_features(req.voice_features)

        # Face analysis: prefer FER on real camera frame when available
        face_probs = {}
        if req.image_data:
            face_probs = analyze_face_image_with_fer(req.image_data)
        if not face_probs:
            face_probs = analyze_face_features(req.face_features)
        fused_probs = fuse_emotions(voice_probs, face_probs)

        dominant_emotion = max(fused_probs, key=fused_probs.get)
        summary = build_emotion_summary(fused_probs)

        emotion_analysis = {
            "agent_id": "multimodal_emotion_agent",
            "agent_name": "Multimodal Emotion Intelligence Agent",
            "status": "success",
            "findings": {
                "voice_emotion": {
                    "scores": voice_probs,
                    "dominant": max(voice_probs, key=voice_probs.get),
                    "features": req.voice_features.dict() if req.voice_features else {},
                },
                "face_emotion": {
                    "scores": face_probs,
                    "dominant": max(face_probs, key=face_probs.get),
                    "features": req.face_features.dict() if req.face_features else {},
                },
                "fused_emotion": {
                    "scores": fused_probs,
                    "dominant": dominant_emotion,
                    "confidence": fused_probs.get(dominant_emotion, 0.0),
                },
                "session_context": req.session_context,
                "text_input": req.text_input,
            },
        }

        # Build data for Groq report (reusing existing structure)
        report_data = {
            "patient_metadata": req.patient_metadata.dict(),
            "physiological_data": {},
            "ecg_analysis": {},
            "emotion_analysis": emotion_analysis,
            "summary": summary,
        }

        try:
            logger.info("  Generating Groq emotion report...")
            groq_report = generate_emotion_intelligence_report(report_data)
            logger.info("  ✓ Groq emotion report done")
        except Exception as e:
            logger.error(f"  ✗ Groq emotion report failed: {e}")
            groq_report = f"Emotion report generation failed: {e}"

        # PDF generation
        pdf_filename = f"{analysis_id}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        report_data["groq_report"] = groq_report
        report_data["analysis_id"] = analysis_id

        try:
            logger.info("  Generating emotion PDF...")
            generate_pdf_report(report_data, pdf_path)
            pdf_url = f"/api/v1/reports/download/{pdf_filename}"
            logger.info("  ✓ Emotion PDF done")
        except Exception as e:
            logger.error(f"  ✗ Emotion PDF failed: {e}")
            pdf_url = None
            pdf_filename = None

        # Persist in AnalysisReport table (reusing schema)
        report = AnalysisReport(
            analysis_id=analysis_id,
            patient_id=patient.id,
            doctor_id=None,
            heart_rate=None,
            bp_systolic=None,
            bp_diastolic=None,
            temperature=None,
            oxygen_saturation=None,
            ecg_analysis="{}",
            emotion_analysis=json.dumps(emotion_analysis),
            overall_summary=json.dumps(summary),
            priority_level=summary.get("priority_level", "low"),
            risk_score=None,
            requires_attention=summary.get("requires_immediate_attention", False),
            groq_report=groq_report,
            pdf_filename=pdf_filename,
            symptoms=None,
            notes=req.session_context,
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        total_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(f"✓ Emotion session complete: {analysis_id} (DB ID: {report.id}) in {total_ms:.1f} ms")

        return {
            "analysis_id": analysis_id,
            "status": "success",
            "dominant_emotion": dominant_emotion,
            "fused_scores": fused_probs,
            "voice_scores": voice_probs,
            "face_scores": face_probs,
            "summary": summary,
            "groq_report": groq_report,
            "pdf_url": pdf_url,
            "database_id": report.id,
            "total_processing_time_ms": total_ms,
        }

    except Exception as e:
        logger.error(f"✗ Emotion session failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Emotion session failed: {e}")

# ===== Auth Endpoints =====
@app.post("/api/v1/auth/register", response_model=Token, tags=["Auth"])
async def register(data: DoctorRegister, db: Session = Depends(get_db)):
    if db.query(Doctor).filter(Doctor.username == data.username).first():
        raise HTTPException(400, "Username exists")
    doctor = Doctor(username=data.username, email=data.email, hashed_password=get_password_hash(data.password),
                   full_name=data.full_name, specialization=data.specialization)
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    token = create_access_token({"sub": doctor.username})
    logger.info(f"✓ Registered: {doctor.username}")
    return {"access_token": token, "token_type": "bearer", 
            "doctor": {"id": doctor.id, "username": doctor.username, "email": doctor.email, "full_name": doctor.full_name}}

@app.post("/api/v1/auth/login", response_model=Token, tags=["Auth"])
async def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    doctor = authenticate_doctor(db, form.username, form.password)
    if not doctor:
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": doctor.username})
    logger.info(f"✓ Login: {doctor.username}")
    return {"access_token": token, "token_type": "bearer",
            "doctor": {"id": doctor.id, "username": doctor.username, "email": doctor.email, "full_name": doctor.full_name}}

@app.get("/api/v1/auth/me", tags=["Auth"])
async def me(doctor: Doctor = Depends(get_current_doctor)):
    return {"id": doctor.id, "username": doctor.username, "email": doctor.email, "full_name": doctor.full_name}

# ===== Analysis Endpoint =====
@app.post("/api/v1/analysis/patient", tags=["Analysis"])
async def analyze(req: AnalysisRequest, db: Session = Depends(get_db)):
    try:
        start = datetime.utcnow()
        analysis_id = f"ANALYSIS-{int(start.timestamp() * 1000)}"
        logger.info(f"→ Analysis {analysis_id}")
        
        # Get/Create Patient
        patient = db.query(Patient).filter(Patient.patient_id == req.patient_metadata.patient_id).first()
        if not patient:
            patient = Patient(patient_id=req.patient_metadata.patient_id, full_name=req.patient_metadata.full_name,
                            age=req.patient_metadata.age, gender=req.patient_metadata.gender)
            db.add(patient)
            db.commit()
            db.refresh(patient)
        
        results = {"analysis_id": analysis_id, "status": "success", "agent_results": {}}
        
        # Run Agents
        if req.physiological_data:
            logger.info("  ECG Agent analyzing...")
            results["agent_results"]["ecg_agent"] = await ecg_agent.analyze(req.physiological_data)
        
        # Debug: Log behavioral data status
        if req.behavioral_data:
            logger.info(f"  Behavioral data received:")
            logger.info(f"    - image_data: {bool(req.behavioral_data.image_data)} (length: {len(req.behavioral_data.image_data) if req.behavioral_data.image_data else 0})")
            logger.info(f"    - audio_data: {bool(req.behavioral_data.audio_data)} (length: {len(req.behavioral_data.audio_data) if req.behavioral_data.audio_data else 0})")
            logger.info(f"    - text_data: {bool(req.behavioral_data.text_data)}")
        else:
            logger.info("  No behavioral_data in request")
        
        # Only run emotion agent if there's actual image or audio data
        if req.behavioral_data and (req.behavioral_data.image_data or req.behavioral_data.audio_data or req.behavioral_data.text_data):
            logger.info("  Emotion Agent analyzing...")
            logger.info(f"    - Image data: {bool(req.behavioral_data.image_data)}")
            logger.info(f"    - Audio data: {bool(req.behavioral_data.audio_data)}")
            logger.info(f"    - Text data: {bool(req.behavioral_data.text_data)}")
            results["agent_results"]["emotion_agent"] = await emotion_agent.analyze(req.behavioral_data)
        else:
            logger.info("  Emotion Agent SKIPPED - no image/audio/text data provided")
        
        # Summary
        results["summary"] = generate_summary(results["agent_results"])
        
        # Groq Report
        report_data = {
            "patient_metadata": req.patient_metadata.dict(),
            "physiological_data": req.physiological_data.dict() if req.physiological_data else {},
            "ecg_analysis": results["agent_results"].get("ecg_agent", {}),
            "emotion_analysis": results["agent_results"].get("emotion_agent", {}),
            "summary": results["summary"]
        }
        
        try:
            logger.info("  Generating Groq report...")
            groq_report = generate_medical_report(report_data)
            results["groq_report"] = groq_report
            logger.info("  ✓ Groq report done")
        except Exception as e:
            logger.error(f"  ✗ Groq failed: {e}")
            groq_report = f"Report generation failed: {e}"
            results["groq_report"] = groq_report
        
        # PDF Generation
        pdf_filename = f"{analysis_id}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        report_data["groq_report"] = groq_report
        report_data["analysis_id"] = analysis_id
        
        try:
            logger.info("  Generating PDF...")
            generate_pdf_report(report_data, pdf_path)
            results["pdf_url"] = f"/api/v1/reports/download/{pdf_filename}"
            logger.info("  ✓ PDF done")
        except Exception as e:
            logger.error(f"  ✗ PDF failed: {e}")
            results["pdf_url"] = None
            pdf_filename = None
        
        # Save to DB
        report = AnalysisReport(
            analysis_id=analysis_id, patient_id=patient.id, doctor_id=None,
            heart_rate=req.physiological_data.heart_rate if req.physiological_data else None,
            bp_systolic=req.physiological_data.blood_pressure_systolic if req.physiological_data else None,
            bp_diastolic=req.physiological_data.blood_pressure_diastolic if req.physiological_data else None,
            temperature=req.physiological_data.temperature if req.physiological_data else None,
            oxygen_saturation=req.physiological_data.oxygen_saturation if req.physiological_data else None,
            ecg_analysis=json.dumps(results["agent_results"].get("ecg_agent", {})),
            emotion_analysis=json.dumps(results["agent_results"].get("emotion_agent", {})),
            overall_summary=json.dumps(results["summary"]),
            priority_level=results["summary"].get("priority_level", "low"),
            requires_attention=results["summary"].get("requires_immediate_attention", False),
            groq_report=groq_report, pdf_filename=pdf_filename, 
            symptoms=json.dumps(req.symptoms) if req.symptoms else None, notes=req.notes
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        
        results["database_id"] = report.id
        results["total_processing_time_ms"] = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(f"✓ Analysis complete: {analysis_id} (DB ID: {report.id})")
        return results
        
    except Exception as e:
        logger.error(f"✗ Analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Analysis failed: {e}")

# ===== Patient History =====
@app.get("/api/v1/reports/all", tags=["Reports"])
async def get_all_reports(db: Session = Depends(get_db)):
    reports = db.query(AnalysisReport).order_by(AnalysisReport.created_at.desc()).all()
    return {
        "total_reports": len(reports),
        "reports": [{
            "id": r.id,
            "analysis_id": r.analysis_id,
            "patient": {
                "patient_id": r.patient.patient_id,
                "full_name": r.patient.full_name,
                "age": r.patient.age,
                "gender": r.patient.gender
            },
            "vital_signs": {
                "heart_rate": r.heart_rate,
                "blood_pressure": f"{r.bp_systolic}/{r.bp_diastolic}" if r.bp_systolic and r.bp_diastolic else None,
                "temperature": r.temperature,
                "oxygen_saturation": r.oxygen_saturation
            },
            "priority_level": r.priority_level,
            "requires_attention": r.requires_attention,
            "created_at": r.created_at.isoformat(),
            "pdf_url": f"/api/v1/reports/download/{r.pdf_filename}" if r.pdf_filename else None
        } for r in reports]
    }

@app.get("/api/v1/patients/{patient_id}/history", tags=["Patients"])
async def history(patient_id: str, doctor: Doctor = Depends(get_current_doctor), db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")
    reports = db.query(AnalysisReport).filter(AnalysisReport.patient_id == patient.id).order_by(AnalysisReport.created_at.desc()).all()
    return {
        "patient": {"patient_id": patient.patient_id, "full_name": patient.full_name, "age": patient.age, "gender": patient.gender},
        "total_reports": len(reports),
        "reports": [{"id": r.id, "analysis_id": r.analysis_id, "created_at": r.created_at.isoformat(),
                    "priority_level": r.priority_level, "requires_attention": r.requires_attention,
                    "doctor": r.doctor.full_name, "pdf_available": r.pdf_filename is not None} for r in reports]
    }

@app.get("/api/v1/reports/{report_id}", tags=["Reports"])
async def get_report(report_id: int, doctor: Doctor = Depends(get_current_doctor), db: Session = Depends(get_db)):
    report = db.query(AnalysisReport).filter(AnalysisReport.id == report_id).first()
    if not report:
        raise HTTPException(404, "Report not found")
    return {
        "id": report.id, "analysis_id": report.analysis_id,
        "patient": {"patient_id": report.patient.patient_id, "full_name": report.patient.full_name, "age": report.patient.age, "gender": report.patient.gender},
        "doctor": {"full_name": report.doctor.full_name, "specialization": report.doctor.specialization},
        "vital_signs": {"heart_rate": report.heart_rate, "blood_pressure": f"{report.bp_systolic}/{report.bp_diastolic}" if report.bp_systolic else None,
                       "temperature": report.temperature, "oxygen_saturation": report.oxygen_saturation},
        "ecg_analysis": json.loads(report.ecg_analysis) if report.ecg_analysis else {},
        "emotion_analysis": json.loads(report.emotion_analysis) if report.emotion_analysis else {},
        "summary": json.loads(report.overall_summary) if report.overall_summary else {},
        "groq_report": report.groq_report, "priority_level": report.priority_level, "requires_attention": report.requires_attention,
        "created_at": report.created_at.isoformat(), "pdf_url": f"/api/v1/reports/download/{report.pdf_filename}" if report.pdf_filename else None
    }

@app.get("/api/v1/reports/download/{filename}", tags=["Reports"])
async def download_pdf(filename: str):
    pdf_path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)

# ===== Chatbot Endpoint =====
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        logger.info("Gemini chatbot initialized successfully")
        GEMINI_CONFIGURED = True
    else:
        GEMINI_CONFIGURED = False
        logger.warning("GEMINI_API_KEY not found in environment variables")
except Exception as e:
    GEMINI_AVAILABLE = False
    GEMINI_CONFIGURED = False
    logger.error(f"Failed to initialize Gemini: {e}")


COGNIWAVE_CONTEXT = """
You are CogniWave AI Assistant, an intelligent healthcare chatbot for the CogniWave medical diagnostics platform.

CogniWave is a comprehensive medical diagnostics system that includes:
- ECG Analysis: Real-time ECG signal monitoring and heart rate detection
- Emotion Analysis: Facial emotion detection using camera input and stress detection from audio
- Medical Report Generation: AI-powered medical reports using GROQ API
- PDF Reports: Automatic generation of patient medical reports
- Patient Management: Database tracking of patients and their analysis history
- Appointment Scheduling: Calendar-based appointment management system
- Analytics Dashboard: Visual charts and statistics for patient health metrics
- History Tracking: Complete medical analysis history with downloadable reports

Key Features:
1. Diagnostics Page: Capture ECG signals, facial images, and audio for comprehensive health analysis
2. Schedule Page: Manage patient appointments with calendar view
3. Analytics: View trends in heart rate, emotion patterns, and health metrics
4. History: Access past medical analyses and download PDF reports

Your role:
- Answer questions about CogniWave features and functionality
- Guide users on how to use different features
- Explain medical diagnostics results in simple terms
- Help with navigation and troubleshooting
- Provide health-related information within the CogniWave context

Important: Only answer questions related to CogniWave platform, medical diagnostics, healthcare, and related topics. 
Politely decline questions outside this scope by redirecting users to CogniWave features.
"""

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None

@app.post("/api/v1/chatbot/message", tags=["Chatbot"])
async def chatbot_message(req: ChatRequest):
    try:
        if not GEMINI_CONFIGURED:
            return {
                "success": False,
                "error": "Chatbot service unavailable. Gemini API not configured."
            }
        
        logger.info(f"→ Chatbot request: {req.message[:50]}...")
        
        # Build conversation context
        conversation = COGNIWAVE_CONTEXT + "\n\nConversation:\n"
        
        # Add conversation history
        if req.conversation_history:
            for msg in req.conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation += f"{role.capitalize()}: {content}\n"
        
        # Add current user message
        conversation += f"User: {req.message}\nAssistant:"
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(conversation)
        
        ai_response = response.text
        logger.info(f"✓ Chatbot response generated")
        
        return {
            "success": True,
            "response": ai_response
        }
            
    except Exception as e:
        logger.error(f"✗ Chatbot error: {e}")
        return {
            "success": False,
            "error": f"Failed to generate response: {str(e)}"
        }

# ===== Health =====
@app.get("/api/v1/health/", tags=["System"])
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(),
            "agents": {"ecg_agent": "ready", "emotion_agent": "ready", "multimodal_emotion_session": "ready"},
            "features": {"auth": "enabled", "database": "enabled", "groq": "enabled", "pdf": "enabled"}}


def generate_summary(agent_results: Dict) -> Dict:
    priority = "low"
    status = "Normal parameters"
    attention = False
    
    ecg = agent_results.get("ecg_agent", {})
    if ecg.get("status") == "success":
        risk = ecg.get("findings", {}).get("risk_level", "low")
        if risk == "high":
            priority = "high"
            attention = True
            status = "Abnormal physiological parameters"
        elif risk == "medium":
            priority = "medium"
            status = "Elevated risk indicators"
    
    emotion = agent_results.get("emotion_agent", {})
    if emotion.get("status") == "success" and emotion.get("findings", {}).get("risk_level") == "high" and priority == "low":
        priority = "medium"
        status = "Emotional distress detected"
    
    return {"overall_status": status, "priority_level": priority, "requires_immediate_attention": attention}


# ===== Multimodal Emotion Helper Functions =====

def _safe_feature(value: Optional[float], default: float = 0.0) -> float:
    return float(value) if value is not None else default


def analyze_voice_features(voice: Optional["VoiceFeatures"]) -> Dict[str, float]:
    """
    Heuristic-based voice emotion analysis using paralinguistic cues.
    Returns probabilities over a small emotion set.
    """
    if not voice:
        return {"calm": 0.5, "stressed": 0.2, "frustrated": 0.15, "engaged": 0.15}

    pitch = _safe_feature(voice.avg_pitch)
    intensity = _safe_feature(voice.avg_intensity)
    speaking_rate = _safe_feature(voice.speaking_rate)
    pause_ratio = _safe_feature(voice.pause_ratio, default=0.3)

    stressed_score = max(0.0, min(1.0, (pitch / 300.0) * 0.4 + intensity * 0.4 + (speaking_rate / 4.0) * 0.2))
    calm_score = max(0.0, min(1.0, (1.0 - stressed_score) * (0.6 + pause_ratio * 0.4)))
    engaged_score = max(0.0, min(1.0, (speaking_rate / 4.0) * 0.5 + intensity * 0.3))
    frustrated_score = max(0.0, min(1.0, stressed_score * 0.7 + (1.0 - pause_ratio) * 0.3))

    raw = {
        "calm": calm_score,
        "stressed": stressed_score,
        "frustrated": frustrated_score,
        "engaged": engaged_score,
    }
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def analyze_audio_with_librosa(audio_data: str) -> Dict[str, float]:
    """
    Use librosa + soundfile on base64-encoded audio to estimate voice emotion.
    """
    if not audio_data:
        return {}

    try:
        import io
        import soundfile as sf  # type: ignore
        import librosa  # type: ignore

        # Strip data URL prefix if present
        if "," in audio_data:
            audio_data = audio_data.split(",", 1)[1]

        audio_bytes = base64.b64decode(audio_data)
        data, sr = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = data.mean(axis=1)

        y = data.astype("float32")

        # Basic acoustic features
        rmse = float(librosa.feature.rms(y=y).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y=y).mean())

        # Approximate pitch using librosa's piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[magnitudes > magnitudes.mean()]
        avg_pitch = float(pitch_vals.mean()) if pitch_vals.size > 0 else 150.0

        # Heuristic mapping: higher energy + higher zcr => more stressed/engaged
        energy_level = min(1.0, rmse * 10.0)
        zcr_level = min(1.0, zcr * 100.0)
        stress_raw = (energy_level * 0.6) + (zcr_level * 0.4)

        stressed = min(1.0, stress_raw)
        calm = max(0.0, 1.0 - stressed * 0.8)
        engaged = min(1.0, energy_level * 0.8 + zcr_level * 0.2)
        frustrated = stressed * 0.7

        raw = {
            "calm": calm,
            "stressed": stressed,
            "frustrated": frustrated,
            "engaged": engaged,
        }
        total = sum(raw.values()) or 1.0
        probs = {k: v / total for k, v in raw.items()}
        logger.info(f"Librosa voice probs: {probs} (rmse={rmse:.4f}, zcr={zcr:.4f}, pitch≈{avg_pitch:.1f})")
        return probs
    except Exception as e:
        logger.error(f"Librosa audio analysis failed: {e}")
        return {}


def analyze_face_features(face: Optional["FaceFeatures"]) -> Dict[str, float]:
    """
    Heuristic facial-expression / simplified action-unit analysis.
    """
    if not face:
        return {"neutral": 0.5, "happy": 0.2, "sad": 0.15, "angry": 0.15}

    smile = _safe_feature(face.smile_score)
    frown = _safe_feature(face.frown_score)
    brow = _safe_feature(face.brow_tension)
    eye = _safe_feature(face.eye_openness, default=0.5)

    happy_score = max(0.0, min(1.0, smile * 0.7 + eye * 0.3))
    sad_score = max(0.0, min(1.0, frown * 0.5 + (1.0 - eye) * 0.5))
    angry_score = max(0.0, min(1.0, frown * 0.4 + brow * 0.6))
    neutral_score = max(0.0, min(1.0, 1.0 - max(happy_score, sad_score, angry_score)))

    raw = {
        "neutral": neutral_score,
        "happy": happy_score,
        "sad": sad_score,
        "angry": angry_score
    }
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def fuse_emotions(voice_probs: Dict[str, float], face_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Simple weighted fusion of voice + face emotion probabilities.
    Face is given higher weight so visual expressions
    have stronger impact on the final mood.
    """
    emotions = set(voice_probs.keys()) | set(face_probs.keys())
    fused: Dict[str, float] = {}
    for e in emotions:
        v = voice_probs.get(e, 0.0)
        f = face_probs.get(e, 0.0)
        fused[e] = 0.3 * v + 0.7 * f
    total = sum(fused.values()) or 1.0
    return {k: v / total for k, v in fused.items()}


def build_emotion_summary(fused_probs: Dict[str, float]) -> Dict[str, Any]:
    """
    Build an overall summary / priority from fused emotions.
    """
    if not fused_probs:
        return {
            "overall_status": "No emotion data",
            "priority_level": "low",
            "requires_immediate_attention": False,
            "dominant_emotion": "unknown"
        }

    # Compute overall positive vs negative evidence
    negative_keys = {"stressed", "frustrated", "sad", "angry", "fear", "disgust"}
    positive_keys = {"happy", "engaged"}
    neutral_keys = {"neutral", "calm"}

    negative_score = sum(fused_probs.get(k, 0.0) for k in negative_keys)
    positive_score = sum(fused_probs.get(k, 0.0) for k in positive_keys)
    neutral_score = sum(fused_probs.get(k, 0.0) for k in neutral_keys)

    # Default dominant emotion = highest probability
    dominant = max(fused_probs, key=fused_probs.get)
    confidence = fused_probs.get(dominant, 0.0)

    # Safety override: if overall negative evidence outweighs positives,
    # force a negative dominant emotion even if happy/neutral are slightly higher.
    if negative_score >= positive_score and negative_score >= 0.30:
        # Pick the strongest negative emotion as dominant
        dominant_negative = max(
            negative_keys,
            key=lambda k: fused_probs.get(k, 0.0),
        )
        if fused_probs.get(dominant_negative, 0.0) > 0.0:
            dominant = dominant_negative
            confidence = fused_probs.get(dominant_negative, 0.0)

    priority = "low"
    requires_attention = False
    status = "Emotionally stable / normal range"

    high_distress_emotions = {"stressed", "frustrated", "sad", "angry"}
    if dominant in high_distress_emotions and confidence >= 0.6:
        priority = "high"
        requires_attention = True
        status = f"Significant emotional distress detected ({dominant})"
    elif dominant in high_distress_emotions:
        priority = "medium"
        status = f"Mild to moderate emotional distress ({dominant})"
    elif dominant in {"engaged", "happy"}:
        status = f"Positive, engaged emotional state ({dominant})"

    return {
        "overall_status": status,
        "priority_level": priority,
        "requires_immediate_attention": requires_attention,
        "dominant_emotion": dominant,
        "confidence": confidence
    }


def analyze_face_image_with_fer(image_data: str) -> Dict[str, float]:
    """
    Use FER model on a base64-encoded image to produce face emotion probabilities.
    Falls back to empty dict on failure.
    """
    if not fer_detector or not image_data:
        return {}

    try:
        # Strip data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(image)

        detections = fer_detector.detect_emotions(img_array)
        if not detections:
            logger.info("FER: no face detected")
            return {}

        scores = detections[0].get("emotions", {})
        if not scores:
            return {}

        # Use FER scores directly as probabilities over its emotion set
        # (neutral, happy, sad, angry, fear, disgust, surprise)
        raw = {k: float(v) for k, v in scores.items()}
        total = sum(raw.values()) or 1.0
        probs = {k: v / total for k, v in raw.items()}
        logger.info(f"FER face probs: {probs}")
        return probs
    except Exception as e:
        logger.error(f"FER face analysis failed: {e}")
        return {}

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 60)
    logger.info("CogniWave Backend v2.0")
    logger.info("Auth + Database + Groq Reports + PDF Export")
    logger.info("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
