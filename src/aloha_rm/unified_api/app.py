from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from requests import RequestException

from aloha_rm.unified_api.services import DetectionDecisionFacade, SpeechModuleAdapter


class PersonDecisionRequest(BaseModel):
    person_ratio_threshold: float = Field(ge=0.0, le=1.0)


class PersonDecisionResponse(BaseModel):
    result: bool


class PlayRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50)


class PlayResponse(BaseModel):
    success: bool
    request_status: str


class StopResponse(BaseModel):
    success: bool
    request_status: str


class StatusResponse(BaseModel):
    status: str


app = FastAPI(title="Aloha RM Unified API", version="0.1.0")

detection_facade = DetectionDecisionFacade()
speech_adapter = SpeechModuleAdapter()


@app.post("/api/detect/person-decision", response_model=PersonDecisionResponse)
def person_decision(payload: PersonDecisionRequest) -> PersonDecisionResponse:
    try:
        result = detection_facade.decide(payload.person_ratio_threshold)
    except RequestException as exc:
        raise HTTPException(status_code=502, detail="MODEL_SERVICE_ERROR") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="MODEL_SERVICE_ERROR") from exc

    return PersonDecisionResponse(result=result)


@app.post("/api/speech/play", response_model=PlayResponse)
def play(payload: PlayRequest) -> PlayResponse:
    status = speech_adapter.play(payload.text.strip())
    if status == "busy":
        raise HTTPException(status_code=409, detail="SPEECH_BUSY")
    if status != "accepted":
        raise HTTPException(status_code=500, detail="SPEECH_EXEC_ERROR")
    return PlayResponse(success=True, request_status="accepted")


@app.post("/api/speech/stop", response_model=StopResponse)
def stop() -> StopResponse:
    status = speech_adapter.stop()
    if status == "error":
        raise HTTPException(status_code=500, detail="SPEECH_EXEC_ERROR")
    return StopResponse(success=True, request_status=status)


@app.get("/api/speech/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(status=speech_adapter.status().value)
