from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import requests

from aloha_rm.unified_api.decision import DecisionService, Detection


class SpeechStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(slots=True)
class YoloServiceConfig:
    endpoint: str = "http://127.0.0.1:18080/detect"
    timeout_s: float = 1.5


class YoloServiceCaller:
    def __init__(self, config: YoloServiceConfig | None = None) -> None:
        self.config = config or YoloServiceConfig()

    def run_detection(self) -> dict[str, Any]:
        response = requests.get(self.config.endpoint, timeout=self.config.timeout_s)
        response.raise_for_status()
        return response.json()


class SpeechModuleAdapter:
    """Wraps local speech command with async single-task behavior."""

    def __init__(self, command_bin: str = "speak") -> None:
        self.command_bin = command_bin
        self._status = SpeechStatus.IDLE
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def status(self) -> SpeechStatus:
        with self._lock:
            return self._status

    def play(self, text: str) -> str:
        with self._lock:
            if self._status == SpeechStatus.RUNNING:
                return "busy"
            self._status = SpeechStatus.RUNNING

        thread = threading.Thread(target=self._run_speak, args=(text,), daemon=True)
        thread.start()
        return "accepted"

    def _run_speak(self, text: str) -> None:
        try:
            self._process = subprocess.Popen(
                [self.command_bin, "--text", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._process.wait()

            with self._lock:
                if self._status == SpeechStatus.STOP_REQUESTED:
                    self._status = SpeechStatus.STOPPED
                elif self._process.returncode == 0:
                    self._status = SpeechStatus.IDLE
                else:
                    self._status = SpeechStatus.ERROR

        except Exception:
            with self._lock:
                self._status = SpeechStatus.ERROR
        finally:
            self._process = None

    def stop(self) -> str:
        with self._lock:
            if self._status in (SpeechStatus.IDLE, SpeechStatus.STOPPED):
                self._status = SpeechStatus.IDLE
                return "idle"
            if self._status != SpeechStatus.RUNNING:
                return "error"
            self._status = SpeechStatus.STOP_REQUESTED

        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

        # Keep async semantic: report stop requested first.
        time.sleep(0.02)
        return "stop_requested"


class DetectionDecisionFacade:
    def __init__(
        self,
        yolo: YoloServiceCaller | None = None,
        decision: DecisionService | None = None,
    ) -> None:
        self.yolo = yolo or YoloServiceCaller()
        self.decision = decision or DecisionService()

    def decide(self, person_ratio_threshold: float) -> bool:
        raw = self.yolo.run_detection()

        image_width = float(raw.get("image_width", 0.0))
        image_height = float(raw.get("image_height", 0.0))
        dets = raw.get("detections", [])

        detections = [
            Detection(
                class_name=str(item.get("class_name", "")),
                confidence=float(item.get("confidence", 0.0)),
                bbox=tuple(item.get("bbox", [0.0, 0.0, 0.0, 0.0]))[:4],
            )
            for item in dets
        ]

        return self.decision.person_decision(
            detections=detections,
            image_width=image_width,
            image_height=image_height,
            person_ratio_threshold=person_ratio_threshold,
        )
