"""HTTP layer of the debug UI: routes, form parsing and JSON encoding.

The model is not loaded (TestClient without the lifespan context), so these run
in milliseconds and cover the encoding bugs a plain function call would miss.
"""

import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.web.app import app


def page_png(text: str = "Texto de teste 123", width=900, height=300) -> bytes:
    # cv2.putText needs no font file, so the test has no external fixture
    canvas = np.full((height, width), 255, np.uint8)
    cv2.putText(canvas, text, (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
    ok, buffer = cv2.imencode(".png", canvas)
    assert ok
    return buffer.tobytes()


class WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def post(self, route: str, data=None):
        return self.client.post(
            route, files={"file": ("page.png", page_png())}, data=data
        )

    def test_index_serves_the_page(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ig-ocr", response.text)

    def test_health_reports_the_model_state(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model": False})

    def test_analyze_returns_lines_and_overlay(self) -> None:
        response = self.post("/api/analyze")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["lines"]), 1)
        self.assertTrue(body["overlay_b64"].startswith("data:image/png;base64,"))
        self.assertGreater(body["ms"], 0)

    def test_analyze_rejects_a_file_that_is_not_an_image(self) -> None:
        response = self.client.post(
            "/api/analyze", files={"file": ("page.png", b"not an image")}
        )
        self.assertEqual(response.status_code, 415)

    def test_analyze_requires_a_file(self) -> None:
        self.assertEqual(self.client.post("/api/analyze").status_code, 422)

    def test_recognize_without_a_model_is_503_not_a_crash(self) -> None:
        self.assertEqual(self.post("/api/recognize").status_code, 503)


if __name__ == "__main__":
    unittest.main()
