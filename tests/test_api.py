"""Tests for the SYNAPSES FastAPI app."""

import importlib.util
import unittest

TESTCLIENT_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)

if TESTCLIENT_AVAILABLE:
    from fastapi.testclient import TestClient

    from synapses.api import app
else:
    TestClient = None
    app = None


@unittest.skipUnless(
    TESTCLIENT_AVAILABLE,
    "FastAPI TestClient dependencies are not installed",
)
class RunSimulationEndpointTests(unittest.TestCase):
    """Verify SYNAPSES API endpoints integrate the simulation stack."""

    def setUp(self) -> None:
        self.client = TestClient(app)


    def test_health_endpoint_reports_ok(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_metrics_endpoint_exports_prometheus_text(self) -> None:
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])

    def test_llm_mode_uses_server_side_openrouter_key(self) -> None:
        response = self.client.post(
            "/run_simulation",
            json={"num_agents": 1, "steps": 1, "director_mode": "llm", "openrouter_api_key": "client-secret"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENROUTER_API_KEY", response.json()["detail"])

    def test_run_simulation_returns_metrics_over_time(self) -> None:
        response = self.client.post(
            "/run_simulation",
            json={"num_agents": 3, "steps": 2, "tax_rate": 0.25},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        metrics = body["metrics_over_time"]
        self.assertEqual(len(metrics), 2)
        self.assertEqual([entry["step"] for entry in metrics], [1, 2])
        self.assertIn("gini", metrics[-1])
        self.assertIn("average_satisfaction", metrics[-1])
        self.assertIn("crime_history", metrics[-1])
        self.assertIn("interventions", metrics[-1])

    def test_tax_rate_defaults_to_zero(self) -> None:
        response = self.client.post(
            "/run_simulation",
            json={"num_agents": 1, "steps": 1},
        )

        self.assertEqual(response.status_code, 200)
        metrics = response.json()["metrics_over_time"]
        self.assertEqual(metrics[0]["gini"], 0.0)

    def test_run_experiment_returns_comparison_results(self) -> None:
        response = self.client.post(
            "/run_experiment",
            json={"num_agents": 3, "steps": 2, "tax_rate": 0.25},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            set(body["experiments"]),
            {"no_director", "random", "director_based"},
        )
        self.assertEqual(
            set(body["comparison"]),
            {"no_director", "random", "director_based"},
        )
        self.assertEqual(
            len(body["experiments"]["director_based"]["metrics_over_time"]),
            2,
        )

    def test_websocket_streams_simulation_steps(self) -> None:
        with self.client.websocket_connect("/ws/run_simulation") as websocket:
            websocket.send_json({"num_agents": 3, "steps": 2, "tax_rate": 0.25})

            first = websocket.receive_json()
            second = websocket.receive_json()
            complete = websocket.receive_json()

        self.assertEqual(first["type"], "step")
        self.assertEqual(first["metrics"]["step"], 1)
        self.assertIn("gini", first["metrics"])
        self.assertEqual(second["type"], "step")
        self.assertEqual(second["metrics"]["step"], 2)
        self.assertEqual(complete["type"], "complete")
        self.assertEqual(
            [entry["step"] for entry in complete["metrics_over_time"]],
            [1, 2],
        )

    def test_websocket_rejects_invalid_request(self) -> None:
        with self.client.websocket_connect("/ws/run_simulation") as websocket:
            websocket.send_json({"num_agents": 0, "steps": -1, "tax_rate": 1.5})
            error = websocket.receive_json()

        self.assertEqual(error["type"], "error")
        self.assertIn("detail", error)

    def test_cors_allows_frontend_origins_for_dev(self) -> None:
        response = self.client.options(
            "/run_simulation",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5173")
        self.assertIn("POST", response.headers["access-control-allow-methods"])

    def test_invalid_request_is_rejected(self) -> None:
        response = self.client.post(
            "/run_simulation",
            json={"num_agents": 0, "steps": -1, "tax_rate": 1.5},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
