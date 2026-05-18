class LLMDirector:
    VALID_ACTIONS = {
        "redistribute_resources",
        "fund_public_services",
        "increase_safety_programs",
        "monitor",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/nemotron-3-super-120b-a12b:free",
        site_url: str | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._site_url = site_url
        self._endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def recommend(self, global_metrics: dict | None) -> list[dict]:
        metrics = global_metrics or {}
        prompt = self._build_prompt(metrics)
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 64,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "SYNAPSES Control Panel",
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url

        try:
            import requests

            response = requests.post(self._endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            return self._parse_response(text)
        except ModuleNotFoundError:
            return self._recommend_with_urllib(payload, headers)
        except Exception as exc:
            reason = "llm_error:unknown"
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is not None:
                reason = f"llm_http_{status_code}"
            return [{"action": "monitor", "reason": reason}]

    def _recommend_with_urllib(self, payload: dict, headers: dict) -> list[dict]:
        import json
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        body = json.dumps(payload).encode("utf-8")
        request = Request(self._endpoint, data=body, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"]
                return self._parse_response(text)
        except HTTPError as exc:
            return [{"action": "monitor", "reason": f"llm_http_{exc.code}"}]
        except (URLError, KeyError, json.JSONDecodeError):
            return [{"action": "monitor", "reason": "llm_error:transport"}]

    def _build_prompt(self, metrics: dict) -> str:
        return f"""
You are the AI Director of a socio-economic simulation.

Current metrics:
- Gini coefficient (wealth inequality): {metrics.get("gini", 0):.3f}
- Crime rate: {metrics.get("crime_rate", 0)}
- Average satisfaction: {metrics.get("average_satisfaction", 0):.1f}
- Food supply: {metrics.get("food_supply", 0)}

Choose exactly ONE intervention from this list:

- redistribute_resources
- fund_public_services
- increase_safety_programs
- monitor

Respond with ONLY the intervention name.
Do not explain your reasoning.
"""

    def _parse_response(self, text: str) -> list[dict]:
        action = text.strip().lower().replace("-", "_")
        if action not in self.VALID_ACTIONS:
            return [{"action": "monitor", "reason": "llm_fallback_invalid_action"}]
        return [{"action": action, "reason": f"llm:{self._model}"}]
