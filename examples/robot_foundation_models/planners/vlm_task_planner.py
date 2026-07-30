"""
VLM-Based Task Planner
======================
Uses a Vision-Language Model (VLM) for task understanding, goal
decomposition, and spatial reasoning.

Unlike the rule-based planner, this planner can handle:
- Ambiguous instructions ("clean the table")
- Multi-step tasks with dependencies
- Spatial reasoning ("pick up the leftmost cup")
- Object identification from images

Architecture::

    User Instruction + Scene Image
            ↓
    VLM (GPT-4o / Gemini / Qwen-VL)
            ↓
    Structured Task Plan (JSON)
            ↓
    Foundation Model (per sub-goal)

The VLM is called *once* at the start of an episode to produce a plan.
The foundation model is then called repeatedly to execute each sub-goal.

Status: 🟡 Tutorial — interface defined, requires API key for real use.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Reuse SubGoal and TaskPlan from rule_based_planner
import sys
sys.path.insert(0, os.path.dirname(__file__))
from rule_based_planner import SubGoal, TaskPlan


# System prompt for the VLM
VLM_SYSTEM_PROMPT = """You are a robot task planner. Given a natural language instruction and a scene image, decompose the task into a sequence of atomic sub-goals.

Each sub-goal must have:
- "action": one of ["locate", "approach", "push", "grasp", "lift", "move", "place", "release", "verify"]
- "target": the object to interact with
- "destination": where to move it (for push/place/move)
- "language": a clear, simple instruction for the VLA policy

Return ONLY a JSON array of sub-goals. Example:
[
  {"action": "locate", "target": "red cube", "language": "find the red cube"},
  {"action": "approach", "target": "red cube", "language": "move behind the red cube"},
  {"action": "push", "target": "red cube", "destination": "green zone", "language": "push the red cube to the green zone"},
  {"action": "verify", "condition": "red cube in green zone", "language": "check if the red cube is in the green zone"}
]
"""


class VLMTaskPlanner:
    """Uses a VLM to decompose tasks into sub-goals.

    Parameters
    ----------
    model_name : str
        VLM model identifier (e.g., "gpt-4o", "gemini-1.5-pro").
    api_key : str or None
        API key.  If None, reads from environment variable.
    max_tokens : int
        Maximum tokens for VLM response.
    mock : bool
        If True, fall back to rule-based planner.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        mock: bool = False,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._mock = mock

        # Resolve API key
        if api_key is None:
            if "gpt" in model_name.lower():
                api_key = os.environ.get("OPENAI_API_KEY")
            elif "gemini" in model_name.lower():
                api_key = os.environ.get("GOOGLE_API_KEY")
        self.api_key = api_key

        if not mock and not api_key:
            print(f"[VLM Planner] No API key for {model_name} — falling back to mock.")
            self._mock = True

        # Fallback planner
        from rule_based_planner import RuleBasedPlanner
        self._fallback = RuleBasedPlanner()

    def plan(
        self,
        instruction: str,
        scene_image: Optional[np.ndarray] = None,
    ) -> TaskPlan:
        """Decompose a task instruction into sub-goals.

        Parameters
        ----------
        instruction : str
            Natural-language task from the user.
        scene_image : np.ndarray or None
            Current camera observation (H, W, 3).  If provided, the VLM
            can reason about object positions.

        Returns
        -------
        TaskPlan
            Decomposed plan with sub-goals.
        """
        if self._mock:
            return self._fallback.plan(instruction)

        return self._vlm_plan(instruction, scene_image)

    def _vlm_plan(
        self,
        instruction: str,
        scene_image: Optional[np.ndarray],
    ) -> TaskPlan:
        """Call the VLM API to generate a plan."""
        try:
            if "gpt" in self.model_name.lower():
                return self._openai_plan(instruction, scene_image)
            elif "gemini" in self.model_name.lower():
                return self._gemini_plan(instruction, scene_image)
            else:
                print(f"[VLM Planner] Unsupported model {self.model_name} — fallback.")
                return self._fallback.plan(instruction)
        except Exception as e:
            print(f"[VLM Planner] API call failed: {e} — fallback to rules.")
            return self._fallback.plan(instruction)

    def _openai_plan(self, instruction: str, image: Optional[np.ndarray]) -> TaskPlan:
        """Call OpenAI GPT-4o for planning."""
        import openai
        import base64

        client = openai.OpenAI(api_key=self.api_key)

        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
        ]

        if image is not None:
            # Encode image as base64
            img_bytes = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
            # Use PIL for JPEG encoding
            from PIL import Image
            import io
            pil_img = Image.fromarray(img_bytes)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Task: {instruction}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Task: {instruction}\n(No image provided — plan based on instruction only.)",
            })

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )

        return self._parse_vlm_response(instruction, response.choices[0].message.content)

    def _gemini_plan(self, instruction: str, image: Optional[np.ndarray]) -> TaskPlan:
        """Call Google Gemini for planning."""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        if image is not None:
            img_bytes = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
            response = model.generate_content(
                [VLM_SYSTEM_PROMPT, f"Task: {instruction}", img_bytes]
            )
        else:
            response = model.generate_content(
                f"{VLM_SYSTEM_PROMPT}\n\nTask: {instruction}\n(No image provided.)"
            )

        return self._parse_vlm_response(instruction, response.text)

    def _parse_vlm_response(self, instruction: str, response_text: str) -> TaskPlan:
        """Parse VLM JSON response into TaskPlan."""
        try:
            # Extract JSON array from response
            text = response_text.strip()
            if text.startswith("```"):
                # Remove markdown code fences
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            sub_goals_raw = json.loads(text)
            sub_goals = [
                SubGoal(
                    action=sg["action"],
                    target=sg.get("target", ""),
                    destination=sg.get("destination", ""),
                    condition=sg.get("condition", ""),
                    language=sg.get("language", ""),
                )
                for sg in sub_goals_raw
            ]
            return TaskPlan(
                original_instruction=instruction,
                sub_goals=sub_goals,
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[VLM Planner] Failed to parse response: {e}")
            print(f"  Raw response: {response_text[:200]}")
            return self._fallback.plan(instruction)


if __name__ == "__main__":
    # Mock mode test
    planner = VLMTaskPlanner(mock=True)
    plan = planner.plan("push the red cube to the target")
    print(plan)
    print("\n✓ VLM planner mock test passed")
