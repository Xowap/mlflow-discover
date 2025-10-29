"""Code for our entity extraction agent"""

import json
from copy import deepcopy
from dataclasses import dataclass, field

import ollama
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

ENTITY_HIERARCHY = {
    "Location": ["Facility", "OtherLOC", "HumanSettlement", "Station"],
    "CreativeWork": ["VisualWork", "MusicalWork", "WrittenWork", "ArtWork", "Software"],
    "Group": [
        "MusicalGRP",
        "PublicCORP",
        "PrivateCORP",
        "AerospaceManufacturer",
        "SportsGRP",
        "CarManufacturer",
        "ORG",
    ],
    "Person": [
        "Scientist",
        "Artist",
        "Athlete",
        "Politician",
        "Cleric",
        "SportsManager",
        "OtherPER",
    ],
    "Product": ["Clothing", "Vehicle", "Food", "Drink", "OtherPROD"],
    "Medical": [
        "Medication/Vaccine",
        "MedicalProcedure",
        "AnatomicalStructure",
        "Symptom",
        "Disease",
    ],
}

ENTITY_TYPES = [v2 for v1 in ENTITY_HIERARCHY.values() for v2 in v1]

# Define the extraction tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_entity",
            "description": "Extract a named entity from the text with its type",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text of the entity",
                    },
                    "type": {
                        "type": "string",
                        "enum": ENTITY_TYPES,
                        "description": "The type/category of the entity",
                    },
                },
                "required": ["text", "type"],
            },
        },
    }
]


@dataclass
class ExtracT800:
    """
    This is a very simple entity extraction agent, that or might not find Sarah Connor
    """

    model_name: str = "qwen3:4b"
    tools: list = field(default_factory=lambda: deepcopy(tools))

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        """Extract entities from text using Ollama with tool calling"""

        messages = [
            {
                "role": "system",
                "content": (
                    "Your job is to extract entities from the following text, "
                    "as long as they match one of the types that the tool "
                    "accepts."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        entities = []

        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            tools=self.tools,
        )

        # Process tool calls
        if response.get("message") and response["message"].get("tool_calls"):
            for tool_call in response["message"]["tool_calls"]:
                if tool_call["function"]["name"] == "extract_entity":
                    args = tool_call["function"]["arguments"]
                    entities.append({"text": args["text"], "type": args["type"]})

        return entities


if __name__ == "__main__":
    console = Console()

    agent = ExtracT800()

    test_text = "rudolph valentino est le premier sex symbol masculin connu."
    result = agent.extract_entities(test_text)
    expected = [{"text": "rudolph valentino", "type": "Artist"}]

    panel = Panel(
        JSON(
            json.dumps(
                dict(
                    extracted=result,
                    expected=expected,
                )
            )
        ),
        title="Extracted Entities",
    )
    console.print(panel)
