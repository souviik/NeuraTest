import json
import os
import sys
import asyncio
from typing import List, Dict

from dotenv import load_dotenv
from gherkin.parser import Parser
from pydantic import BaseModel, SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, ActionResult, Controller

# Add project root to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

class TestResult(BaseModel):
    scenario_name: str
    steps: List[Dict[str, str]]  # {step: str, status: str, error: Optional[str]}

class FeatureResults(BaseModel):
    results: List[TestResult]

class FeatureParser:

    def __init__(self, feature_path: str):
        self.feature_path = feature_path
        self.parser = Parser()

    def get_scenarios(self) -> List[Dict]:
        with open(self.feature_path, 'r') as f:
            feature = self.parser.parse(f.read())

        scenarios = []
        for child in feature['feature']['children']:
            if child['type'] == 'Scenario':
                scenarios.append({
                    'name': child['name'],
                    'steps': [f"{step['keyword'].strip()} {step['text']}"
                              for step in child['steps']]
                })
        return scenarios

model = ChatGoogleGenerativeAI(
    api_key=SecretStr(os.getenv('GEMINI_API_KEY')),
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Controller for structured output
controller = Controller(output_model=FeatureResults)

async def execute_scenario(agent: Agent, scenario: Dict) -> TestResult:
    """Execute a single scenario and track step results"""
    result = TestResult(scenario_name=scenario['name'], steps=[])

    for step in scenario['steps']:
        try:
            print(f"Executing Step: {step}")
            await agent.run(task=step)
            result.steps.append({"step": step, "status": "PASS", "error": None})
            print(f"✅ Step Passed: {step}")
        except Exception as e:
            result.steps.append({"step": step, "status": "FAIL", "error": str(e)})
            print(f"❌ Step Failed: {step} - Error: {str(e)}")
            break  # Stop execution on failure (optional: remove to continue)

    return result

async def main():
    # Load feature file
    feature_parser = FeatureParser("/Users/souvik/Documents/AI/Ai-browser-use-test/features/ProductSearch.feature")
    scenarios = feature_parser.get_scenarios()
    print(f"\n🚀 Successfully fetched scenarios...")

# Initialize agent
    agent = Agent(llm=model, controller=controller)

    # Execute all scenarios
    results = []
    for scenario in scenarios:
        print(f"\n🚀 Executing Scenario: {scenario['name']}")
        scenario_result = await execute_scenario(agent, scenario)
        results.append(scenario_result)
        print(f"🎯 Scenario Completed: {scenario['name']}")

    # Print final results
    final_output = FeatureResults(results=results)
    print("\n📊 Final Results:")
    print(json.dumps(final_output.model_dump(), indent=4))

if __name__ == '__main__':
    asyncio.run(main())