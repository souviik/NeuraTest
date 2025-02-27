import json
import os
import sys
import asyncio
from typing import List, Dict, Optional
from gherkin.parser import Parser
from gherkin.token_scanner import TokenScanner
from pydantic import BaseModel, SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, ActionResult, Controller
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Custom output model for structured results
class TestResult(BaseModel):
    scenario_name: str
    steps: List[Dict[str, str]]  # {step: str, status: str, error: Optional[str]}

class FeatureResults(BaseModel):
    results: List[TestResult]

# Feature file parser
class FeatureParser:
    """Parses Gherkin feature files into executable scenarios"""

    def __init__(self, feature_path: str):
        self.feature_path = feature_path
        self.parser = Parser()

    def get_scenarios(self) -> List[Dict]:
        """Extract all scenarios from the feature file"""
        if not os.path.exists(self.feature_path):
            raise FileNotFoundError(f"Feature file not found: {self.feature_path}")

        with open(self.feature_path, 'r') as f:
            feature_content = f.read()

        # Parse the feature file
        feature = self.parser.parse(TokenScanner(feature_content))

        # Debugging: Print parsed structure
        print(json.dumps(feature, indent=4))

        # Extract scenarios
        scenarios = []
        for child in feature.get('feature', {}).get('children', []):
            if child.get('type') == 'Scenario':
                scenarios.append({
                    'name': child.get('name', 'Unnamed Scenario'),
                    'steps': [
                        f"{step.get('keyword', '').strip()} {step.get('text', '')}"
                        for step in child.get('steps', [])
                    ]
                })
        return scenarios

# Initialize LLM
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
            await agent.run(task=step)  # Pass the step as the task
            result.steps.append({"step": step, "status": "PASS", "error": None})
            print(f"✅ Step Passed: {step}")
        except Exception as e:
            result.steps.append({"step": step, "status": "FAIL", "error": str(e)})
            print(f"❌ Step Failed: {step} - Error: {str(e)}")
            break  # Stop execution on failure (optional: remove to continue)

    return result

async def main():
    try:
        # Load feature file
        feature_parser = FeatureParser("/Users/souvik/Documents/AI/Ai-browser-use-test/features/ProductSearch.feature")
        scenarios = feature_parser.get_scenarios()

        # Initialize agent with a dummy task (will be overridden in execute_scenario)
        agent = Agent(task="Initial task", llm=model, controller=controller)

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

    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
        print("Please ensure the feature file exists in the 'features' directory.")
    except KeyError as e:
        print(f"❌ Parsing error: Missing key in feature file - {str(e)}")
        print("Please check the feature file format.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())