import json
import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent

extend_system_message = (
	'REMEMBER the most important RULE: ALWAYS open first a new tab and go first to url wikipedia.com no matter the task!!!'
)



# or use override_system_message to completely override the system prompt
model = ChatGoogleGenerativeAI(
	api_key=SecretStr(os.getenv('GEMINI_API_KEY')),
	# model="gemini-2.0-flash",
	use_vision=True,
	save_conversation_path="logs/conversation",
	model="gemini-1.5-flash",
	temperature=0,
	max_tokens=None,
	timeout=None,
	max_retries=2
)

async def main():
	task = "do google search to find images of Elon Musk's children"
	# model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, extend_system_message=extend_system_message)

	print(
		json.dumps(
			agent.message_manager.system_prompt.model_dump(exclude_unset=True),
			indent=4,
		)
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
