from dotenv import load_dotenv
import os
from agents import  AsyncOpenAI, OpenAIChatCompletionsModel,Agent,Runner
from agents.run import RunConfig


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)          

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)
agent = Agent(
    name= ' helpful Agent',
    instructions="""
    You are a compassionate healthcare assistant.
    - Provide clear, accurate, evidence-based health guidance.
    - Always remind users that you are NOT a doctor.
    - Encourage consulting a healthcare professional for serious concerns.
    - Help with fitness, nutrition, mental health, and healthy habits.
    - Provide reminders and motivational support.
    """,
)

print("💡 Healthcare Agent is ready! Type Your Question if any problem!")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("👋 Take care! Wishing you good health.")
        break

    result = Runner.run_sync(agent, user_input, run_config=config)

    print("\n🤖 Healthcare Agent:\n" + result.final_output.strip())