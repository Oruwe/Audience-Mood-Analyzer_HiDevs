import asyncio
import litellm
from dotenv import load_dotenv

# Force environment variables to load
load_dotenv()

# Suppress LiteLLM's internal info logs so we only see the clean results
litellm.suppress_debug_info = True 

async def main():
    print("--- Testing API Keys & Models ---\n")
    
    models = [
        "gemini/gemini-3.6-flash",
        "groq/openai/gpt-oss-20b",
    ]
    
    for model in models:
        print(f"Testing {model}...")
        try:
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=10
            )
            print(f"✅ SUCCESS: {model} is working!\n")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}\n")

if __name__ == "__main__":
    asyncio.run(main())
