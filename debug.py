import os
import asyncio
import litellm
from dotenv import load_dotenv

load_dotenv()

async def main():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        print("❌ GROQ_API_KEY is empty! (Check if Windows accidentally named your file .env.txt)")
        return
        
    print(f"🔍 GROQ_API_KEY loaded successfully! (Starts with: {key[:4]}...)")
    
    try:
        print("⏳ Testing Groq API...")
        response = await litellm.acompletion(
            model="groq/llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Testing 123"}]
        )
        print("✅ SUCCESS! The model and key are working.")
    except Exception as e:
        print("\n❌ ERROR DETAILS:")
        print(str(e))

if __name__ == "__main__":
    asyncio.run(main())
