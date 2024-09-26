import os
import sys
from dotenv import load_dotenv
import anthropic
import json

print("Script started")

def process_with_claude(content):
    print("Entering process_with_claude function")
    client = anthropic.Anthropic()
    print("Anthropic client created")
    
    try:
        print("Sending request to Claude")
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            system="You are a helpful AI assistant.",
            messages=[
                {"role": "user", "content": content},
            ]
        )
        print("Received response from Claude")
    except Exception as e:
        print(f"Error occurred while calling Claude: {str(e)}")
        return None

    print("Logging Claude response")
    log_claude_response(response)

    print("Extracting content from response")
    if isinstance(response.content, list):
        cleaned_content = ' '.join(item.text for item in response.content if hasattr(item, 'text'))
    else:
        cleaned_content = response.content
    
    cleaned_content = cleaned_content.strip()
    print(f"Cleaned content length: {len(cleaned_content)}")
    
    return cleaned_content

def log_claude_response(response):
    print("Entering log_claude_response function")
    log_file = 'zclaude.txt'
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            print(f"Writing to {log_file}")
            f.write("--- New Claude Response ---\n")
            f.write("--- Raw Response Object ---\n")
            f.write(json.dumps(response.model_dump(), indent=2))
            f.write("\n--- End of Raw Response Object ---\n")
        print(f"Successfully wrote to {log_file}")
    except Exception as e:
        print(f"Error occurred while writing to {log_file}: {str(e)}")

def main():
    print("Entering main function")
    load_dotenv()
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not found in .env file")
        return

    # First request
    content = "This is our first interaction. What's your name?"
    result = process_with_claude(content)
    if result:
        print(f"Received result from Claude. Length: {len(result)}")
    else:
        print("No result received from Claude")

    # Second request
    content = "Tell me a short joke."
    result = process_with_claude(content)
    if result:
        print(f"Received result from Claude. Length: {len(result)}")
    else:
        print("No result received from Claude")

if __name__ == "__main__":
    main()

print("Script finished")