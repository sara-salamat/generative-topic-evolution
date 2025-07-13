import requests
import json

def check_ollama_models():
    """Check what models are available in Ollama"""
    try:
        # Get list of models
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        
        models = response.json()
        print("=== Available Ollama Models ===")
        
        if "models" in models:
            for model in models["models"]:
                print(f"Model: {model['name']}")
                print(f"  Size: {model.get('size', 'Unknown')}")
                print(f"  Modified: {model.get('modified_at', 'Unknown')}")
                print()
        else:
            print("No models found or unexpected response format")
            print(f"Response: {json.dumps(models, indent=2)}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama at http://localhost:11434")
        print("Make sure Ollama is running with: ollama serve")
    except Exception as e:
        print(f"❌ Error checking models: {e}")

def check_ollama_status():
    """Check if Ollama is running and get basic info"""
    try:
        response = requests.get("http://localhost:11434/api/version")
        response.raise_for_status()
        
        version_info = response.json()
        print("=== Ollama Status ===")
        print(f"Version: {version_info.get('version', 'Unknown')}")
        print("✅ Ollama is running")
        
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running")
        print("Start it with: ollama serve")
    except Exception as e:
        print(f"❌ Error checking status: {e}")

if __name__ == "__main__":
    check_ollama_status()
    print()
    check_ollama_models() 