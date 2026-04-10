from src.graph.runtime import local_llm_adapter
# write test weather ollama is running  or not 

def test_ollama():
    config = local_llm_adapter.GraphConfig()
    adapter = local_llm_adapter.LocalLLMAdapter(config)

    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of India?"
    
    try:
        response = adapter.generate(system_prompt, user_prompt)
        
        print(f"Ollama response: {response}")
        
        print(" Ollama is running and responded correctly.")
    except Exception as exc:
        print(f" Ollama test failed: {exc}")
    
test_ollama()