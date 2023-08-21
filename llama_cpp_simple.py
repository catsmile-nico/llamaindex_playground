from llama_cpp import Llama

llm = Llama(model_path="/home/catsmile/models/llama-2-7b-chat.ggmlv3.q4_1.bin", n_threads=12,n_gpu_layers=30,n_ctx=1024)

prompt = """
[INST] <<SYS>>
As an advanced language model, you can generate code as part of your responses. 
To make the code more noticeable and easier to read, please encapsulate it within triple backticks.
For instance, if you're providing Python code, wrap it as follows:

```python
print('hellow world')
```

<</SYS>>

{prompt} [/INST]
"""

def prompter(prompt):
    stream = llm.create_completion(prompt.format(prompt=prompt)
                                    , stream=True
                                    , repeat_penalty=1.1
                                    , max_tokens=0
                                    , stop=["USER:"
                                    , "ASSISTANT:"]
                                    , echo=False
                                    , temperature=0
                                    , mirostat_mode = 2
                                    , mirostat_tau=4.0
                                    , mirostat_eta=1.1)
    for output in stream:
        print(output['choices'][0]['text'],end="")

prompter("give me a matplotlib visualization code")
