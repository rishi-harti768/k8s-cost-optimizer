from openai import OpenAI
import json

client = OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key='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')
try:
    print("Testing Llama")
    comp = client.chat.completions.create(
        model='meta/llama-3.1-405b-instruct',
        messages=[{'role': 'user', 'content': 'Respond with ONLY valid JSON, e.g. {"action_type": "MAINTAIN"}'}],
        temperature=0.3, max_tokens=200
    )
    print('Llama 3.1 405b:', comp.choices[0].message.content)
except Exception as e:
    print('Error llama:', e)

try:
    print("Testing gpt-oss")
    comp2 = client.chat.completions.create(
        model='openai/gpt-oss-120b',
        messages=[{'role': 'user', 'content': 'Respond with ONLY valid JSON, e.g. {"action_type": "MAINTAIN"}'}],
        temperature=0.3, max_tokens=200
    )
    print('GPT-OSS Finish:', comp2.choices[0].finish_reason)
    print('GPT-OSS Content:', repr(comp2.choices[0].message.content))
    print('GPT-OSS Dump:', comp2.model_dump_json())
except Exception as e:
    print('Error gpt-oss:', type(e), e)
