import os
import json
from openai import OpenAI
from models import ActionType

obs_json = json.dumps({'cpu_usage_pct': 100.0, 'mem_usage_pct': 100.0, 'p99_latency_ms': 383.25, 'http_error_rate': 0.8733, 'cpu_steal_pct': 1.0, 'active_replicas': 5, 'buffer_depth': 138, 'node_size_class': 'S', 'current_hourly_cost': 15.0, 'node_bin_density': [0.4624, 0.474, 0.4841, 0.4921, 0.4974, 0.4999, 0.4992, 0.4955, 0.4889, 0.4799]}, indent=2)
available = ', '.join(a.value for a in ActionType)
prompt = f'Task: cold_start\n\nAvailable actions: {available}\n\nCurrent cluster state:\n{obs_json}\n\nRespond with ONLY valid JSON, e.g. {{"action_type": "MAINTAIN"}}'

client = OpenAI(base_url='https://integrate.api.nvidia.com/v1', api_key='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')
SYSTEM_PROMPT = 'You are a Kubernetes cost optimization expert. Analyse the cluster state and return ONLY a JSON object with one field: action_type. Choose from the available actions list provided.'

try:
    comp = client.chat.completions.create(model='openai/gpt-oss-120b', messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}], temperature=0.3, max_tokens=200)
    print("DUMP:", comp.model_dump_json())
    print("CONTENT:", repr(comp.choices[0].message.content))
except Exception as e:
    print("ERROR:", e)
