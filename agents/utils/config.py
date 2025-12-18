import os
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ClientConfig:
    base_url: str
    api_key: str
    model: str
    proxy: Optional[str] = None

class ClientRotator:
    def __init__(self, client_configs: List[ClientConfig]):
        if not client_configs:
            raise ValueError("Client configurations list cannot be empty.")
        self.client_configs = client_configs
        self.current_index = 0
        self.lock = threading.Lock()

    def get_next_client_config(self) -> ClientConfig:
        with self.lock:
            config = self.client_configs[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.client_configs)
            return config

# Load configurations from environment variables (support up to 500 configs)
evaluation_client_configs: List[ClientConfig] = []
for i in range(1, 501):  # Support CLIENT_CONFIGS_1 through CLIENT_CONFIGS_500
    base_url = os.getenv(f"CLIENT_CONFIGS_{i}_BASE_URL")
    if not base_url:
        continue

    api_key = os.getenv(f"CLIENT_CONFIGS_{i}_API_KEY")
    model = os.getenv(f"CLIENT_CONFIGS_{i}_MODEL")
    
    if not api_key or not model:
        print(f"Warning: Incomplete configuration for CLIENT_CONFIGS_{i}. Skipping.")
        continue

    proxy = os.getenv(f"CLIENT_CONFIGS_{i}_PROXY_HTTP")

    evaluation_client_configs.append(ClientConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        proxy=proxy
    ))

# Fallback to default if no environment variables are set
if not evaluation_client_configs:
    DEFAULT_API_KEYS = os.getenv("DEFAULT_API_KEYS", "AIzaSyCoXKEaKODUXLZ7W6vhgm6jN6QDpvQc9PM").split(',')
    DEFAULT_BASE_URL = os.getenv("DEFAULT_BASE_URL", "http://103.228.38.165:1337/v1")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "default")

    for key in DEFAULT_API_KEYS:
        if key:
            evaluation_client_configs.append(ClientConfig(
                base_url=DEFAULT_BASE_URL,
                api_key=key,
                model=DEFAULT_MODEL
            ))

client_rotator = None
if evaluation_client_configs:
    client_rotator = ClientRotator(evaluation_client_configs)
    print(f"✅ Loaded {len(evaluation_client_configs)} CLIENT_CONFIGS from environment")
    
    # Show endpoint distribution
    endpoint_counts = {}
    for config in evaluation_client_configs:
        endpoint_counts[config.base_url] = endpoint_counts.get(config.base_url, 0) + 1
    
    print(f"📊 Endpoint distribution:")
    for endpoint, count in endpoint_counts.items():
        endpoint_name = endpoint.split('//')[1].split('/')[0]
        print(f"   • {endpoint_name}: {count} models")
else:
    raise RuntimeError("No LLM client configurations found. Please set environment variables or provide defaults.")
