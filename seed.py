import redis
import json
import os
import glob
from brain.config import config

def seed():
    r = redis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=True
    )
    
    seeded_domains = []
    
    seed_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seed', '*.json'))
    for file_path in seed_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        domain_name = data.get("domain_name")
        if domain_name:
            key = f"brain:lfm:domain:{domain_name}:knowledge"
            r.set(key, json.dumps(data))
            seeded_domains.append(domain_name)
            print(f"Seeded domain: {domain_name}")
            
    print(f"Seeded domains: {', '.join(seeded_domains)}")
    print("All domain knowledge available from cold start")

if __name__ == "__main__":
    seed()
