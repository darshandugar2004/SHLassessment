from config import TEST_TYPE_MAPPING
import re

def extract_duration(text: str) -> int:
    if not text: return -1
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else -1

def translate_test_types(symbols_list):
    if not isinstance(symbols_list, list): 
        symbols_list = [symbols_list] if isinstance(symbols_list, str) else []
    full_names = []
    for symbol in symbols_list:
        full_name = TEST_TYPE_MAPPING.get(symbol.upper(), None)
        if full_name: full_names.append(full_name)
    return full_names

def safe_join(data, is_test_type=False):
    if is_test_type: data = translate_test_types(data)
    if isinstance(data, list):
        return ', '.join(filter(lambda x: isinstance(x, str), data))
    return str(data)

def join_description(data: dict) -> str:
    desc = data.get("description", {})
    parts = []

    for key, value in desc.items():
        if isinstance(value, list):
            joined = ", ".join(value)
            parts.append(f"{key.replace('_', ' ').title()}: {joined}")
        else:
            parts.append(f"{key.replace('_', ' ').title()}: {value}")

    return " | ".join(parts)