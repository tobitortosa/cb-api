import ulid

def new_id(prefix: str) -> str:
    return f"{prefix}_{ulid.new().str.lower()}"