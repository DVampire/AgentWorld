from langchain_core.utils import secret_from_env
from pydantic import SecretStr

def get_env(env_key: str) -> SecretStr:
    return secret_from_env([env_key])()