import os
import hvac
from dotenv import load_dotenv

load_dotenv(verbose=True)


class VaultClient:
    def __init__(self, url: str = None, token: str = None, secret_engine_path: str = None):
        self._client = hvac.Client(
            url=url or os.environ["VAULT_ADDR"],
            token=token or os.environ["VAULT_TOKEN"],
        )
        self._path = secret_engine_path or os.environ["SECRET_ENGINE_PATH"]
        self._cache: dict | None = None

    def _read(self) -> dict:
        """Fetch secrets from Vault, using cache if already loaded."""
        if self._cache is None:
            result = self._client.read(self._path)
            if result is None:
                raise KeyError(f"Vault path not found: {self._path}")
            self._cache = result["data"]
        return self._cache

    def register(self, *keys: str) -> None:
        """Read specified keys from the secret engine path and register them as environment variables."""
        data = self._read()
        for key in keys:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in Vault path '{self._path}'")
            os.environ[key] = data[key].strip()

    def register_all(self) -> None:
        """Read all keys from the secret engine path and register them as environment variables."""
        for key, value in self._read().items():
            os.environ[key] = value.strip()

    def get(self, key: str) -> str:
        """Retrieve a single value by key without registering it as an environment variable."""
        values = self._read()
        if key not in values:
            return ""
        return values[key].strip()


hvac_client = VaultClient()
hvac_client.register_all()