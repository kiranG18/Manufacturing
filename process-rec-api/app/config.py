# app/config.py
#
# Application settings loaded from environment variables.
# This pattern keeps secrets out of source code and makes the service
# trivially configurable across dev, staging, and production environments.

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Model artifact paths
    classifier_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "CLASSIFIER_MODEL_PATH", "models/classifier_v2.pkl"
        )
    )
    cost_models_dir: str = field(
        default_factory=lambda: os.environ.get(
            "COST_MODELS_DIR", "models/cost_v2/"
        )
    )

    # Material price feed
    price_feed_url: str = field(
        default_factory=lambda: os.environ.get("PRICE_FEED_URL", "")
    )
    price_feed_api_key: str = field(
        default_factory=lambda: os.environ.get("PRICE_FEED_API_KEY", "")
    )
    price_cache_ttl_hours: int = field(
        default_factory=lambda: int(os.environ.get("PRICE_CACHE_TTL_HOURS", "1"))
    )

    # Logging and versioning
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO")
    )
    model_version_tag: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL_VERSION_TAG", "classifier_v2_cost_v2"
        )
    )

    # Service behaviour
    default_top_k: int = 3
    max_top_k: int = 5
    request_timeout_seconds: int = 30


# Module-level singleton — imported by all modules that need configuration
settings = Settings()
