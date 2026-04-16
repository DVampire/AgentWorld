workdir = "workdir"
assets_name = "crypto"
source = None
data_type = None
level = None
tag = f"{assets_name}"
workdir = f"{workdir}/{tag}"
log_path = "agentworld.log"

processor = dict(
    type = "AggProcessor",
    procs_config = [
        dict(
            type="Processor",
            assets_name=assets_name,
            data_path=f"workdir/{assets_name}_binance_price_1day/price",
            data_type="price",
            assets_path = f"configs/_asset_list_/{assets_name}.json",
            source="binance",
            start_date = "2025-11-01",
            end_date = "2025-12-22",
            level="1day",
            format="%Y-%m-%d",
            feature_type = "Crypto",
            max_concurrent = 6,
        ),
        dict(
            type="Processor",
            assets_name=assets_name,
            data_path=f"workdir/{assets_name}_binance_price_1day/price",
            data_type="feature",
            assets_path = f"configs/_asset_list_/{assets_name}.json",
            source="binance",
            start_date = "2025-11-01",
            end_date = "2025-12-22",
            level="1day",
            format="%Y-%m-%d",
            feature_type = "Crypto",
            max_concurrent = 6,
        ),
        dict(
            type="Processor",
            assets_name=assets_name,
            data_path=f"workdir/{assets_name}_binance_price_1min/price",
            data_type="price",
            assets_path = f"configs/_asset_list_/{assets_name}.json",
            source="binance",
            start_date = "2025-11-01",
            end_date = "2025-12-22",
            level="1min",
            format="%Y-%m-%d %H:%M:%S",
            feature_type = "Crypto",
            max_concurrent = 6,
        ),
        dict(
            type="Processor",
            assets_name=assets_name,
            data_path=f"workdir/{assets_name}_binance_price_1min/price",
            data_type="feature",
            assets_path = f"configs/_asset_list_/{assets_name}.json",
            source="binance",
            start_date = "2025-11-01",
            end_date = "2025-12-22",
            level="1min",
            format="%Y-%m-%d %H:%M:%S",
            feature_type = "Crypto",
            max_concurrent = 6,
        )
    ],
    assets_path = f"configs/_asset_list_/{assets_name}.json",
    max_concurrent = 6,
    repo_id = tag,
    repo_type = "dataset",
)