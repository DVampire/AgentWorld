retriever_tool = dict(
    base_dir = "workdir/tool/retriever",
    model_name = "openrouter/o3",
    top_k = 20,
    query_mode = "naive",
    extract_metadata = True,
    require_grad = False
)