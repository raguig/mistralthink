def smoke_cases():
    return [
        {
            "name": "basic_math",
            "query": "Compute 12 * (3 + 4)",
            "expects_tool": "calculator",
        },
        {
            "name": "recent_fact",
            "query": "Find the latest CPI release and summarize it",
            "expects_tool": "web_search",
        },
    ]

