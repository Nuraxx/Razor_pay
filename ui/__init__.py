"""
Day-13 "Adaptive Recovery" dashboard package.

    ui/styles.py       -- color tokens + injected CSS
    ui/utils.py         -- pure formatting helpers, zero project-internal imports
    ui/data.py          -- data loaders (synthetic benchmark reports + live orchestrator runs)
    ui/components.py    -- reusable render helpers (cards, badges, timeline)
    ui/app.py            -- the Streamlit entry point (`streamlit run ui/app.py`)

One-directional import graph -- ui.app -> ui.components -> ui.data, with
ui.styles and ui.utils as leaf modules underneath both. Nothing here is ever
imported back by app.db/app.models/policy/recovery/llm/model.

The UI sits entirely on top of the existing system -- see ui/app.py's
module docstring.
"""
