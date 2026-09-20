"""GameFin Intelligence — RAG over gaming-industry SEC filings.

Package layout:
    gamefin.config   — typed config loading (YAML + .env)
    gamefin.ingest   — bronze layer: fetch raw filings from SEC EDGAR
    (later phases add: parse/chunk -> silver, embed/index -> gold, serve, eval)
"""

__version__ = "0.1.0"
