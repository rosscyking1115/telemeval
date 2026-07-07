# telemeval — Roadmap

## v1 (current target)

See v1-scope.md. Contract + event-wise + affiliation + reports + ESA-ADB &
TimeEval readers + sklearn wrappers + registry.

## v1.x — metric hierarchy (evidence-ranked)

1. ~~**ADTQC**~~ — **shipped in v0.2** (global-timeline variant; registry
   extensibility proven). Channel-aware variant lands with subsystem work.
2. **Subsystem-aware F0.5** — adds channel->subsystem mapping input.
3. **Modified affiliation-based F0.5** (ESA-ADB variant).

## v1.x — adoption bridges (evidence-ranked)

1. xarray input adapter (optional dependency).
2. Yamcs parameter-CSV ingestion (real operator export format; ISS/VIPER
   deployments).
3. Grafana annotation JSON -> interval labels.
4. Optional HAPI/hapiclient loader (space-physics audience).
5. A telemeval MCP server exposing `evaluate` as an agent tool.

## Watched (not promised)

- PATE, LARM/ALARM metrics as consensus evolves.
- polars/Arrow-native input if user demand appears (core stays
  Arrow-friendly).

## Sustainability

- Zenodo DOI per release; CITATION.cff maintained.
- Short technical report / preprint after v1.
- Open-core door left open but unused: the library stays Apache-2.0; any
  future commercial layer would be a separate work (solicitor consult first).
