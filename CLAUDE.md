# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ListofListofLists is a curated database of AI safety, biosecurity, and resilience projects. It aggregates solution-oriented ideas from 40+ sources (research papers, policy documents, gap analyses) into a searchable web interface.

**Data flow:** Markdown project files (in resilience-proposals) → `generate_from_resilience.py` → JSON → `index.html` web UI

## Updating Projects

The source of truth for projects is:
```
~/Desktop/Claude Code/projects/resilience-proposals/interventions-new-new/entries/
```

To update the website with new/modified projects:

```bash
# 1. Run the generator script (from ListofListofLists directory)
# This updates both projects-v2.json AND embeddings.json
python3 generate_from_resilience.py

# 2. Test locally
python3 -m http.server 8001
# Open http://localhost:8001

# 3. Commit and push
git add data/projects-v2.json data/embeddings.json
git commit -m "Update projects from resilience-proposals"
git push
```

**Important:** Do NOT edit `data/projects-v2.json` or `data/embeddings.json` directly - they are generated.

**Embedding generation requires:**
```bash
pip install sentence-transformers
```
If not installed, the script will still generate projects-v2.json but skip embeddings.

## Architecture

```
data/
  projects-v2.json       # Generated from resilience-proposals (do not edit directly)
  embeddings.json        # Generated embeddings for semantic search (do not edit directly)
  sources.json           # Source metadata (manually curated)
index.html               # Single-page web UI (vanilla JS, no build step)
generate_from_resilience.py  # Parser: resilience-proposals markdown → JSON + embeddings
sources/                 # Peregrine and other source extractions (linked from projects)
```

## Project Format (in resilience-proposals)

Each markdown file follows this structure:

```markdown
# [Title]

**Status**: [Research|Implementation/Scale|Coordination|Pilot|Idea]
**Tag**: [Science|Security|Society]

## Problem
[Description of the problem]

## Approach
[How this intervention addresses the problem]

## Current State
[What exists today, who's working on it]

## Uncertainties
[Open questions, risks, unknowns]

## Sources
- [Link text](https://example.com)

---

# RAW EXTRACTION (Original)
[Original source material - parser stops here]
```

## Key Patterns

- **Two source types**:
  1. Peregrine proposals (extracted from RAW EXTRACTION → GitHub links)
  2. Enrichment sources (from ## Sources section → external URLs)
- **Tags**: Science, Security, Society
- **Status filters**: Research, Implementation/Scale, Coordination, Pilot, Idea
- **Source links**: Peregrine links point to `sources/peregrine-2025/interventions/peregrine-XXX.md` on GitHub

## Website Features

- Search by title/content (lexical)
- Semantic search toggle (uses transformers.js + pre-computed embeddings)
- Filter by category (Science/Security/Society)
- Filter by stage (Research/Implementation/Coordination/Pilot/Idea)
- Modal with full details: Problem, Approach, Current State, Uncertainties, Next Steps, Sources
- CSV export
