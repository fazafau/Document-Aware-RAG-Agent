---
name: knowledge-capture
description: Capture durable project knowledge and conceptual understanding into `knowledge.md` at the repo root. Use when the user says to save or store something to knowledge, remember a project decision, preserve important facts from a Codex conversation, or capture concepts and mental models they need to understand later.
---

# Knowledge Capture

## Workflow

1. Open the current `knowledge.md` in the repo root if it exists.
2. Extract only durable facts from the conversation:
   - project decisions
   - conventions and naming choices
   - commands that should be reused
   - file locations and repository structure
   - constraints, assumptions, and open questions worth preserving
   - concepts, patterns, and mental models that improve understanding
   - tradeoffs and why a concept or approach matters
   - dependencies between ideas that the user should remember later
   - Python-specific concepts, idioms, and language mechanics worth remembering
   - library-specific facts, roles, and why a library was chosen
3. Ignore transient chat, repetition, and brainstorming that is not useful later.
4. Merge new knowledge into the existing notes instead of duplicating old entries.
5. Keep each entry short and concrete. Prefer bullets over paragraphs.
6. If a fact is revised later, update the existing line or mark the older item as superseded.

## Writing Rules

- Write to `knowledge.md` at the repo root.
- Use simple headings such as `Decisions`, `Concepts`, `Python`, `Libraries`, `Conventions`, `Commands`, `File Locations`, and `Open Questions`.
- Preserve exact names, paths, and commands when they matter.
- Record only information that should still be true when the project is reopened later.
- Do not copy the full conversation transcript.

## Suggested File Shape

Use this shape unless the repo already has a better structure:

```md
# Knowledge

## Decisions
- ...

## Concepts
- ...

## Python
- ...

## Libraries
- ...

## Conventions
- ...

## Commands
- ...

## File Locations
- ...

## Open Questions
- ...
```
