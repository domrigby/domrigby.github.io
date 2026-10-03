---
name: new-paper
description: Create a blank paper-summary template for the Paper Diary and add it to the README's Full Diary. Takes a paper name and a date (e.g. "/new-paper Attention Is All You Need, 2026-09-26" or "... a week ago"). Use when the user wants to add a new paper, blog post or article to the diary.
---

# New Paper Template

Arguments: `<paper name>` and `<date>`. If either is missing, ask. Resolve relative dates ("a week ago", "yesterday") against today's date.

## 1. Find the paper and its links
- Search the web for the paper (arXiv first). Use the **exact published title**, not the user's wording; if they differ, tell the user.
- Collect: abstract link, PDF link, and code / model / project-page links if they exist. Never invent a URL — omit any you can't confirm.
- For blog posts, use the post URL as the single link.

## 2. Check for duplicates and related notes
- `grep -ri` the README and `*.md` files for the title/arXiv ID. If a note already exists, stop and ask.
- Note any closely related existing summaries (same method family / follow-ups) to link under "Related".

## 3. Create the note
- Folder: pick the best-fitting existing topic folder (`LLM_reinforcement_learning/`, `non_LLM_reinforcement_learning/`, `marl/`, `open_endedness_and_auto_curriculums/`, `distribution_and_gpu_acceleration/`, `general_training/`, `architectures/`, `LLMs/`, `robotics/`, `finance_applications/`, `self_improvement/`). Don't create new folders without asking.
- Filename: short CamelCase description, `.md` (e.g. `ScalingUpRLProlongedTraining.md`).
- Content — leave the bullets empty for the user to fill in; do not write a summary:

```markdown
# <Exact Paper Title>

**Date:** <ordinal day> <Month> <YYYY>

[arXiv Link](<abs url>) | [PDF](<pdf url>) | [Code](<url>)

Related: [<Name>](<relative path>.md)

## Key Points
* 

## Key Methods
* 

## Results
* 

## Thoughts
* 
```
Drop the `Related:` line if there's nothing related; drop link items you couldn't find.

## 4. Add to the README Full Diary
- Entry format: `* <ordinal day>: [<Exact Paper Title>](<folder>/<File>.md)`
- Put it under the `### <Month> <YYYY>` heading in the `# 📖 Full Diary` section, in day order. If the heading doesn't exist, add it after the last month section (chronological order), before the `&#x20;&#x20;` / `---` that closes the diary.
- Change nothing else in the README.

## 5. Report
Show the note path, the README line added, and the links found (flag any you couldn't find or title mismatches).
