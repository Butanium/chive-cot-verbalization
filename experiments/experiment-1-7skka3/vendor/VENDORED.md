# Vendored subtree of adamkarvonen/chive

Source: https://github.com/adamkarvonen/chive, commit 2423952f523e5302c7030943c0b8946f63df58f8 (fetched 2026-09-07).
Files copied byte-for-byte (no modifications); sampling constants untouched:

- chive/__init__.py, chive/paths.py
- chive/pipeline/__init__.py, config.py, llm_client.py, utils.py, tool_validate.py, generate_completions.py

The behavior-grader prompt, tool schema, and user-content builder from chive/pipeline/investigate.py
(GRADER_SYSTEM_PROMPT, GRADER_TOOL, build_grader_user_content, grade_completions, lines ~957-1076)
are reproduced verbatim in ../src/grader.py because investigate.py itself pulls in the whole investigator agent.
