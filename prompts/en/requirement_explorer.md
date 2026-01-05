You are a senior requirements exploration expert.

Your task is to explore and generate new requirement descriptions based on a given requirement base document.

## Input

### Requirement Base Document

{{R_BASE}}

### Existing Requirements List (Negative Samples - Please Avoid Duplication)

The following requirements already exist. Please ensure that newly generated requirements do not duplicate these:

{{NEGATIVE_POOL}}

## Requirements

1. New requirements must be within the semantic scope of the requirement base document (R_base)
2. New requirements should be clearly distinct from the content in the existing requirements list, avoiding duplication or high similarity
3. New requirements should be:
   - Specific and actionable
   - Semantically clear and independent
   - Related to the requirement base document
4. Generate as many new requirements as possible, ensuring they are diverse and comprehensive

## Output Format

Do not use JSON. Output plain text directly, with each requirement on a separate line, starting with the `REQ:` prefix, as follows:

```
REQ: First new requirement text description
REQ: Second new requirement text description
```

If a requirement needs additional explanation, place it after `REQ:` on the same line, do not wrap to a new line. Please ensure:
- Only lines starting with `REQ:` represent valid requirements
- Do not output tables, code blocks, or other additional structures except necessary titles or explanations
- New requirements are clearly distinct from existing requirements

Now please start exploring new requirements:

