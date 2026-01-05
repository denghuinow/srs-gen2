You are a senior requirements improvement expert.

Your task is to generate new requirements and extend positive requirements based on a given requirement base document, while avoiding generating content similar to negative-scored requirements.

## Input

### Requirement Base Document

{{R_BASE}}

### Positive Requirements List (for extension)

The following requirements received positive scores. Please generate extended requirements based on these (supplementing details, related scenarios, boundary conditions, etc.):

{{POSITIVE_UNITS}}

### Existing Requirements List (negative samples - avoid duplicates and negative requirements)

The following requirements already exist. Please ensure that newly generated requirements are distinct from these, especially avoiding content similar to negative-scored requirements:

{{NEGATIVE_POOL}}

## Requirements

1. **Generate New Requirements**:
   - Generate brand new requirements based on the requirement base document (R_base)
   - New requirements must be within the semantic scope of the requirement base document
   - New requirements should be specific, actionable, semantically clear, and independent

2. **Extend Positive Requirements**:
   - Generate extended requirements based on the positive requirements list
   - Extended requirements should supplement details, related scenarios, boundary conditions, exception handling, etc.
   - Extended requirements should be related to the original positive requirements but should not simply repeat them

3. **Avoid Negative Requirements**:
   - Ensure that newly generated requirements are clearly distinct from the content in the existing requirements list, avoiding duplicates or high similarity
   - Especially avoid generating content similar to negative-scored requirements
   - New requirements should be clearly distinct from the content in the existing requirements list

4. **Requirement Quality**:
   - All requirements should be:
     - Specific and actionable
     - Semantically clear and independent
     - Related to the requirement base document
   - Generate as many new requirements as possible, ensuring diversity and comprehensiveness

## Output Format

Do not use JSON. Output plain text directly, with each requirement on a separate line, starting with the `REQ:` prefix, as follows:

```
REQ: First new requirement text description
REQ: Second extended requirement text description
```

If a requirement needs additional explanation, place it after `REQ:` on the same line, do not wrap to a new line. Please ensure:
- Only lines starting with `REQ:` represent valid requirements
- Do not output tables, code blocks, or other additional structures except necessary titles or explanations
- New requirements are clearly distinct from existing requirements

Now please start generating new requirements and extending positive requirements:
