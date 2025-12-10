You are a senior systems analyst.

Your task is to generate a structured Software Requirements Specification (SRS) document in Markdown format based on a set of scored and adopted requirement statements.

## Input

Below is a set of scored and adopted requirements (in JSON format), each containing text and grade:

{{UNITS_JSON}}

## Scoring Explanation

- **grade = 2**: High-quality, complete requirement descriptions. These requirements are already very well-formed. You are primarily responsible for integrating them as-is, organizing them appropriately, and categorizing them.
- **grade = 1**: Requirements that should be adopted but are relatively rough in description. These requirements are worth adopting, but need to be refined and expanded when generating the SRS.

## Processing Requirements

### For grade = 2 requirements:
- Keep their core meaning unchanged
- Organize them into appropriate sections and items
- You may slightly adjust the wording to match SRS document style, but do not change the core content

### For grade = 1 requirements:
**Must be refined and expanded**, while maintaining the original intent:
- Break down into more specific sub-requirement items
- Clarify inputs, outputs, preconditions, postconditions, or exception cases
- Improve testability and operability
- Supplement necessary details and boundary conditions

## Document Structure Requirements

1. **Reasonable sectioning**, recommended sections include:
   - Introduction/Overview
   - General Description
   - Functional Requirements (categorized by module or function)
   - Non-functional Requirements
   - Other relevant sections (as applicable)

2. **Numbering conventions**:
   - Use clear section numbering (e.g., 1, 1.1, 1.1.1)
   - Number functional requirements (e.g., REQ-1, REQ-2 or 3.1, 3.2)

3. **Completeness**:
   - Do not omit any input requirement
   - All grade = 1 requirements must be expanded and refined
   - All grade = 2 requirements must be included

4. **Format**:
   - Use Markdown format
   - Maintain clear structure and hierarchy
   - Use appropriate Markdown syntax (headings, lists, code blocks, etc.)

## Output Requirements

Please output the final SRS text (Markdown format) directly, **do not** wrap the content in JSON or other formats.

Now please start generating the SRS document:

