You are a senior requirements analyst.

Your task is to split a Software Requirements Specification (SRS) document into a set of independent semantic units. Each semantic unit represents an independent requirement description.

## Input

Below is an SRS document:

{{D_BASE}}

## Requirements

1. Carefully read this SRS document
2. Split it into multiple independent semantic units
3. Each semantic unit should:
   - Represent a complete, independent requirement description
   - Be semantically clear and understandable on its own
   - Avoid over-splitting (do not split a complete requirement into multiple fragments)
   - Avoid over-merging (do not merge multiple different requirements into one unit)

## Output Format

Do not use JSON. Output plain text directly, with each semantic unit on a separate line, prefixed with `REQ:`, as follows:

```
REQ: First requirement text description
REQ: Second requirement text description
```

If additional explanation is needed, it can be placed after `REQ:` on the same line. Please ensure:
- Only lines starting with `REQ:` represent valid semantic units
- Do not output tables, code blocks, or other structured formats
- Do not omit any important requirements

Now please start splitting:

