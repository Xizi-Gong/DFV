# Magnetic/AGENTS.md

## Scope
- Work only on Magnetic-related files unless cross-project shared code is clearly involved.

## Magnetic-specific benchmark focus
- Include source-count or field-resolution scaling if supported by the code.
- Track source interaction, field update, sampling, and rendering costs when possible.
- Flag instability such as direction flips, abnormal spikes, invalid normalization, or divergent field values.

## Magnetic optimization hints
- Look for repeated field recomputation that can be cached or batched.
- Check whether arrow/heatmap/contour visualization updates can be decoupled from simulation frequency.

## Magnetic correctness constraints
- Do not change units, field direction conventions, or coordinate semantics silently.
- Do not remove source/boundary validation for performance alone.