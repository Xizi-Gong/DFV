# Heat/AGENTS.md

## Scope
- Work only on Heat-related files unless cross-project shared code is clearly involved.

## Heat-specific benchmark focus
- Include grid-resolution scaling if supported by the code.
- Track timestep update, solver iteration, boundary update, and heatmap rendering costs when possible.
- Flag instability such as NaN, negative-temperature artifacts if invalid in this simulator, oscillation, or unbounded growth.

## Heat optimization hints
- Look for repeated full-grid passes that can be fused.
- Look for unnecessary frontend heatmap refreshes or oversized state transfers.

## Heat correctness constraints
- Do not remove required boundary conditions or stabilization terms.
- Do not change discretization semantics without reporting error impact.