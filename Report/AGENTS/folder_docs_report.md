ISSUE ID: DOCS-001
SEVERITY: Medium
CATEGORY: Documentation mismatches
FILES: d:\AI\Jarvis\docs\design_doc.md
DESCRIPTION: The design document embeds images using hardcoded, absolute local file URIs that point to a temporary AI workspace, which will be broken for any other user or system.
ROOT CAUSE: Images were referenced directly from an LLM chat brain directory during document creation, rather than being saved into the repository as persistent relative assets.
EVIDENCE: 
`![Jarvis System Architecture Infographic](file:///C:/Users/FEZ%20NASEER/.gemini/antigravity/brain/b174c800-fc53-4e9e-9930-11744ec2b80d/system_architecture_1780132164075.png)`
POTENTIAL IMPACT: Users reading the documentation will not be able to view the embedded diagrams, resulting in a loss of critical visual context.
RECOMMENDED FIX: Move the image assets into a dedicated directory (e.g., `docs/images/`) within the repository and update the markdown file to use relative links (e.g., `![Jarvis System Architecture Infographic](images/system_architecture_1780132164075.png)`).

ISSUE ID: DOCS-002
SEVERITY: Low
CATEGORY: Documentation mismatches
FILES: d:\AI\Jarvis\docs\design_doc.md
DESCRIPTION: The "Codebase Directory Hierarchy" section is outdated and missing several directories and files that exist in the actual `core/` module.
ROOT CAUSE: The architecture document's directory tree block was not updated to reflect recent additions and refactoring within the `core/` module.
EVIDENCE: The documentation tree fails to list the `controller_v2.py` file (which is mentioned elsewhere in the same doc) and numerous active directories such as `config/`, `context/`, `logging/`, and `ops/`.
POTENTIAL IMPACT: New developers may experience confusion due to discrepancies between the documentation and the actual repository structure, hindering onboarding.
RECOMMENDED FIX: Update the directory hierarchy tree block in `design_doc.md` to accurately reflect the current state of the repository.

ISSUE ID: DOCS-003
SEVERITY: Low
CATEGORY: Syntax issues
FILES: d:\AI\Jarvis\docs\design_doc.md
DESCRIPTION: There is a Mermaid diagram syntax error and broken linkage in the "System Bootstrapping Flow" flowchart due to a mismatch between a subgraph ID and its reference.
ROOT CAUSE: The subgraph is declared with an ID containing a space (`subgraph Service Pool`), but it is referenced later without the space (`Controller --> ServicePool`).
EVIDENCE: 
```mermaid
    subgraph Service Pool [Core Services Bootstrapping]
...
    Controller --> ServicePool
```
POTENTIAL IMPACT: The architecture diagram may fail to render in standard markdown viewers, or it will render incorrectly by creating a disconnected node named "ServicePool" instead of linking to the subgraph.
RECOMMENDED FIX: Rename the subgraph ID to remove the space so it matches the reference: `subgraph ServicePool [Core Services Bootstrapping]`.
