# Reconstruction Simulation: Prompts Subsystem

## Simulation Result: ❌ FAILED

## Overview
A blind rebuild of the Prompts subsystem was attempted using *only* the provided architecture documents (`09_Prompts.md` and `Adversarial/GrillMaster_Prompts.md`). The simulation was aborted due to critical missing structural definitions, state schemas, and architectural integration details. A real software engineer cannot reconstruct the system from these documents alone.

## Critical Missing Dependencies & Schemas

### 1. Main Agent Loop / Plan Output Schema is Undefined
Section 4 of `09_Prompts.md` states that the Prompts subsystem forces the LLM into "strict output schemas (like JSON for GUI targeting or specific plan steps) required by `TaskPlanner`." While the document defines the schema for GUI targeting (`found, x, y, width, height, confidence, reason`), it **completely omits the schema for the plan steps**. 
* **Failure Point:** What specific JSON keys are expected by the `TaskPlanner`? Does the model need to output an array of `steps`? Does each step require `tool_name`, `arguments`, `thought`, or `dependencies`? Without knowing the strict contract the prompt must enforce, the core agent planning prompt cannot be written.

### 2. Memory and State Injection Schemas are Opaque
Section 3 mentions that "Prompts format memory episodes and preference states to inject them into the active context window." However, there is absolutely no definition of these data structures.
* **Failure Point:** How are "memory episodes" structured before they are converted to text? What defines a "preference state"? Without the schemas for these underlying state objects, an engineer cannot create the `f-string` parsing logic that translates the state into the prompt window.

### 3. Tool Exposure Formatting
The documentation discusses the use of specialized capabilities and web tools, but it fails to define how the system makes the main LLM aware of these tools.
* **Failure Point:** Are the tools injected into the `JARVIS_SYSTEM` prompt as a text block (e.g., a standard ReAct or system-description format)? Or does the system assume native API tool calling (like OpenAI function calling)? If the tools are injected into the prompt context, the template for rendering tool signatures and descriptions is entirely missing.

### 4. Error Correction and Retry Prompts
As highlighted by the `GrillMaster_Prompts.md` adversarial critique, the current architecture lacks failure fallbacks. Crucially, the documentation does not define any prompts for error recovery.
* **Failure Point:** When a zero-shot prompt (like web extraction or GUI targeting) returns malformed JSON, what is the exact prompt structure used to auto-correct the LLM? An engineer would need to know the schema for feeding execution errors and parser tracebacks back into the context window.

## Conclusion
The extraction package for the Prompts domain relies too heavily on conceptual instructions (e.g., "be technical", "summarize search results") and completely fails to define the programmatic interfaces—the exact string templates, JSON schemas, and state objects—that the prompts are meant to bridge. The package fails the blind rebuild criteria.
