# RTL_BugFixer_LLM
the full framework for automatic repair of hardware bugs using LLM
Register transfer level (RTL) bugs are critical issues that affect the functional correctness, security, and performance of systems-on-chip (SoC). 
Traditionally, repairing these bugs is a time-consuming process that requires the expertise of experienced engineers, leading to prolonged SoC development cycles and reduced vendor competitiveness. 
In this work, we propose an automated framework that utilizes a Large Language Model (LLM) to repair RTL bugs. We investigate four prompting strategies (each with around five variants), where each strategy incorporates different levels of context (such as simulation output) to guide the LLM. Our framework achieves a repair success rate of 65.6% on benchmarks without prior knowledge of the bug's location or the specific steps needed for the repair.
