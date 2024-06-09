# RTL_BugFixer_LLM Introduction
the full framework for automatic repair of hardware bugs using LLM

Register transfer level (RTL) bugs are critical issues that affect the functional correctness, security, and performance of systems-on-chip (SoC). 
Traditionally, repairing these bugs is a time-consuming process that requires the expertise of experienced engineers, leading to prolonged SoC development cycles and reduced vendor competitiveness. 
In this work, we propose an automated framework that utilizes a Large Language Model (LLM) to repair RTL bugs. 
We investigate four prompting strategies (each with around five variants), where each strategy incorporates different levels of context (such as simulation output) to guide the LLM. 
Our framework, using the best-performing strategy, achieves a repair success rate of 65.6% on benchmarks without prior knowledge of the bug's location or the specific steps needed for the repair.

# How to Start Working with the Repository

## Prerequisites

Before you begin, ensure you have the following installed:

- Python version 3.9.6
- PyVerilog version 1.2.1
- Synopsys VCS
- OpenAI API (preferably GPT-4)

## Download the Cirfix Repository

1. Download the Cirfix repository from GitHub [here](https://github.com/hammad-a/verilog_repair).
2. Follow the steps provided on the repository page to download it.

## Add Cirfix Folders

1. After downloading the Cirfix repository, add the Cirfix folders to the `cirfix_benchmarks_code` directory in this repository.

## Running the Code

To run the code, you need to provide several command-line arguments to choose the required strategy. Below are the arguments:

### Command-Line Arguments

- `pandas_csv_path` (str): Path to the CSV file.
- `number_iterations` (int): Number of iterations to repeat passing the same prompt to GPT.
- `choose_file` (str): Choose file name to test. Use a specific file name or "all" to process all files.
- `scenario_ID` (int): Chooses the prompt scenario.
- `experiment_number` (str): Write the experiment number.
- `feedback_logic` (int): Choose your feedback logic.

### Example Command

```bash
python3 openapi_code.py output.csv <number_of_iterations> all <choosing_prompt_scenario> <experiment_number> <feedback_logic-->future work>



