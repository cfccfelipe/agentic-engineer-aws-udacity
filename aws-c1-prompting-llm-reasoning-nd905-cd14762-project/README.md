# Purpose of This Repo

This repo is the source of truth for the course project **"Customer Support Chatbot with Amazon Bedrock AgentCore"** (Prompting for Effective LLM Reasoning). It contains the starter files students use to build the project.

> **Note:** Bedrock *Agents Classic* was closed to new customers on July 30, 2026. This project runs on its successor, the **Amazon Bedrock AgentCore managed harness**, with tools exposed through an **AgentCore Gateway**. Bedrock Evaluations, which the project uses for testing, is unaffected.

## Folder Structure

### Project Folder

The `project` folder contains all files and instructions necessary for the project:

* `project/README.md` — the full project instructions (setup, building the harness, testing, cleanup).
* `project/starter/` — the files students start from:
  * CloudFormation templates for the bug-report tool (Lambda + DynamoDB + IAM roles) and the testing resources (S3 + evaluation role)
  * Python setup scripts for the AgentCore resources (`setup_gateway.py`, `create_harness.py`), a chat client (`chat.py`), and cleanup (`cleanup_agentcore.py`)
  * `system_prompt.txt` — the student's main deliverable
  * the FAQ document, the evaluation-dataset generator, and a test-suite template

The reference solution, rubric, and detailed docs (`docs/tools-setup.md`, `docs/testing.md`) live in the companion solution repo.

### What students build

1. Deploy the tool stack (CloudFormation) and create the gateway (`setup_gateway.py`).
2. Design the system prompt: route each message to bug-report collection, FAQ answering, or a polite human hand-off; collect all bug details across a multi-turn session before filing a ticket with the `create_bug_report` tool.
3. Create the harness (`create_harness.py`), iterate with `chat.py`.
4. Test automatically: run a test suite through `generate-eval-dataset.py` and score the results with Bedrock Evaluations.
5. Clean up all resources.

All work happens in **us-east-1**, with the model pinned to `us.amazon.nova-pro-v1:0`.
