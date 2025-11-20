---
license: mit
---
# README

## Introduction

Download Dataset from : [https://huggingface.co/datasets/SGJQovo/AgentRecBench]

## Directory Structure

### `task/`

This folder contains three major types of recommendation tasks:

- **Classic Tasks**: These tasks focus on traditional recommendation scenarios where the goal is to provide accurate recommendations based on historical user-item interactions.
- **Evolving Interest Tasks**: These tasks are designed to capture the dynamic nature of user preferences over time, requiring agents to adapt to changing interests.
- **Cold-Start Tasks**: These tasks address the challenge of making recommendations for new users or items with limited interaction history.

### dataset

The following three folders contain the datasets for each task, processed and formatted for retrieval by the agent:

- **process_data_all**: Contains datasets for both classic and cold-start tasks.
- **process_data_long**: Contains datasets for long-term evolving interest tasks.
- **process_data_short**: Contains datasets for short-term evolving interest tasks.