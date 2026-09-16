# Audience Mood Analyzer Explainability

## Agent Decision Reasoning
The agent determines audience sentiment trends by clustering normalized comments through an LLM evaluation pipeline. It decides the emotional classification by measuring semantic consensus across grouped user feedback.

## Data Inputs
The primary data source is the YouTube Data API, providing video metadata, comment threads, and engagement statistics. Additionally, it takes developer configuration parameters to structure the batch ingestion runs into PostgreSQL.

## Known Limitations
One major constraint is that the agent cannot process comments on videos with disabled or private interaction settings. Another known issue is that sarcasm and colloquial slang can occasionally result in edge-case sentiment misclassifications.