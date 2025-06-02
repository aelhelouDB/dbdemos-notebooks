# Databricks notebook source
# MAGIC %md 
# MAGIC ## Demo bundle configuration
# MAGIC Please ignore / do not delete, only used to prep and bundle the demo

# COMMAND ----------

{
  "name": "llm-fine-tuning",
  "category": "data-science",
  "custom_schema_supported": True,
  "default_catalog": "main",
  "default_schema": "dbdemos_llm_fine_tuning",
  "title": "Fine Tune your LLMs with Mosaic AI Model Training",
  "description": "Discover how to Fine Tune and deploy your own LLMs, but also how to evaluate them with MLFlow.",
  "fullDescription": "Databricks makes it easy to Fine Tune existing OSS model. In this demo, we explore how to build and serve your own fine-tuned model, and evaluating them with MLFlow.",
  "usecase": "Data Science & AI",
  "products": ["LLM", "Vector Search", "AI"],
  "related_links": [
      {"title": "View all Product demos", "url": "<TBD: LINK TO A FILTER WITH ALL DBDEMOS CONTENT>"}, 
      {"title": "Free Dolly", "url": "https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm"}],
  "recommended_items": ["sql-ai-functions", "feature-store", "mlops-end2end"],
  "demo_assets": [],
  "bundle": False,
  "tags": [{"ds": "Data Science"}],
  "notebooks": [
    {
      "path": "_resources/00-setup",
      "pre_run": False,
      "publish_on_website": False,
      "add_cluster_setup_cell": False,
      "title":  "Setup",
      "description": "Init data for demo."
    },
    {
      "path": "_resources/01-Data-Preparation-full",
      "pre_run": False,
      "publish_on_website": False,
      "add_cluster_setup_cell": False,
      "title":  "Training dataset",
      "description": "Craft training FT dataset with DBRX."
    },
    {
      "path": "01-classification-fine-tuning-customer-support",
      "pre_run": True,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Fine Tune LLM for Classification using Databricks API",
      "description": "Build the training dataset and leverage the Chat API to Fine Tune models."
    },
    {
      "path": "02-chatbot-rag-fine-tuning/02.1-llm-rag-fine-tuning",
      "pre_run": True,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Fine Tune a RAG LLM",
      "description": "Build the training dataset and leverage the Chat API to Fine Tune models for RAG application (including history)."
    },
    {
      "path": "02-chatbot-rag-fine-tuning/02.2-llm-evaluation",
      "pre_run": True, 
      "publish_on_website": True, 
      "add_cluster_setup_cell": False, 
      "title":  "Evaluate your Fine Tuned model vs baseline", 
      "description": "Leverage MLFlow Evaluate to measure the fine tuning improvements"
    },
    {
      "path": "config", 
      "pre_run": False, 
      "publish_on_website": True, 
      "add_cluster_setup_cell": False, 
      "title":  "Config file", 
      "description": "Define catalog and schema"
    },
    {
      "path": "03-entity-extraction-fine-tuning/03.1-llm-entity-extraction-drug-fine-tuning", 
      "pre_run": True, 
      "publish_on_website": True, 
      "add_cluster_setup_cell": False, 
      "title":  "Instruction Fine Tuning NER for drug extraction", 
      "description": "Specialize an existing LLM on drug extraction and evaluate its performance."
    }
  ],
  "cluster": {
    "num_workers": 0,
    "spark_conf": {
        "spark.master": "local[*, 4]"
    },
    "spark_version": "16.4.x-scala2.12",
    "single_user_name": "{{CURRENT_USER}}",
    "data_security_mode": "SINGLE_USER"
  }  
}
