# MLflow LLM fine-tuning with synthetic data

Training demo for Databricks + MLflow with LLama 3.0 LLM.

1. Load PDFs - European Financial Regulation PDFs: parse + split the PDF data and save to a Delta table `splitted_documents`
2. Generate good quality Q&A data about Capital Requirements Regulation using Chain-of-Thought reasoning.  Write the training and validation data to Delta tables: `qa_dataset_train`, `qa_dataset_val`
3. Fine-tune a Llama 3.0 8B model on the generated synthetic dataset from step 2.  Source the LLM model from the available [Databricks Foundation Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) 
4. Evaluate the model against the eval dataset.  

## Setup Notes

This Demo relies upon [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html) with [Serverless Compute](https://docs.databricks.com/en/compute/serverless/index.html).  

Databricks charges for fine-tuning LLMs are listed [here](https://www.databricks.com/product/pricing/mosaic-foundation-model-training)

Check the [regional availability](https://docs.databricks.com/en/resources/feature-region-support.html) for Model Training.

### Data Staging

The source PDFs (located in the `data` folder of this Repo) need to be staged in a Unity Catalog volume:   
`/<catalog>/<schema>/data`
 
