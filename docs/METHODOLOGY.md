# 🔬 Research Methodology

The **Nexus Equity Terminal** was designed as an applied research project to explore the intersection of Generative AI (specifically Large Language Models) and traditional deterministic financial modeling.

## 1. Problem Statement

Traditional qualitative financial analysis (e.g., reading 10-K risk factors, management discussion and analysis) is highly subjective and time-consuming. Conversely, quantitative modeling (like DCF valuations) relies on rigid inputs that often fail to capture nuanced strategic shifts buried in corporate filings.

## 2. Theoretical Framework

This platform attempts to bridge this gap by operationalizing **Retrieval-Augmented Generation (RAG)** as an automated Equity Research Assistant.

- **Data Grounding:** By forcing the LLM to only answer based on top-k retrieval documents from a FAISS index built dynamically from a specific company's 10-K, we drastically reduce the risk of "LLM Hallucinations" (where the model invents financial figures).
- **Structured Schema Adherence:** We utilize `Pydantic` to enforce a strict output schema. This forces the LLM to grade companies on a deterministic 1-10 scale for specific metrics (Innovation, Risk Profile, Management Tone).

## 3. Integration with Deterministic Models

Once the qualitative unstructured data is transformed into structured metrics, the terminal feeds these outputs into traditional financial models:

### Discounted Cash Flow (DCF) Integration
The LLM reads the financials and proposes a Base Free Cash Flow, a Weighted Average Cost of Capital (WACC), and a Terminal Growth Rate (g). The terminal then builds an interactive mathematical model on top of these seeds, allowing the human analyst to stress-test the AI's assumptions.

### Macroeconomic Stress Testing (Cross-Sectional Analysis)
To prevent the target asset from being analyzed in a vacuum, the terminal maintains a simulated "Macro Universe" database. This allows the newly generated competitive scores of the target company to be plotted simultaneously against industry peers, enabling relative valuation and risk analysis under changing macroeconomic conditions (e.g., Interest Rate shocks).
