# FinRAG: Graph-Enhanced RAG on GKE

FinRAG is an end-to-end Graph-Enhanced Retrieval-Augmented Generation (RAG) system for financial documents. This deployment runs on Google Kubernetes Engine (GKE) and combines:

- **Neo4j** as the financial knowledge graph database
- **vLLM** as an OpenAI-compatible inference server
- **Streamlit** as the user-facing RAG assistant
- **Google Cloud Storage** as the artifact/data bucket
- **Kubernetes ConfigMaps and Jobs** for lightweight ingestion

This README provides a clean, copy-pasteable “Zero to Hero” deployment path for Google Cloud Shell.

Also important, **data** folder is how lora adaptor is trained, **demo** folder is how our method as a function is added into granite model.

---

## Architecture

```text
User
 |
 | Web Preview / Port Forward
 v
Streamlit Frontend
 |
 | Retrieves context
 v
Neo4j Knowledge Graph
 |
 | Sends prompt + retrieved context
 v
vLLM OpenAI-Compatible API
 |
 v
Generated Answer
```

---

## Prerequisites

Before running the commands below, make sure you have:

1. A GKE cluster already created and selected with `kubectl`.
2. A GPU node pool available for vLLM inference.
3. NVIDIA GPU support enabled on the cluster.
4. A Kubernetes service account named `finrag-ksa`.
5. Workload Identity configured so `finrag-ksa` can access the target GCS bucket.
6. A GCS bucket named:

```text
finrag-artifacts-bucket-gke2630
```

7. Google Cloud Shell open with the correct project selected.

Check your current project:

```bash
gcloud config get-value project
```

Check that `kubectl` is connected to your cluster:

```bash
kubectl get nodes
```

---

## Important Notes

This README intentionally includes the GPU toleration required for vLLM:

```yaml
tolerations:
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"
```

Without this, the vLLM pod may remain stuck in `Pending` because Kubernetes will not schedule it onto tainted GPU nodes.

For a production deployment, do **not** hard-code passwords in Kubernetes YAML. Use Kubernetes Secrets or a managed secret store instead.

---

# Phase 1: Core Infrastructure Setup

This phase enables the required Google Cloud services and deploys the core backend services.

## 1. Enable Required GCP Services

```bash
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com
```

---

## 2. Deploy the Neo4j Knowledge Graph

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finrag-neo4j
spec:
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.19.0
        env:
        - name: NEO4J_AUTH
          value: "neo4j/FinRAG-Graph-2026!"
        ports:
        - containerPort: 7687
        - containerPort: 7474
---
apiVersion: v1
kind: Service
metadata:
  name: finrag-neo4j
spec:
  selector:
    app: neo4j
  ports:
    - name: bolt
      port: 7687
      targetPort: 7687
    - name: http
      port: 7474
      targetPort: 7474
EOF
```

Verify Neo4j is running:

```bash
kubectl get pods
kubectl get svc finrag-neo4j
```

---

## 3. Deploy the vLLM Inference Engine

This deployment includes the required GPU toleration fix so the pod can schedule onto L4 GPU nodes.

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finrag-vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      # This toleration ensures the pod schedules on your L4 GPU nodes.
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args: ["--model", "HuggingFaceH4/zephyr-7b-beta"]
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: finrag-vllm-service
spec:
  selector:
    app: vllm
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
EOF
```

Verify vLLM is scheduled and running:

```bash
kubectl get pods -l app=vllm
kubectl describe pod -l app=vllm
kubectl logs -l app=vllm
```

---

# Phase 2: Data Ingestion Pipeline

This phase uses a Kubernetes `ConfigMap` to inject the Python ingestion script directly into a Kubernetes Job. This avoids requiring local Docker access or image-build permissions.

## 1. Deploy the Data Ingestion Job

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: finrag-script
data:
  ingest.py: |
    import os
    import logging
    from google.cloud import storage
    from neo4j import GraphDatabase

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def main():
        neo4j_uri = os.environ.get("NEO4J_URI", "neo4j://finrag-neo4j.default.svc.cluster.local:7687")
        neo4j_user = os.environ.get("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "FinRAG-Graph-2026!")
        bucket_name = os.environ.get("GCP_BUCKET_NAME")

        logger.info(f"Connecting to GCS bucket: {bucket_name}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        logger.info("Successfully connected to Google Cloud Storage via Workload Identity!")

        logger.info(f"Connecting to Neo4j at {neo4j_uri}...")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        logger.info("Successfully connected to the Neo4j Graph Database!")

        dummy_doc = "AAPL_10K_2023.pdf"
        dummy_chunks = [
            "Apple Inc. reported strong Q4 earnings...",
            "Services revenue grew by 16% year-over-year...",
            "iPhone sales remained resilient in emerging markets..."
        ]

        def insert_data(tx, doc_name, chunks):
            tx.run("MERGE (d:Document {name: $doc_name})", doc_name=doc_name)
            for i, chunk_text in enumerate(chunks):
                tx.run("""
                    MATCH (d:Document {name: $doc_name})
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $text
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """, doc_name=doc_name, chunk_id=f"{doc_name}_chunk_{i}", text=chunk_text)

        with driver.session() as session:
            logger.info(f"Writing document '{dummy_doc}' and its chunks to Neo4j...")
            session.execute_write(insert_data, dummy_doc, dummy_chunks)

        logger.info("Ingestion pipeline completed successfully!")
        driver.close()

    if __name__ == "__main__":
        main()
---
apiVersion: batch/v1
kind: Job
metadata:
  name: finrag-data-ingestion
spec:
  backoffLimit: 2
  template:
    spec:
      serviceAccountName: finrag-ksa 
      restartPolicy: Never
      containers:
      - name: ingestion-worker
        image: python:3.10-slim
        command: ["/bin/sh", "-c"]
        args: ["pip install --no-cache-dir neo4j google-cloud-storage==2.16.0 && python /app/ingest.py"]
        env:
        - name: NEO4J_URI
          value: "neo4j://finrag-neo4j.default.svc.cluster.local:7687"
        - name: NEO4J_USERNAME
          value: "neo4j"
        - name: NEO4J_PASSWORD
          value: "FinRAG-Graph-2026!"
        - name: GCP_BUCKET_NAME
          value: "finrag-artifacts-bucket-gke2630"
        volumeMounts:
        - name: script-volume
          mountPath: /app
      volumes:
      - name: script-volume
        configMap:
          name: finrag-script
EOF
```

---

## 2. Wait for Ingestion to Complete

```bash
kubectl wait --for=condition=complete job/finrag-data-ingestion --timeout=120s
```

View ingestion logs:

```bash
kubectl logs -l job-name=finrag-data-ingestion
```

Expected successful output includes:

```text
Successfully connected to Google Cloud Storage via Workload Identity!
Successfully connected to the Neo4j Graph Database!
Ingestion pipeline completed successfully!
```

---

# Phase 3: RAG Frontend with Streamlit

This phase deploys a Streamlit web application that:

1. Accepts a user question.
2. Retrieves graph context from Neo4j.
3. Sends the retrieved context to vLLM.
4. Displays the generated answer.

The Streamlit server is configured to work with Cloud Shell Web Preview by disabling CORS and XSRF protection.

---

## 1. Deploy the Streamlit Application

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: finrag-streamlit-code
data:
  app.py: |
    import streamlit as st
    import os
    from neo4j import GraphDatabase
    from openai import OpenAI

    NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://finrag-neo4j.default.svc.cluster.local:7687")
    NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "FinRAG-Graph-2026!")

    VLLM_URL = "http://finrag-vllm-service:8000/v1"
    client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")

    st.set_page_config(page_title="FinRAG Assistant", page_icon="📈")
    st.title("📈 FinRAG Knowledge Assistant")
    st.write("Ask questions based on your ingested financial graph data.")

    question = st.text_input("Enter your question (e.g., 'What did Apple report for Q4?'):")

    if st.button("Generate Answer") and question:
        with st.spinner("Querying Neo4j Knowledge Graph..."):
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            query = """
            MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
            RETURN c.text AS text LIMIT 5
            """
            with driver.session() as session:
                result = session.run(query)
                context_chunks = [record["text"] for record in result]
            driver.close()
            context_str = "\n".join(context_chunks)

        with st.spinner("Generating answer via vLLM..."):
            prompt = f"Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information, answer the following question. If the answer isn't in the context, say you don't know.\n\nQuestion: {question}"

            try:
                response = client.chat.completions.create(
                    model="HuggingFaceH4/zephyr-7b-beta",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                    temperature=0.1
                )
                st.success("Done!")
                st.write("### Answer")
                st.info(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error connecting to vLLM API: {e}")

            with st.expander("🔍 View Retrieved Graph Context"):
                st.write(context_chunks)
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finrag-streamlit
spec:
  replicas: 1
  selector:
    matchLabels:
      app: streamlit
  template:
    metadata:
      labels:
        app: streamlit
    spec:
      containers:
      - name: frontend
        image: python:3.10-slim
        command: ["/bin/sh", "-c"]
        args: ["pip install --no-cache-dir streamlit neo4j openai && streamlit run /app/app.py --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false"]
        ports:
        - containerPort: 8501
        volumeMounts:
        - name: code-volume
          mountPath: /app
      volumes:
      - name: code-volume
        configMap:
          name: finrag-streamlit-code
---
apiVersion: v1
kind: Service
metadata:
  name: finrag-streamlit-service
spec:
  selector:
    app: streamlit
  ports:
    - protocol: TCP
      port: 8501
      targetPort: 8501
EOF
```

Verify the Streamlit deployment:

```bash
kubectl get pods -l app=streamlit
kubectl logs -l app=streamlit
kubectl get svc finrag-streamlit-service
```

---

## 2. Expose the Application in Cloud Shell

```bash
kubectl port-forward service/finrag-streamlit-service 8501:8501
```

Once the `port-forward` command is running:

1. Click the **Web Preview** button in Cloud Shell.
2. Select **Preview on port 8501**.
3. Open the FinRAG Streamlit UI.
4. Try this question:

```text
What did Apple report for Q4?
```

---

# Verification Checklist

Run these commands to confirm the full system is deployed correctly.

## Check All Pods

```bash
kubectl get pods
```

Expected components:

```text
finrag-neo4j
finrag-vllm
finrag-streamlit
finrag-data-ingestion
```

---

## Check Services

```bash
kubectl get svc
```

Expected services:

```text
finrag-neo4j
finrag-vllm-service
finrag-streamlit-service
```

---

## Check Ingestion Logs

```bash
kubectl logs -l job-name=finrag-data-ingestion
```

Expected result:

```text
Ingestion pipeline completed successfully!
```

---

## Check vLLM Logs

```bash
kubectl logs -l app=vllm
```

Look for output indicating that the OpenAI-compatible API server is running on port `8000`.

---

## Check Streamlit Logs

```bash
kubectl logs -l app=streamlit
```

Look for output indicating that Streamlit is running on port `8501`.

---

# Troubleshooting

## vLLM Pod Stuck in `Pending`

Check the pod events:

```bash
kubectl describe pod -l app=vllm
```

Common causes:

- No GPU node pool exists.
- GPU quota is unavailable.
- NVIDIA device plugin is not available.
- The GPU node is tainted and the pod does not tolerate the taint.

This README includes the required toleration:

```yaml
tolerations:
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"
```

---

## Ingestion Job Fails

Check the logs:

```bash
kubectl logs -l job-name=finrag-data-ingestion
```

Common causes:

- `finrag-ksa` does not exist.
- Workload Identity is not configured.
- The GCS bucket name is incorrect.
- Neo4j is not ready yet.

Check the service account:

```bash
kubectl get serviceaccount finrag-ksa
```

Check Neo4j:

```bash
kubectl get pods -l app=neo4j
kubectl logs -l app=neo4j
```

---

## Streamlit Cannot Connect to vLLM

Check vLLM service discovery:

```bash
kubectl get svc finrag-vllm-service
```

Check the vLLM pod:

```bash
kubectl get pods -l app=vllm
kubectl logs -l app=vllm
```

The Streamlit app expects vLLM at:

```text
http://finrag-vllm-service:8000/v1
```

---

## Streamlit Cannot Connect to Neo4j

Check the Neo4j service:

```bash
kubectl get svc finrag-neo4j
```

The Streamlit app expects Neo4j at:

```text
neo4j://finrag-neo4j.default.svc.cluster.local:7687
```

---

# Cleanup

To remove all FinRAG resources from the current Kubernetes namespace:

```bash
kubectl delete deployment finrag-neo4j finrag-vllm finrag-streamlit
kubectl delete service finrag-neo4j finrag-vllm-service finrag-streamlit-service
kubectl delete job finrag-data-ingestion
kubectl delete configmap finrag-script finrag-streamlit-code
```

---

# Project Structure Recommendation

If you later convert this one-shot deployment into a full GitHub repository, use this structure:

```text
finrag/
├── README.md
├── k8s/
│   ├── neo4j.yaml
│   ├── vllm.yaml
│   ├── ingestion-job.yaml
│   └── streamlit.yaml
├── app/
│   └── app.py
├── ingestion/
│   └── ingest.py
└── docs/
    └── architecture.md
```

---

# Security Improvements for Production

For a production-grade version, replace the hard-coded credentials with Kubernetes Secrets.

Recommended improvements:

- Store `NEO4J_PASSWORD` in a Kubernetes Secret.
- Pin container image versions instead of using `latest`.
- Add persistent storage for Neo4j.
- Add readiness and liveness probes.
- Use a LoadBalancer or Ingress instead of manual port forwarding.
- Add resource requests and limits for all containers.
- Add autoscaling for the frontend.
- Add observability with Cloud Logging and Cloud Monitoring.

---

# Demo Prompt

After opening the Streamlit app, test the system with:

```text
What did Apple report for Q4?
```

Expected behavior:

1. Streamlit accepts the question.
2. Neo4j returns the ingested dummy financial chunks.
3. vLLM generates an answer grounded in the retrieved context.
4. The app displays both the final answer and the retrieved graph context.

---

# License

This project is intended for educational and research use.
