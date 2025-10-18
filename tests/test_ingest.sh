""" 
Make sure the container is running
./tests/test_ingest.sh

"""
set -euo pipefails
set -x

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Ingesting demo docs into ${BASE_URL} ..."

curl -sS -v -X POST "${BASE_URL}/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "demo_1",
        "title": "AI Overview",
        "text": "Artificial intelligence enables computers to perform tasks associated with human intelligence.",
        "metadata": {"source":"tests"}
      },
      {
        "id": "demo_2",
        "title": "AI in Healthcare",
        "text": "AI is used in medical imaging, triage and clinical decision support.",
        "metadata": {"source":"tests","domain":"healthcare"}
      }
    ],
    "chunk_size": 600,
    "chunk_overlap": 80
  }' \
  -w '\nHTTP %{http_code}\n'

echo "Done."
