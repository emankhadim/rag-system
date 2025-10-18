""" 
Make sure the container is running
./tests/test_query.sh

"""
set -euo pipefail
set -x
BASE_URL="${1:-http://localhost:8000}"

QUERY=${2:-"What are common AI applications in healthcare?"}

echo "==> POST $BASE_URL/query"
curl -sS -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"$QUERY\",
    \"top_k\": 4,
    \"return_sources\": true,
    \"max_new_tokens\": 150,
    \"temperature\": 0.0
  }" | jq .
