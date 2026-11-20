docker build -t credit-scoring-api .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
docker run --rm -p 8000:8000 credit-scoring-api
