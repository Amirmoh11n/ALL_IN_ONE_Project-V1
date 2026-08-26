#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   AWS_REGION=eu-west-1 ECR_REPO=brain-tumor-web SERVICE_NAME=neura-mri ./aws/deploy_ecr_ecs_express.bash
# Required beforehand: AWS CLI configured, Docker installed, exported ONNX model present.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${AWS_REGION:=eu-west-1}"
: "${ECR_REPO:=brain-tumor-web}"
: "${SERVICE_NAME:=neura-mri}"
: "${IMAGE_TAG:=latest}"
: "${EXECUTION_ROLE_ARN:?Set EXECUTION_ROLE_ARN}"
: "${INFRASTRUCTURE_ROLE_ARN:?Set INFRASTRUCTURE_ROLE_ARN}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build -t "$IMAGE" -f docker/Dockerfile .
docker push "$IMAGE"

aws ecs create-express-gateway-service \
  --region "$AWS_REGION" \
  --service-name "$SERVICE_NAME" \
  --execution-role-arn "$EXECUTION_ROLE_ARN" \
  --infrastructure-role-arn "$INFRASTRUCTURE_ROLE_ARN" \
  --primary-container "image=$IMAGE,containerPort=8000,environment=[{name=MODEL_PATH,value=/app/artifacts/exports/brain_tumor_efficientnet_b3.onnx}]" \
  --health-check-path "/api/health/ready" \
  --cpu 2 \
  --memory 4 \
  --scaling-target '{"minTaskCount":1,"maxTaskCount":5}' \
  --monitor-resources
