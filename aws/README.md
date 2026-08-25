# AWS deployment — NeuraMRI

The application is designed so the browser only uploads the MRI image. The model stays on the server and inference runs inside the FastAPI container.

## Recommended architecture

`Browser → HTTPS Load Balancer → ECS Express Mode / Fargate → FastAPI → ONNX Runtime`

For a production setup, store the exported ONNX model in a private S3 bucket and provide `MODEL_S3_URI` plus an IAM task role with `s3:GetObject`. The container then downloads the model at startup if it is not already present.

## Local Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

Open `http://127.0.0.1:8000`.

## AWS

AWS ECS Express Mode is the preferred current managed path in this project: it creates the ECS/Fargate service, Application Load Balancer, health monitoring and autoscaling around the container. AWS documentation currently recommends ECS Express Mode for new customers instead of App Runner.

1. Build and push the image to Amazon ECR.
2. Create the ECS task execution role and Express infrastructure role.
3. Export `EXECUTION_ROLE_ARN` and `INFRASTRUCTURE_ROLE_ARN`.
4. Run:

```bash
AWS_REGION=eu-west-1 \
EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole \
INFRASTRUCTURE_ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/ecsInfrastructureRoleForExpressServices \
./aws/deploy_ecr_ecs_express.bash
```

The command prints the service URL after provisioning.

### Model in S3

Upload the export:

```bash
aws s3 cp artifacts/exports/brain_tumor_efficientnet_b3.onnx s3://YOUR_BUCKET/models/brain_tumor_efficientnet_b3.onnx
```

Then configure the container with:

```text
MODEL_S3_URI=s3://YOUR_BUCKET/models/brain_tumor_efficientnet_b3.onnx
MODEL_PATH=/tmp/models/brain_tumor_efficientnet_b3.onnx
```

Attach only `s3:GetObject` for that object/prefix to the ECS task role.
