# Ingress, Service Mesh, and TLS: Istio vs Certbot

This guide explains when and how to deploy Istio and Certbot across Docker Compose and Kubernetes environments.

## Architecture Comparison

| Capability | Certbot + Nginx (Docker Compose / VM) | Istio + cert-manager (Kubernetes / Cloud) |
|---|---|---|
| **Target Runtime** | Single server, Docker Compose, standalone VM | Kubernetes cluster (EKS, GKE, bare metal k8s) |
| **TLS Certificate** | Automated Let's Encrypt renewal via Certbot cron | Automated ACME issuance via `cert-manager` |
| **Internal Security** | Docker bridge network | Strict mutual TLS (mTLS) zero-trust encryption |
| **Traffic Control** | Static Nginx upstream proxy | Dynamic Canary splits, circuit breaking, retries |
| **Observability** | Access and error logs | Integrated Envoy sidecar metrics, Kiali, tracing |

## 1. Option A: Standalone Docker Deployment (Nginx + Certbot)

For standalone deployments on a Linux server:

1. Configure DNS A record pointing to server IP.
2. Initial certificate issuance:
   ```bash
   docker run -it --rm \
     -v certbot_conf:/etc/letsencrypt \
     -v certbot_www:/var/www/certbot \
     certbot/certbot certonly --webroot -w /var/www/certbot \
     -d yourdomain.com
   ```
3. Start reverse proxy with automated renewal:
   ```bash
   docker compose -f docker-compose.ssl.yml up -d
   ```

## 2. Option B: Cloud Kubernetes Deployment (Istio + cert-manager)

For enterprise microservices on Kubernetes:

1. Install Istio with default profile:
   ```bash
   istioctl install --set profile=default -y
   kubectl label namespace default istio-injection=enabled
   ```
2. Apply Cert-Manager cluster issuer and certificate:
   ```bash
   kubectl apply -f k8s/cert-manager/certificate.yaml
   ```
3. Apply Istio Ingress Gateway and mTLS Destination Rules:
   ```bash
   kubectl apply -f k8s/istio/gateway.yaml
   kubectl apply -f k8s/istio/destination-rule.yaml
   ```
