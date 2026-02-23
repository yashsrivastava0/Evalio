# ── Terraform skeleton for Antigravity AI Commerce ──
# This provides a starting point for Render or AWS deployment.
# Uncomment and configure the appropriate provider block.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ── Variables ──
variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "staging"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t4g.medium"
}

# ── Provider ──
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "antigravity-ai-commerce"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── NOTE: Full AWS resources (VPC, Aurora, ElastiCache, ECR, ECS)
# will be defined in Sprint 11. This is the skeleton structure.

output "info" {
  value = "Terraform skeleton ready. Configure provider and run: terraform init"
}
