# Enterprise Cloud Security & Compliance Policy (SEC-2026-v4)

## 1. Authentication & Access Control
- **Multi-Factor Authentication (MFA)**: Mandatory for all corporate identities accessing internal resources, code repositories, and cloud control planes. Hardware security keys (FIDO2/WebAuthn) are required for all infrastructure engineers and database administrators.
- **Session Timeout Policy**: Web console sessions automatically terminate after **15 minutes** of inactivity. CLI and API session tokens expire after a maximum of **8 hours** and require re-authentication.
- **Principle of Least Privilege**: Access to AWS and GCP production environments is granted via Just-In-Time (JIT) access requests approved by team leads, valid for at most **4 hours**.

## 2. Cryptographic Standards & Data Protection
- **Encryption at Rest**: All databases, S3 buckets, and persistent volumes must use **AES-256** encryption or customer-managed KMS keys with annual key rotation.
- **Encryption in Transit**: All network communication across public and private endpoints requires **TLS 1.3**. Legacy protocols (TLS 1.0, 1.1, SSLv3) are strictly disabled at the load balancer level.
- **Data Retention & Disposal**:
  - Financial transaction and billing records: retained for **7 years** for statutory compliance.
  - Raw HTTP access and server access logs: retained for **90 days** in hot storage, archived for **365 days**, then permanently deleted.
  - User telemetry and session metadata: retained for **180 days** before automated anonymization.

## 3. Incident Response Protocol & SLA
- **Severity 0 (P0 - Critical Breach/System Outage)**:
  - Incident commander assigned within **15 minutes**.
  - Customer notification and regulatory disclosure within **1 hour** of confirmation.
- **Severity 1 (P1 - High Risk / Partial Impairment)**:
  - Triage response within **30 minutes**.
  - Remediation plan and executive disclosure within **4 hours**.
- **Incident Post-Mortems**: Blameless post-mortem document must be published internally within **72 hours** of incident resolution.
