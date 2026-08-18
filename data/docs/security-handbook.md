# Information Security Handbook

## Passwords and MFA

All employees must use the company password manager for work credentials.
Passwords must be at least 16 characters. Multi-factor authentication is
mandatory for email, the VPN, the cloud console, and any system holding customer
data.

Hardware security keys are issued to engineering and finance staff. SMS-based
codes are not accepted as a second factor for production systems.

## Data classification

Data is classified into four levels:

- **Public** — approved for external publication.
- **Internal** — default for company documents; not for external sharing.
- **Confidential** — customer data, contracts, financials. Access on a
  need-to-know basis.
- **Restricted** — credentials, encryption keys, personal identifiers of
  customers. Access requires a ticketed request and expires after 90 days.

## Device requirements

Company laptops must have full-disk encryption enabled and the endpoint agent
running. Personal devices may access email and chat only, and only through the
managed app container. Storing Confidential or Restricted data on a personal
device is prohibited.

## Incident reporting

Suspected security incidents must be reported to the security team within one
hour of discovery, through the #security-incidents channel or the on-call pager.
Do not attempt to remediate a suspected breach yourself, and do not delete logs
or affected files — preserving evidence takes priority over cleanup.

The security team acknowledges P1 incidents within 15 minutes and publishes a
written post-mortem within five business days.

## Third-party tools

Any new SaaS tool that will process Internal data or above requires a security
review before purchase. Reviews typically take five business days.
