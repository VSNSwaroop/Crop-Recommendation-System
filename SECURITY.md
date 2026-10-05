# Security Policy

## Supported version

The latest code on the default branch is the version currently maintained.

## Reporting a vulnerability

Please do **not** disclose an unpatched vulnerability in a public GitHub issue.

Use GitHub's private vulnerability reporting feature if it is enabled for this repository. If private reporting is unavailable, contact the maintainer privately through the contact method listed on the maintainer's GitHub profile and include:

- the affected component
- steps to reproduce
- expected impact
- any suggested mitigation

Please do not include real user credentials, private datasets, or unnecessary personal information in a report.

## Secrets and local data

This project must not contain production secrets, real passwords, `.env` files, or runtime SQLite databases. Credentials found in Git history should be treated as compromised and replaced.
