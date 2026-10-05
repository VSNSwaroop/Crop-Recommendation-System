# Changelog

All notable changes to this project will be documented here.

## Unreleased

### Added
- Open-source project documentation and contribution guidance
- Security policy and environment-variable template
- CSRF protection and safer session configuration
- Authenticated `/predict` endpoint connecting the trained soil and crop models
- Upload validation and model-confidence response fields
- Soil label metadata generation during CNN training
- Lightweight tests and GitHub Actions CI

### Changed
- Admin bootstrap credentials now come from environment variables
- Flask debug mode is disabled by default
- Dependency ranges are documented
- Training scripts are more reproducible

### Removed
- Tracked runtime SQLite database
- Hard-coded default administrator password from the current codebase
