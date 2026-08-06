# Security Policy

## Supported versions

The latest commit on the default branch is the supported development version. This repository has not yet published a stable release; older snapshots are not maintained with security backports.

## Report a vulnerability

Do not open a public issue for a vulnerability that could expose credentials, enable unsafe robot motion, bypass action limits, or compromise a machine running the examples.

Report it privately to [Steven.LI@connect.hku.hk](mailto:Steven.LI@connect.hku.hk) with:

- the affected file and commit;
- reproduction steps and expected impact;
- whether the issue can reach hardware or only simulation;
- a safe minimal proof of concept, if available.

An initial acknowledgement is targeted within seven days. A fix timeline depends on severity and the need for hardware-specific validation.

## Robot-safety boundary

Repository examples are educational software, not certified robot safety components. Before hardware use, operators must independently validate coordinate frames, units, limits, watchdogs, emergency stops, communication failure behavior, payload constraints, and rollback procedures. See [the validation policy](docs/VALIDATION.md) and [Sim-to-Real pipeline](docs/pipelines/07-sim-to-real.md).
