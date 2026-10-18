#!/bin/bash
# Re-run the experimental evidence. Install the pinned environment first.
set -euo pipefail
cd "$(dirname "$0")"
make test
make workshop-evidence
make followup-evidence
make followup-optional
make release
