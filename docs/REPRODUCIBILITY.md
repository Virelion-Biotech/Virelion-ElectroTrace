# Reproducibility

When the source tree is not a git checkout, set ELECTROTRACE_GIT_SHA in the execution environment. The CLI records unknown rather than inventing a commit identifier.

Every batch result has a manifest with source hashes, detector configuration, software version, and the recorded git SHA.
