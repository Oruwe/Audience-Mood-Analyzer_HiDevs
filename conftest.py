"""Repo-root conftest.

Having this file here (rather than only under tests/) is what makes pytest
put the repo root on sys.path, so tests can `import ingestion...`,
`import engine...`, `import config...` etc. without installing the project
as a package.
"""
