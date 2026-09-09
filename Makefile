# `ruff` lives in mason's bin dir, which is on PATH interactively but not in every shell.
RUFF ?= ruff
UV   ?= uv

# The release targets publish whatever `pyproject.toml` says, so the version is never typed twice.
VERSION := $(shell grep -m1 '^version' pyproject.toml | cut -d '"' -f2)

.PHONY: check
check:
	@$(RUFF) check fau_tools
	@$(UV) run python -m fau_tools.test.test
	@rm -rf MNIST_*

.PHONY: build
build:
	@rm -rf dist
	@$(UV) build

# PyPI never lets a version number be reused, not even after deleting the release, and a dirty
# tree means the uploaded artifact matches no commit. Both mistakes are unfixable after upload.
# A deleted release vanishes from the API too, so the tag is the only local trace left of it:
# prefer yanking a bad release over deleting it, which keeps both the tag and the API honest.
.PHONY: guard
guard:
	@git diff-index --quiet HEAD -- || { echo "release: commit the tracked changes first"; exit 1; }
	@$(PYPI_TOKEN_CMD) >/dev/null 2>&1 || { echo "release: no token; export UV_PUBLISH_TOKEN or set PYPI_TOKEN_CMD"; exit 1; }
	@python3 -c "import json,sys,urllib.request as u; r=json.load(u.urlopen('https://pypi.org/pypi/fau-tools/json'))['releases']; sys.exit('release: $(VERSION) is already on PyPI') if '$(VERSION)' in r else 0"
	@git rev-parse -q --verify refs/tags/v$(VERSION) >/dev/null && { echo "release: v$(VERSION) is already tagged, so it was published once; PyPI keeps a deleted filename reserved forever"; exit 1; } || true
	@echo "release: $(VERSION) is free, tree is clean"

.PHONY: release
release: guard check build
	@token=$$($(PYPI_TOKEN_CMD)) && UV_PUBLISH_TOKEN=$$token $(UV) publish
	@git tag -a v$(VERSION) -m "fau-tools $(VERSION)"
	@echo "release: tagged v$(VERSION); push it with 'git push origin v$(VERSION)'"

# Same chain, stops right before the upload.
.PHONY: release-dry
release-dry: guard check build
	@$(UV) publish --dry-run
