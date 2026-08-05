default: run

# Any VAR=value on the command line is forwarded to the CLI as a flag.
SHORT := f
flag = $(if $(filter $1,$(SHORT)),-$1,--$1)
arg = $(call flag,$1)$(if $(filter 1 true yes,$($1)),, "$($1)")
CLI = $(strip $(foreach v,$(.VARIABLES),$(if $(filter command line,$(origin $(v))),$(call arg,$(v)))))

# torch lives in the optional cuda extra, and `uv run` syncs the environment to
# whatever it is told — so ask for the extra here, or the GPU build gets removed on
# the next run. `UV_EXTRAS= make run ...` (an env var, not a make argument, which
# would be forwarded to the CLI) skips it on machines without an NVIDIA GPU.
UV_EXTRAS ?= --extra cuda

run:
	uv run $(UV_EXTRAS) python -m main $(CLI)

lint:
	uv run --group dev ruff check . --fix

test:
	uv run --group dev pytest
