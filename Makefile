default: run

# Any VAR=value on the command line is forwarded to the CLI as a flag.
SHORT := f
flag = $(if $(filter $1,$(SHORT)),-$1,--$1)
arg = $(call flag,$1)$(if $(filter 1 true yes,$($1)),, "$($1)")
CLI = $(strip $(foreach v,$(.VARIABLES),$(if $(filter command line,$(origin $(v))),$(call arg,$(v)))))

run:
	uv run python -m main $(CLI)

lint:
	uv run --group dev ruff check . --fix
