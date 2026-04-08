SHELL := /bin/bash # Use bash syntax
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
CYAN   := $(shell tput -Txterm setaf 6)
RESET  := $(shell tput -Txterm sgr0)

MAIN   := main
LATEX  := pdflatex
LFLAGS := -interaction=nonstopmode -halt-on-error

.PHONY: all pdf biber full clean hot-reload help

all: help

## PDF (quick)
pdf: ## Quick compile: pdflatex only (no bibliography updates)
	@echo "$(CYAN)→ Running pdflatex...$(RESET)"
	@$(LATEX) $(LFLAGS) $(MAIN).tex

## BIBER
biber: ## Run biber on $(MAIN).bcf (process bibliography)
	@echo "$(CYAN)→ Running biber...$(RESET)"
	@biber $(MAIN)

## FULL (complete compilation)
full: ## Complete compile: pdflatex → biber → pdflatex → pdflatex
	@echo "$(CYAN)→ First pass (generate .bcf file)...$(RESET)"
	@$(LATEX) $(LFLAGS) $(MAIN).tex
	@echo "$(CYAN)→ Processing bibliography with biber...$(RESET)"
	@biber $(MAIN)
	@echo "$(CYAN)→ Second pass (insert bibliography)...$(RESET)"
	@$(LATEX) $(LFLAGS) $(MAIN).tex
	@echo "$(CYAN)→ Final pass (resolve references)...$(RESET)"
	@$(LATEX) $(LFLAGS) $(MAIN).tex
	@echo "$(GREEN)✓ Build complete: $(MAIN).pdf$(RESET)"

## CLEAN
clean: ## Remove all generated files (.aux, .log, .bbl, .bcf, etc.)
	@echo "$(YELLOW)→ Cleaning up...$(RESET)"
	@rm -f $(MAIN).{aux,log,out,toc,bbl,blg,bcf,run.xml,pdf,fdb_latexmk,fls,synctex.gz}
	@rm -f *-blx.bib
	@rm -f *.aux *.log *.out

## HOT RELOAD
hot-reload: ## Watch $(MAIN).tex and auto-compile (single pass)
	@echo "$(CYAN)Watching $(MAIN).tex for changes...$(RESET)"
	@echo "$(YELLOW)Note: Run 'make full' manually when you change bibliography entries$(RESET)"
	@while true; do \
		ls $(MAIN).tex 2>/dev/null | entr -d make pdf; \
	done

## HELP
help: ## Show this help menu
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} { \
        if (/^[a-zA-Z_-]+:.*?##.*$$/) {printf "    ${YELLOW}%-20s${GREEN}%s${RESET}\n", $$1, $$2} \
        else if (/^## .*$$/) {printf "  ${CYAN}%s${RESET}\n", substr($$1,4)} \
        }' $(MAKEFILE_LIST)
	@echo ''
	@echo "$(CYAN)Workflow tips:$(RESET)"
	@echo "  1. Use $(YELLOW)make full$(RESET) when you change citations or bibliography"
	@echo "  2. Use $(YELLOW)make pdf$(RESET)  for quick text-only edits"
	@echo "  3. Use $(YELLOW)make hot-reload$(RESET) while writing content"

