# Convenience targets. See Snakefile for the dependency graph.

.PHONY: all panel analysis figures tables clean reproduce

reproduce: panel analysis tables figures ## reproduce CPU results with local inputs

panel:
	python scripts/analysis/00_build_panel.py

analysis:
	python scripts/analysis/12_main_regression.py
	python scripts/analysis/13_identification.py
	python scripts/sentiment/07b_lm_dictionary.py
	python scripts/benchmark/13_benchmark.py

tables:
	python scripts/analysis/16_make_tables.py

figures:
	python scripts/figures/make_figures.py

syncpaper:
	cp results/tables/*.tex paper/tables/
	cp results/figures/*.pdf paper/figures/

paper: syncpaper
	cd paper && latexmk -pdf main.tex && latexmk -pdf online_appendix.tex

clean:
	rm -f results/tables/*.tex results/figures/*.png results/figures/*.pdf
