#####################################
#   Makefile for a LaTeX document   #
#####################################

# Name of the file can be overrided, e.g. "make DOC=proposal"

DOC = laghos

# Rules

.SUFFIXES: .tex .pdf
.tex.pdf:
	latexmk -pdf $<

# Targets

default: pdf

pdf: ${DOC}.pdf
${DOC}.pdf: ${DOC}.tex *.tex

view: ${DOC}.pdf
	@acroread ${DOC}.pdf 2>/dev/null &
	@open ${DOC}.pdf 2>/dev/null &
	@evince ${DOC}.pdf 2>/dev/null &

clean:
	latexmk -c
	rm -f *.log *.aux *.bak *.bbl *.blg *.brf *.toc *.out *.spl ${DOC}.pdf
	rm -f *.fls *.fdb_latexmk

distclean veryclean: clean
	latexmk -C
	rm -f *~
