all: mlmethods_44814P.pdf classified.pdf

%.pdf %.aux %.log %.out %.toc %.lol %.lof: %.tex
	-pdflatex -interaction nonstopmode -halt-on-error $<
	pdflatex -interaction nonstopmode -halt-on-error $<

%.pdf: %.eps
	epstopdf $<

%.eps: %.dot
	dot -Teps $< > $@

classified.dot: decisiontree.py
	python decisiontree.py
