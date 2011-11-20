all: mlmethods_44814P.pdf classified.pdf best.pdf report.pdf

report.pdf: no-pruning-1000.pdf postpruned-original-1000.pdf \
	postpruned-pruned-1000.pdf prepruned-1000.pdf \
	prune-both-original-1000.pdf prune-both-original-6000.pdf \
	prune-both-pruned-1000.pdf prune-both-pruned-6000.pdf \
	postpruned-original-6000.pdf postpruned-pruned-6000.pdf \
	trsize-vs-accuracy.pdf

trees: classified-original.pdf classified-pruned.pdf

%.pdf %.aux %.log %.out %.toc %.lol %.lof: %.tex
	-pdflatex -interaction nonstopmode -halt-on-error $<
	pdflatex -interaction nonstopmode -halt-on-error $<

%.pdf: %.eps
	epstopdf $<

%.eps: %.dot
	dot -Teps $< > $@

classified.dot: decisiontree.py
	python decisiontree.py --dump-tree $<
