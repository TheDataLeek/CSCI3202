all:
	make notes
	make review
	make clean

notes:
	jupyter nbconvert --to latex ./Notes.ipynb
	pdflatex --shell-escape --interaction=batchmode Notes.tex
	pdflatex --shell-escape --interaction=batchmode Notes.tex
	pdflatex --shell-escape --interaction=batchmode Notes.tex

review:
	cat head.tex > review.tex
	tail -n +14 Notes.tex | sed "s/\\\usepackage{geometry} //" >> review.tex
	pdflatex --shell-escape --interaction=batchmode review.tex
	pdflatex --shell-escape --interaction=batchmode review.tex
	pdflatex --shell-escape --interaction=batchmode review.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done
