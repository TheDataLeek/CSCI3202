all:
	jupyter nbconvert --to latex ./Notes.ipynb
	cat head.tex > review.tex
	tail -n +14 Notes.tex | sed "s/\\\usepackage{geometry} //" >> review.tex
	pdflatex --shell-escape review.tex
	pdflatex --shell-escape review.tex
	pdflatex --shell-escape review.tex

clean:
	for FILE in `find . -type f | grep -e "\.pyg\|\.aux\|\.log\|\.toc\|\.out\|\.equ\|\.lof\|\.tar\.gz"`; do rm $$FILE; done
