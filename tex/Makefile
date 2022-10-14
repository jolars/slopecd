MAIN_FILE=main.tex
AISTATS_VERSION=1
ARXIV_VERSION=1
OUTPUT_DIR=submissions

all: latexify

latexify: main.pdf
	latexmk ${MAIN_FILE}

aistats: latexify
	mkdir -p ${OUTPUT_DIR}
	pdftk main.pdf cat 1-10 output ${OUTPUT_DIR}/aistats-main-v${AISTATS_VERSION}.pdf
	pdftk main.pdf cat 11-end output ${OUTPUT_DIR}/aistats-supplement-v${AISTATS_VERSION}.pdf

arxiv: latexify
	mkdir -p ${OUTPUT_DIR}
	tar cvzf "${OUTPUT_DIR}/arxiv-v${ARXIV_VERSION}.tar.gz" figures/*.pdf *.tex *.sty *.bbl
