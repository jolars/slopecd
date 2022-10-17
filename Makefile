MAIN_FILE=main.tex
AISTATS_VERSION=1
ARXIV_VERSION=1
OUTPUT_DIR=submissions

all: latexify

bundle-code:
	mkdir -p ${OUTPUT_DIR}
	git archive HEAD . \
		":(exclude)tex/*" \
		":(exclude)code/expes/old/*" \
		":(exclude)Makefile" \
		":(exclude).github/" \
		":(exclude).gitignore" \
		":(exclude)code/.gitignore" \
		--format=tar.gz \
		-o ${OUTPUT_DIR}/aistats-code-v${AISTATS_VERSION}.tar.gz
	
latexify: tex/main.pdf
	cd tex/; latexmk ${MAIN_FILE}

aistats: latexify bundle-code
	mkdir -p ${OUTPUT_DIR}
	pdftk tex/main.pdf cat 1-10 output ${OUTPUT_DIR}/aistats-main-v${AISTATS_VERSION}.pdf
	pdftk tex/main.pdf cat 11-end output ${OUTPUT_DIR}/aistats-appendix-v${AISTATS_VERSION}.pdf
	cd ${OUTPUT_DIR}; \
		tar czf aistats-supplement-v${AISTATS_VERSION}.tar.gz \
			aistats-appendix-v${AISTATS_VERSION}.pdf \
			aistats-code-v${AISTATS_VERSION}.tar.gz

arxiv: latexify
	mkdir -p ${OUTPUT_DIR}
	cd tex; \
		tar cvzf "../${OUTPUT_DIR}/arxiv-v${ARXIV_VERSION}.tar.gz" figures/*.pdf *.tex *.sty *.bbl
