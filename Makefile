MAIN_FILE=main.tex
AISTATS_VERSION=1
ARXIV_VERSION=1
OUTPUT_DIR=submissions
SUPPLEMENT=aistats-supplement-v${AISTATS_VERSION}.tar

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
		-o ${OUTPUT_DIR}/${SUPPLEMENT}

get-benchmark:
	mkdir -p ${OUTPUT_DIR}
	cd ${OUTPUT_DIR}; \
		if [ ! -d "benchmark_slope" ]; \
		then git clone git@github.com:Klopfe/benchmark_slope.git; \
		fi; \
		cd benchmark_slope; \
		git pull; \
		git archive HEAD . ":!.github" ":!.gitignore" ":!README.rst" \
		--prefix=benchmark/ \
		-o ../benchmark.tar
	
latexify: tex/main.pdf
	cd tex/; latexmk ${MAIN_FILE}

aistats: latexify bundle-code get-benchmark
	mkdir -p ${OUTPUT_DIR}
	pdftk tex/main.pdf cat 1-10 output ${OUTPUT_DIR}/aistats-main-v${AISTATS_VERSION}.pdf
	pdftk tex/main.pdf cat 11-end output ${OUTPUT_DIR}/appendix.pdf
	cd ${OUTPUT_DIR}; \
		tar --concatenate --file=${SUPPLEMENT} benchmark.tar; \
		tar --append --file=${SUPPLEMENT} appendix.pdf

arxiv: latexify
	mkdir -p ${OUTPUT_DIR}
	cd tex; \
		tar czf "../${OUTPUT_DIR}/arxiv-v${ARXIV_VERSION}.tar.gz" \
		figures/*.pdf *.tex *.sty *.bbl

clean:
	rm -rf ${OUTPUT_DIR}/*
