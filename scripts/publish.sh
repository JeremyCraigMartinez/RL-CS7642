#!/usr/bin/env sh

dir="$(basename $PWD)"

if [[ "$dir" != "p1" ]] && [[ "$dir" != "p2" ]] && [[ "$dir" != "p3" ]]; then
    echo 'Must run in individual Project directory'
    exit 1
fi

ln -s ../Other/main-style.tex
docker run -ti --rm --volume "$PWD/..":/app -w /app/"$dir" schickling/latex pdflatex main
unlink main-style.tex
