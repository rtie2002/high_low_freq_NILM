$pdf_mode = 1; # Use pdflatex

# Clean up auxiliary files automatically upon completion
END {
    unlink glob "*.aux *.blg *.fls *.fdb_latexmk *.log *.out *.toc *.synctex.gz";
}
