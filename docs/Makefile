.PHONY: help Makefile

# You can set these variables from the command line.
PREPROCESS    = sphinx-apidoc
SPHINXOPTS    = -W
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PROJECTDIR    = "../skfp/"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
