.PHONY: help prepare-dev test check_all updatetools test-disable-gpu doc ipynb-to-rst
.DEFAULT: help

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests and linting on py36, py37, py38"
	@echo "make check_all"
	@echo "       check all files using pre-commit tool"
	@echo "make updatetools"
	@echo "       updatetools pre-commit tool"
	@echo "make test-disable-gpu"
	@echo "       run test with gpu disabled"
	@echo "make doc"
	@echo "       build Sphinx docs documentation"
	@echo "ipynb-to-rst"
	@echo "       Transform notebooks to .rst files in documentation and generate the doc"

prepare-dev:
	python3 -m pip install virtualenv
	python3 -m venv deel_lip_dev_env
	. deel_lip_dev_env/bin/activate && pip install --upgrade pip
	. deel_lip_dev_env/bin/activate && pip install -r requirements.txt
	. deel_lip_dev_env/bin/activate && pip install -r requirements_dev.txt
	. deel_lip_dev_env/bin/activate && pre-commit install
	. deel_lip_dev_env/bin/activate && pre-commit install-hooks
	. deel_lip_dev_env/bin/activate && pre-commit install --hook-type commit-msg

test:
	. deel_lip_dev_env/bin/activate && tox

check_all:
	. deel_lip_dev_env/bin/activate && pre-commit run --all-files

updatetools:
	. deel_lip_dev_env/bin/activate && pre-commit autoupdate

test-disable-gpu:
	. deel_lip_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox

doc:
	. deel_lip_dev_env/bin/activate && cd doc && make html && cd -

ipynb-to-rst:
	. deel_lip_dev_env/bin/activate && cd doc && ./generate_doc.sh && cd -
