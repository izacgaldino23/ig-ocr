VENV = .venv
PYTHON = ${VENV}\Scripts\python
ARGS = $(filter-out $@,$(MAKECMDGOALS))

train:
	${PYTHON} train.py ${ARGS}

test:
	${PYTHON} test.py ${ARGS}

visualize:
	${PYTHON} visualize.py ${ARGS}

%:
	@: