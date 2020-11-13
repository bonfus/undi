
%.ipynb: %.py
	py2nb $<
	jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=600 $@

all: Cu.ipynb FmuF.ipynb LaCuO4.ipynb Al.ipynb

