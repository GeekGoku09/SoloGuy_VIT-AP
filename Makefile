.PHONY: help setup data-sample inspect features train eval whatif app docs check-submission demo test

help:
	@python -m dtwin --help

setup:
	python -m dtwin setup

data-sample:
	python -m dtwin data-sample

inspect:
	python -m dtwin inspect

features:
	python -m dtwin features

train:
	python -m dtwin train

eval:
	python -m dtwin eval

whatif:
	python -m dtwin whatif

app:
	python -m dtwin app-cmd

docs:
	python -m dtwin docs

check-submission:
	python -m dtwin check-submission

demo:
	python -m dtwin demo

test:
	python -m dtwin test
