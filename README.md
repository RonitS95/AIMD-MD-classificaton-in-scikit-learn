# AIMD-MD-classificaton-in-scikit-learn

This repository contains Jupyter notebooks describing several classification
algorithms in scikit-learn based on the Applied Machine Learning in Python 
[course](https://www.coursera.org/learn/python-machine-learning).

## Background

The task is based on our published work (DOI: 10.26434/chemrxiv-2022-754cx-v3) 
on the solvation of thiocyanate anion (SCN $^-$) in  water using classical force field methods (MD) and 
ab-inito molecular dynamics (AIMD).

We showed that the two calculations yield significantly different local structure around the ion.

## The Problem

Given the distances of the nearest two oxygens and two hydrogens from the S, C, and N atoms,
can we classify the particular snapshot into MD or AIMD classes?

## Data available

We have ~200 snapshots each from the MD and AIMD runs and we collect the distances of two nearest oxygens
and two nearest hydrogens from S, C, and N atoms with a total of about 400 instances with 12 features. 

## Environment 

We use scikit-learn in these notebooks. The detailed environment list is as follows, 
- python 3.10.8
- numpy 1.23.4
- scipy 1.9.3
- pandas 1.5.2
- scikit-learn 1.0.1
- matplotlib 3.6.2
- graphviz 2.50.0
