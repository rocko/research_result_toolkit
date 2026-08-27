# Design decisions

## Common schema instead of universal importers

The toolkit does not try to understand every framework, spreadsheet, or custom experiment logger. Projects export the measurements they want analyzed into the common result schema.

This keeps the analysis layer small and makes comparisons independent of PyTorch, TensorFlow, or other training code.

## Decision support instead of automatic statistical authority

`suggest_test()` proposes candidate tests from explicit structural information such as paired vs. independent samples and whether a normality assumption is being made.

It intentionally does not infer whether an assumption is scientifically justified and does not replace statistical review.

## Small set of transparent tests

The prototype includes a few common paired and independent-sample comparisons rather than attempting to wrap the full SciPy statistics API.

The goal is consistent, inspectable research-group tooling rather than feature completeness.
