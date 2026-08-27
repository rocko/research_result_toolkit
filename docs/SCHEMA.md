# Common result schema

The toolkit expects one row per experiment run and metric.

Required columns:

- `experiment` - experiment or method identifier
- `run` - repeated-run identifier
- `metric` - metric name
- `value` - numeric metric value

Example:

```csv
experiment,run,metric,value
baseline,1,accuracy,0.812
baseline,2,accuracy,0.824
method_a,1,accuracy,0.841
method_a,2,accuracy,0.836
```

Projects remain free to store or report their native outputs however they want. To use this toolkit, they only need to export the measurements they want analyzed into this common representation.

Additional columns are allowed and preserved by the loader.
