Training datasets are created from pymimir and exported to respective domain JSON files `datasets/jsons/DOMAIN` . 
This preprocessing action is separate and integrated into `run.py` workflow, but takes quite some time.
The dataset folder also contains `datasets/lrnn/*` subdirs that get automatically extracted out of 
these json files as part of the main workflow, depending on the requested format settings, particularly:
 - `--state_regression` for ADDING the current state distance target as a regression label
 - `--action_regression` for switching between classification/regression of the action targets

The resulting lrnn training samples will get exported into
 - `--limit` state samples from the JSON file

split into `*/examples.txt` and `*/queries.txt`, linked together via example ids (rows), in a human-readable format 
so that you can check them even manually.
