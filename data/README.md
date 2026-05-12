# Data

Sample files are derived from the public Wyscout event dataset in:

https://github.com/koenvo/wyscout-soccer-match-event-dataset

Run this command to refresh the sample files:

```bash
python scripts/fetch_sample_data.py
```

Generated files:

- `data/raw/wyscout_2499754_liverpool_manchester_city.json`: original nested Wyscout JSON for one match.
- `data/sample/wyscout_2499754_events.csv`: flattened event table for the app.
- `data/sample/wyscout_2499754_events.json`: flattened JSON records for the app.
- `data/sample/wyscout_multi_match_events.csv`: flattened multi-match event table.
- `data/sample/wyscout_multi_match_events.json`: flattened multi-match JSON records.
- `data/sample/matches/*.csv`: individual match files used by the multi-match page.
- `data/sample/wyscout_2499754_notes.md`: optional unstructured context.

The upstream dataset is released under CC BY 4.0. Keep attribution in downstream demos.
