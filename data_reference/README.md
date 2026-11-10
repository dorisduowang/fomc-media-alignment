# Data inputs and reproduction

`per_meeting.parquet` and `cosine.parquet` contain derived meeting, lag and source aggregates from the earlier analysis. The checked-in result tables and figures are snapshots from that analysis. The extension's FOMC meeting calendar is at `MakkaPakka update/data_reference/meetings_all88.csv`.

Before rebuilding the baseline tables and figures, stage the inputs below locally:

| Path | Input |
| --- | --- |
| `data_reference/market/SPX_daily_2014_2025.csv` | Frozen S&P 500 daily close file |
| `data_reference/market/VIX_2014_2025.csv` | Frozen daily VIX observations |
| `data_reference/market/mps_fomc.csv` | Monetary policy surprise series used by the baseline |
| `data_reference/sentiment.parquet` | Document-level scores from the scoring stage |
| `MakkaPakka update/data_reference/fomc_minutes_raw.csv` | Frozen meeting minutes from the Federal Reserve |
| `MakkaPakka update/data_reference/market/` | Frozen market inputs for the extension notebook |

Run `make reproduce` from the repository root after supplying the baseline inputs. The extension notebook describes the official GDELT archive download plan and staging layout. It can regenerate the full plan from the public GDELT master file list. Keep each generated plan and its checksum with the corresponding research run.

Source references: [FOMC meeting calendars and minutes](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm), [GDELT data](https://www.gdeltproject.org/data.html), [Guardian Open Platform](https://open-platform.theguardian.com/documentation/), [NYT Article Search](https://developer.nytimes.com/docs/articlesearch-product/1/overview), [FRED VIX](https://fred.stlouisfed.org/series/VIXCLS), and [US Monetary Policy Event-Study Database](https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/).
