# Flag Intelligence

**Flag recognition first** — a Streamlit computer-vision portfolio project.

**Live demo:** https://flag-intelligence.streamlit.app/

Upload a national flag, photograph it, or choose a demo flag. Flag Intelligence
attempts to identify the country and presents a concise sourced fact sheet:
**country name, capital, area (km²), and latest available annual population**.

## Recognition pipeline (v0.2)

```
Flag image (upload / camera)
       ↓
Image validation
       ↓
Reference-flag comparison ── confident and distinctive? ── yes → country
       │ no
       ↓
MobileNetV3-Small (250 classes: ISO territories + Kosovo)
       ↓
Uncertain? → explicit candidate list and manual confirmation
       ↓
Bundled 250-country reference + World Bank → short country fact sheet
```

- **Reference matching** improves reliability for clean, centered flag pictures;
  it uses normalized image difference and **does not claim a model confidence**.
  Templates are retrieved from FlagCDN and cached; connection failures are
  handled without stopping the application.
- **MobileNetV3-Small** remains the trained fallback for other inputs. Confidence
  is uncalibrated for difficult photos, so uncertain predictions are not forced.
- **Manual correction** is always available when flags look alike or a photo
  does not allow reliable identification.
- **Recent data**: geography and capital from the bundled mledoze/countries dataset; the newest
  non-null **annual** population observation from the World Bank
  (`SP.POP.TOTL`), including year and source. A REST Countries fallback is
  clearly marked when no World Bank observation can be fetched.
- **Quick JSON export** preserves year, source, and retrieval timestamp.
- **Advanced country reports and conversations** remain in the existing
  codebase but are intentionally removed from the main user flow.

## What the accuracy results mean

On the human-reviewed **real-world challenge** documented for the original
classifier, **Top-1 accuracy was 19.61% and Top-5 was 27.45%**. The
new reference-matching layer has not been evaluated on a new independent
photo benchmark. It would be misleading to treat clean official-flag tests as
proof of robust recognition in photos with folds, occlusion, and backgrounds.

Similar designs include **Côte d'Ivoire / Ireland**, **Chad / Romania**, and
**Monaco / Indonesia**. Users should confirm uncertain predictions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To launch the previous report-rich experience instead:

```bash
FLAG_INTELLIGENCE_MODE=advanced streamlit run app.py
```

The Streamlit Community Cloud deployment may use `app.py`,
`streamlit_app.py`, or `main.py`; all three reach the recognition-first app.

## Tests and source provenance

```bash
python -m pytest -q tests/test_flag_first.py
```

GitHub Actions includes a dedicated **Flag-first Streamlit smoke test** and
tests for visually distinct/ambiguous tricolors, population provenance, and
fallback behavior. Additional older workflows validate the advanced report
features independently.

**Geographic reference:** [mledoze/countries](https://github.com/mledoze/countries) (ODbL 1.0).\n**Data services:**
[World Bank indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api),
and [FlagCDN](https://flagcdn.com/).

Never interpret the last annual population observation as a real-time live
headcount. The application makes source years visible for this reason.
