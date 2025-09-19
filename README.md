# Alliance–Rebel Deterrence Simulator (Streamlit)

A small app with sliders to explore how alliance *credibility*, *effectiveness*,
*capacity-building*, and *conditionality*, together with a government's
**repression** and **concessions**, shape the rebel attack decision.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser. Adjust sliders, select alliance types,
and click **Run simulation**.

## Files

- `app.py` – the Streamlit UI
- `model.py` – the core model and simulation functions
- `requirements.txt` – Python dependencies
- `README.md` – this file

## Notes

- Charts use **matplotlib** (no seaborn), one chart per figure.
- To export results, use the download buttons after running a simulation.