# Add-ons examples

This package contains 4 examples of add-ons for Weegit:

- `spikes` – searches for spikes on selected channels and draws markers in the traces.
- `digital_events` – detects digital events and adds them to `UserSession.events`.
- `analog_events` – detects events on an analog channel and adds them to `UserSession.events`.
- `csd` – visualizes a CSD background beneath the signals.

All add-ons are launched via the `Analysis > Run` side panel in Weegit.

---

## Spikes (`spikes.py`)

### What it does

- Processes **only the current sweep**.
- Searches for negative peaks (spikes) on selected channels.
- Saves the result in `add_ons/data/<module_name>/<SWEEP_IDX>.spikes`.
- In `View` mode, draws red markers on the detected spikes.

### Form fields and their effect

- `Channel group` – which group to take available channels from.
- `Channels` – on which channels to perform the search.
- `Threshold` – multiplier for the threshold using MAD estimation:
  - threshold is calculated as `threshold * MAD / 0.6745` (if `MAD > 0`);
  - larger value → lower sensitivity, fewer detected spikes.
- `Use filter` – enable pre‑filtering before the search.
- `Filter type` – type of filter.
- `Filter params` – parameters of the selected filter:
  - define the band / suppression characteristics;
  - affect the signal shape and therefore the number/position of detected peaks.

---

## Digital events (`digital_events.py`)

### What it does

- Processes **all sweeps**.
- For each selected channel, finds peaks and interprets them as events.
- Adds the detected events to `UserSession.events` with a new event name.

### Form fields and their effect

- `Channel group` – channel group for selection.
- `Digital channels` – channels on which to perform the search.
- `Event name` – name of the new entry in the events dictionary (must be unique).
- `Height (threshold)` – threshold for `find_peaks`:
  - positive → searches for positive peaks;
  - negative → the signal is inverted, looks for “dips” below the threshold.
- `Min distance, ms` – minimum distance between adjacent events:
  - larger value → dense detections are merged.
- `Use filter` / `Filter type` / `Filter params` – optional pre‑filtering of each channel.

---

## Analog events (`analog_events.py`)

### What it does

- Processes **all sweeps**.
- Searches for events on **one selected analog channel**.
- After detection, adds the events to `UserSession.events` with a new name.

### Form fields and their effect

- `Channel group` – group from which the analog channel is chosen.
- `Analog channel` – channel to search on.
- `Event name` – name of the new entry in the events dictionary (must be unique).
- `Stimulation time, ms` – time offset added to each detected event time.
- `Height (threshold)` – peak detection threshold:
  - if negative, the signal is inverted (search for negative deflections).
- `Use filter` / `Filter type` / `Filter params` – optional filtering before the search.

Note: after filtering, an impulse may produce paired peaks; the add‑on keeps the first peak of each pair.

---

## CSD (`csd.py`)

### What it does

- In `View`, computes CSD over the visible channels and draws a color map beneath the traces.
- In `Run`, does not perform lengthy computations but opens a form to adjust the scale and render density.
- Settings are saved in `add_ons/data/<module_name>/csd_config.json`.

### Form fields and their effect

- `Scale min` / `Scale max`:
  - set a manual range for the color scale;
  - with a manual scale, colors remain stable across redraws.
- `X pixel step`:
  - time step during rendering;
  - larger value → faster drawing, but lower X‑detail.
- `Y pixel step`:
  - channel/height step during rendering;
  - larger value → faster drawing, but lower Y‑detail.
- `Reset to auto`:
  - resets manual `min/max`;
  - the scale is again taken automatically from the current CSD.

---
