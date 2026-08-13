# Sebastian Voice Assistant

Sebastian is a local-first voice assistant for Windows 10. It uses separately
configured microphone and speaker devices, listens for a custom wake word, stops
recording when speech ends, and runs speech recognition, conversation, and speech
synthesis locally.

Current features:

- Separate microphone and speaker selection, independent of Windows defaults
- Custom openWakeWord wake-word model
- Automatic end-of-speech detection
- Faster Whisper speech-to-text
- Ollama with `qwen3:8b` for conversation and function calling
- Piper text-to-speech with `en_US-ryan-medium`
- Local Windows time, date, and day-of-week tools
- Location-aware time and weather for home or arbitrary spoken places
- Current conditions plus seven-day forecasts from Open-Meteo without an API key
- Gmail reading, Google Calendar management, and automatic pre-event reminders
- Idle GPU model release for long-running wake-word operation
- Inactivity-based sessions with bounded rolling context

## 1. Requirements

- Windows 10
- Python 3.12 recommended
- [Ollama for Windows](https://ollama.com/download/windows)
- A microphone and an audio output device
- An NVIDIA GPU for the default Faster Whisper settings, or use the CPU settings
  in [CPU Speech Recognition](#cpu-speech-recognition)
- Internet access for initial package/model downloads and live weather

All normal assistant processing is local. Weather and explicit place-name lookup
call Open-Meteo; home time/date uses the configured timezone without a network call.

## 2. Install

Open PowerShell in the repository root and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
ollama pull qwen3:8b
```

Using `.\.venv\Scripts\python.exe` directly avoids PowerShell activation-policy
problems. Start Ollama if it is not already running, then check the model:

```powershell
.\.venv\Scripts\python.exe .\tools\test_ollama.py
```

## 3. Private Local Configuration

Committed files under `config/` contain shareable defaults and placeholders.
Machine-specific values belong in these ignored files:

- `config/devices.local.json`: microphone and speaker
- `config/assistant.local.json`: location and optional model overrides

Local files are recursively merged over their matching base files. This keeps
personal settings out of Git while allowing the shared defaults to evolve.

## 4. Select Microphone and Speaker

List Windows audio devices:

```powershell
.\.venv\Scripts\python.exe .\tools\list_audio_devices.py
```

Create `config/devices.local.json` with short, unique pieces of the device names:

```json
{
  "microphone": {
    "name_query": "YOUR MICROPHONE NAME",
    "device_id": null,
    "sample_rate": 48000,
    "channels": 1
  },
  "speaker": {
    "name_query": "YOUR SPEAKER NAME",
    "device_id": null,
    "sample_rate": 48000,
    "channels": 2
  }
}
```

`name_query` is preferred because Windows device IDs can change after a reboot or
USB reconnect. Use a numeric `device_id` only if a name matches multiple devices.
The two endpoints are independent; changing the microphone never changes the
speaker.

Test both devices:

```powershell
.\.venv\Scripts\python.exe .\tools\test_microphone.py --seconds 5
.\.venv\Scripts\python.exe .\tools\test_speaker.py
```

Listen to `outputs/mic_test.wav`. To test a device without editing config:

```powershell
.\.venv\Scripts\python.exe .\tools\test_microphone.py --mic "YOUR MICROPHONE" --seconds 5
.\.venv\Scripts\python.exe .\tools\test_speaker.py --speaker "YOUR SPEAKER"
```

Replace the placeholders with unique parts of the device names reported by
`list_audio_devices.py`. You can change either device later without changing the
other one.

## 5. Speech Recognition

The first test downloads the configured Faster Whisper model and caches it outside
the repository:

```powershell
.\.venv\Scripts\python.exe .\tools\test_speech_to_text.py --seconds 5
```

To transcribe an existing WAV:

```powershell
.\.venv\Scripts\python.exe .\tools\test_speech_to_text.py --audio .\outputs\mic_test.wav
```

### CPU Speech Recognition

The shared config uses CUDA. On a PC without a compatible NVIDIA setup, add this
override to `config/assistant.local.json`:

```json
{
  "speech_to_text": {
    "device": "cpu",
    "compute_type": "int8"
  }
}
```

You can also test CPU mode without changing config:

```powershell
.\.venv\Scripts\python.exe .\tools\test_speech_to_text.py --device cpu --seconds 5
```

## 6. Install and Test a Piper Voice

Download the configured Ryan medium voice:

```powershell
.\.venv\Scripts\python.exe -m piper.download_voices en_US-ryan-medium --download-dir .\data\voices
.\.venv\Scripts\python.exe .\tools\test_text_to_speech.py
```

Piper downloads an `.onnx` model and its JSON metadata. To use another Piper
voice, download it and override `text_to_speech.voice` and
`text_to_speech.model_path` in `config/assistant.local.json`.

## 7. Install or Create a Wake Word

The configured wake phrase is **Sebastian**, with its model expected at:

```text
data/wake_words/sebastian.onnx
```

You can use another wake word or train a custom one through the official
openWakeWord training workflow. The configured phrase and model path must match
the wake-word model you choose.

### Use an Existing Model

Download an openWakeWord-compatible ONNX model from a source you trust, check its
license, and put it in `data/wake_words/`. Either rename it to `sebastian.onnx` or
set its relative path in `config/assistant.local.json`:

```json
{
  "wake_word": {
    "phrase": "Sebastian",
    "model_path": "data/wake_words/your-model.onnx"
  }
}
```

The displayed phrase does not change what a trained model recognizes. The model
itself must have been trained for the desired trigger phrase.

### Train a Custom Model

Follow [docs/train-sebastian.md](docs/train-sebastian.md). It links to the
[official openWakeWord training Colab](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing)
and includes the current Colab `webrtcvad` fix. Change the model name and target
phrase in the notebook if you choose a different trigger.

Prepare openWakeWord's shared runtime files and test detection:

```powershell
.\.venv\Scripts\python.exe .\tools\setup_wake_word.py
.\.venv\Scripts\python.exe .\tools\test_wake_word.py --timeout 30
```

If needed, adjust `wake_word.threshold` in `config/assistant.local.json`. Lower
values detect more easily but increase false activations; higher values are more
selective.

## 8. Configure Weather

Create or extend `config/assistant.local.json` with your home coordinates and IANA
timezone:

```json
{
  "home_location": {
    "latitude": null,
    "longitude": null,
    "name": "City, State",
    "timezone": "America/New_York"
  }
}
```

Replace both `null` coordinate placeholders with valid decimal values. Open-Meteo's
[geocoding search](https://open-meteo.com/en/docs/geocoding-api) can find them by
city or postal code. Fahrenheit, mph, and inches are the shared defaults. No API
key or IP geolocation is used. Your configured home stays in the ignored local file;
asking about another city never changes it.

When a request names a place, Sebastian sends the place name to Open-Meteo's
geocoder, receives coordinates and an IANA timezone, and uses those resolved values
for that turn. The 128 most recent successful lookups are cached in memory. Clear winners such as Tokyo,
London, Raleigh, and Miami are selected automatically; similarly plausible matches
such as Springfield cause a short clarification question.

Examples:

```text
"What's the weather?"                       -> configured home coordinates
"What's the weather in Tokyo?"              -> geocode Tokyo, then fetch weather
"What time is it in London?"                -> geocode London, then use Europe/London
"Give me the time and weather in Tokyo."    -> both tools use the same Tokyo result
"What's the forecast in Paris tomorrow?"    -> tomorrow's Paris forecast
```

Timezone conversion uses Python `zoneinfo` and the `tzdata` package. Sebastian uses
the Windows clock as the current instant and applies the resolved IANA timezone; it
does not calculate offsets manually.

Test real tool selection and responses:

```powershell
.\.venv\Scripts\python.exe .\tools\test_assistant_tools.py
.\.venv\Scripts\python.exe .\tools\test_assistant_tools.py --simulate-weather-failure
```

Use `--debug` to see selected tools and parsed results. Coordinates are not
written to logs.

## 9. Connect Gmail and Google Calendar

Sebastian uses Google's official installed-app OAuth flow and keeps Gmail and
Calendar authorization in separate least-privilege token files. Gmail is read-only;
Calendar can read, create, update, and delete events. Email sending, archiving, and
deletion are not implemented.

### Create the Google Cloud application

1. Open the [Google Cloud Console](https://console.cloud.google.com/), sign in with
   the Google account you want Sebastian to use, and create or select a project such
   as `Sebastian Assistant`.
2. Open **APIs & Services > Library**. Search for and enable **Gmail API**, then
   search for and enable **Google Calendar API**.
3. Open **Google Auth platform > Branding**. If prompted, click **Get Started**.
   Enter `Sebastian` as the app name, choose your email for user support and contact,
   accept the user-data policy, and finish the setup.
4. Open **Google Auth platform > Audience**. Use **Internal** only for an account in
   your own Google Workspace organization. For a normal personal Gmail account,
   choose **External** and leave the publishing status as **Testing**.
5. On the same **Audience** page, find **Test users**, click **Add users**, enter the
   exact Google account Sebastian will access, and save it. For example:

```text
YOUR_GOOGLE_ACCOUNT_EMAIL
```

The account used in the browser must exactly match an entry under **Test users**.
Otherwise, Google displays `Access blocked`, says Sebastian has not completed Google
verification, and ends authorization with `Error 403: access_denied`.

6. Open **Google Auth platform > Data Access** and add these scopes:

```text
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/calendar.events
```

`gmail.readonly` is required because Sebastian can fetch a selected message body for
summarization. It cannot modify or send mail. `calendar.events` is narrower than the
full Calendar scope and permits event reads and writes without calendar-management
access. A public app using Gmail read access has additional Google verification
requirements; keep this personal project in Testing unless you intend to distribute it.

For normal inbox questions, Sebastian ranks at most 12 recent Gmail candidates and
keeps at most five lightweight references for session follow-ups. He speaks no more
than two emails by default, using only this controlled format:

```text
The sender is Professor Smith. The snippet is: The project deadline is Friday.
```

Email addresses, subjects, received dates, IDs, labels, asterisks, URLs, emoji,
non-English characters, and raw metadata are removed from list speech. If a sender
has no readable display name, Sebastian says `an unnamed sender`. Snippets are
limited to 140 characters. This response is enforced after the LLM returns, so the
model cannot substitute a verbose metadata dump. A selected follow-up such as
"What did the first one say?" fetches only that message and asks the LLM for a one-
or two-sentence summary instead of reading it verbatim.

The limits are configurable under `gmail` in `config/assistant.json`:

```json
{
  "gmail": {
    "max_results": 5,
    "important_candidate_limit": 12,
    "spoken_result_limit": 2,
    "list_snippet_character_limit": 140
  }
}
```

7. Open **Google Auth platform > Clients**, click **Create Client**, choose
   **Desktop app**, name it `Sebastian Windows`, and click **Create**.
8. Download the client JSON. Create the private destination directory in PowerShell:

```powershell
cd "$HOME\Desktop\projects\voice-assistant"
New-Item -ItemType Directory -Path .\secrets\google -Force
```

9. Move the downloaded file into that directory, rename it to
   `client_secret.json`, and confirm the final path is exactly:

```text
%USERPROFILE%\Desktop\projects\voice-assistant\secrets\google\client_secret.json
```

If Windows hides filename extensions, verify that the file was not accidentally
named `client_secret.json.json`:

```powershell
Test-Path .\secrets\google\client_secret.json
```

The command must print `True`.

10. Authorize both services:

```powershell
cd "$HOME\Desktop\projects\voice-assistant"
.\.venv\Scripts\python.exe .\tools\authenticate_google.py --service all
```

Sebastian opens the default browser twice: once for Gmail and once for Calendar.
Select the intended account, review each requested permission, and click **Continue**
or **Allow**. With an External app in Testing, Google may show an unverified/test-app
warning. Use the same account entered under **Test users**.

11. Confirm that both local token files were created without printing their contents:

```powershell
Test-Path .\secrets\google\gmail_token.json
Test-Path .\secrets\google\calendar_token.json
```

Both commands must print `True`. The tokens are stored at:

```text
secrets/google/gmail_token.json
secrets/google/calendar_token.json
```

The client secret, both tokens, and the entire `secrets/` directory are ignored by
Git and must never be committed. Tokens refresh automatically when Google provides a
refresh token. Google currently limits External apps in Testing and their test-user
authorizations may expire after seven days; rerun the authorization command if Google
requires consent again.

### Fix `Error 403: access_denied`

If Google shows **Access blocked: "Sebastian" has not completed Google
verification**, do not publish the app and do not submit it for verification. For a
private installation in Testing, fix the test-user configuration instead:

1. Open [Google Auth Platform > Audience](https://console.cloud.google.com/auth/audience).
2. Confirm the project selector shows the project containing the `Sebastian Windows`
   Desktop OAuth client.
3. Confirm **User type** is **External** and **Publishing status** is **Testing**.
4. Under **Test users**, click **Add users**, enter the same Google account you will
   use during authorization, and save.
5. Wait about one minute, then rerun:

```powershell
cd "$HOME\Desktop\projects\voice-assistant"
.\.venv\Scripts\python.exe .\tools\authenticate_google.py --service all
```

If authorization previously failed before consent completed, no token cleanup is
normally required. Check the two token paths above; if they do not exist, simply run
the authorization command again after fixing the test user.

### Calendar behavior

Sebastian uses `home_location.timezone` from `config/assistant.local.json` for
relative expressions such as `today`, `tomorrow`, `Friday`, and `in two minutes`.
Parsing is timezone-aware and backed by `dateparser` plus Python `zoneinfo`. A start
time without an end uses `calendar.default_event_duration_minutes`, which defaults to
60. Override Calendar policy in the ignored local config if needed:

```json
{
  "calendar": {
    "calendar_id": "primary",
    "default_event_duration_minutes": 60,
    "confirm_updates": true,
    "confirm_deletes": true
  }
}
```

Explicit, unambiguous event creation runs immediately. Updates and deletions first
resolve one event and ask for confirmation; the following yes/no response performs or
rejects the pending action. Multiple matches cause a clarification instead. Pending
actions and lightweight event/email IDs expire with session memory. Full Google data
is not stored in conversation history.

Sebastian only speaks a Calendar success response after the corresponding Google tool
returns success. A deletion uses two logged tool calls across two voice turns:
`delete_event`, followed by `confirm_calendar_action`. If either call is missing or
Google fails, Sebastian must not say the event was deleted. A temporary Google failure
keeps the pending action available so another `Yes` can retry it.

Try these after authorization:

```text
"What's my plan today?"
"What do I have tomorrow?"
"Schedule test meeting tomorrow at 3 PM."
"Move the test meeting to 4 PM."            then answer "Yes."
"Cancel the test meeting."                  then answer "Yes."
"Do I have any important emails?"
"What did the first one say?"
```

Verify test events in Google Calendar and delete them when finished. Use a known
sender for Gmail tests so unrelated message bodies are never requested.

### Persistent reminders

All reminders are stored in the ignored SQLite database
`data/reminders.sqlite3`, outside conversation memory. While
`tools/run_wake_assistant.py` is running, Sebastian polls this local database every
five seconds. A due reminder interrupts idle listening, Piper speaks it, and the
record is marked fired. Failed playback returns it to pending; an interrupted
`firing` record is recovered after restart.

Sebastian also synchronizes every timed event in the configured Google Calendar,
which is `primary` by default. Each event gets one persistent local reminder 30
minutes before its start. The background worker checks Google every five minutes and
keeps a rolling seven-day window. It updates the reminder when an event moves,
cancels it when the event is deleted or changed to all-day, and does not duplicate
unchanged events. Calendar/network failures are logged and leave existing reminders
intact. All-day events are intentionally skipped because they have no specific start
time to subtract 30 minutes from.

These defaults can be overridden in the ignored `config/assistant.local.json`:

```json
{
  "reminders": {
    "calendar_sync_enabled": true,
    "calendar_reminder_minutes_before": 30,
    "calendar_sync_interval_seconds": 300,
    "calendar_sync_lookahead_hours": 168,
    "calendar_sync_max_results": 2500
  }
}
```

Run a one-shot synchronization diagnostic after Calendar authorization. It reports
counts only and does not print event titles:

```powershell
.\.venv\Scripts\python.exe .\tools\test_calendar_reminder_sync.py --debug
```

For an event named `Project meeting` at 3 PM, the spoken alert is concise:

```text
Reminder: You have Project meeting at 3 PM.
```

Test the end-to-end path while the wake assistant remains running:

```text
"Remind me in two minutes to test Sebastian's reminder system."
```

Or run the persistence/synthesis diagnostic (`--play` uses the configured speaker):

```powershell
.\.venv\Scripts\python.exe .\tools\test_reminder_system.py --delay-seconds 5 --play
```

The push-to-talk scripts are not persistent background schedulers and do not run
Calendar synchronization. Keep the wake assistant running for reminders to stay
synchronized and fire on time.

## 10. GPU and Session Configuration

The shared defaults in `config/assistant.json` are:

```json
{
  "llm": {
    "context_window": 4096,
    "keep_alive": "5m"
  },
  "conversation": {
    "session_ttl_minutes": 30,
    "max_context_tokens": 2400
  },
  "resources": {
    "stt_idle_unload_minutes": 15,
    "maintenance_interval_seconds": 5
  }
}
```

`llm.keep_alive` is sent with every Ollama chat request. Five minutes keeps normal
back-and-forth responsive, then allows Ollama to unload Qwen from VRAM. Set a shorter
duration in `config/assistant.local.json` for a future gaming-oriented profile.

Faster Whisper is lazy-loaded on the first recorded request. During continuous wake
listening, Sebastian checks idle resources every five seconds and uses CTranslate2's
supported model unload API after 15 minutes without STT activity. The next request
reloads it automatically. Piper and openWakeWord run on CPU and remain available.

Conversation memory expires only after 30 minutes without a meaningful interaction.
Each interaction resets that timer. History is trimmed by complete oldest
user/assistant turns when its approximate 2,400-token budget is exceeded. The system
prompt, home location, model settings, audio devices, and integrations are persistent
configuration and are never part of the expiring session.

The token count is a conservative approximation because the local Qwen tokenizer is
not loaded separately. The 2,400-token history budget leaves room inside Qwen's 4,096
token context for the system prompt, tool schemas, current request, tool results, and
response.

In push-to-talk conversation mode, enter `clear` to call `reset_session()`. Code can
also call `VoiceAssistant.reset_session()` directly. Spoken reset routing is not yet
enabled.

Verify model lifetimes with a short temporary Ollama timeout:

```powershell
.\.venv\Scripts\python.exe .\tools\test_resource_lifetimes.py --keep-alive 5s --wait-seconds 8
.\.venv\Scripts\python.exe .\tools\test_resource_lifetimes.py --keep-alive 5s --wait-seconds 8 --audio .\outputs\mic_test.wav
```

The temporary `5s` value is used only by the diagnostic request; normal Sebastian
requests continue using the configured `5m` value.

## 11. Run Sebastian

Test a push-to-talk conversation before enabling the wake word:

```powershell
.\.venv\Scripts\python.exe .\tools\run_conversation.py
```

Press Enter to speak, enter `clear` to erase short-term context, or enter `q` to
quit. Then run the wake-activated assistant:

```powershell
.\.venv\Scripts\python.exe .\tools\run_wake_assistant.py
```

Say the configured wake phrase, wait for the tone, and speak. Recording stops
after about 0.9 seconds of silence and has a 12-second maximum. Use `Ctrl+C` to
stop. Useful variants:

```powershell
.\.venv\Scripts\python.exe .\tools\run_wake_assistant.py --once
.\.venv\Scripts\python.exe .\tools\run_wake_assistant.py --debug
```

Try: "What time is it?", "What's today's date?", "What's it like outside?",
"What's my plan today?", "Any important emails?", or "Remind me in two minutes."

## 12. Tests

Run deterministic tests without a microphone, Ollama, or internet connection:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe .\tools\test_session_behavior.py
.\.venv\Scripts\python.exe .\tools\test_google_failure_routing.py
```

The deterministic suite covers configuration merging, time/date data, weather
parsing and failures, Gmail metadata/body boundaries, Calendar payloads and
confirmation rules, persistent reminder recovery, LLM tool loops, automatic
end-of-speech detection, 12-turn retention, context trimming, session TTL/reset,
and STT load/unload/reload. Live Gmail and Calendar tests require your OAuth setup.
The Google failure diagnostic refuses to run after either real token file exists.

To watch VRAM manually, open a second PowerShell window while Sebastian runs:

```powershell
nvidia-smi -l 2
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" ps
```

After a response, `ollama ps` should show Qwen and its expiration time. With no new
requests it should disappear after approximately five minutes. The first request
afterward reloads Qwen and logs its model load duration. Windows WDDM may report GPU
memory as unavailable per process, so compare the total `Memory-Usage` before, while
active, and after expiration.

## Troubleshooting

**Audio device not found:** rerun `list_audio_devices.py`, shorten or correct the
local `name_query`, and verify Windows microphone privacy permissions.

**Ollama connection failed:** open Ollama, run `ollama list`, and confirm
`qwen3:8b` exists. The default endpoint is `http://127.0.0.1:11434`. Sebastian
automatically retries a transient local connection reset twice before failing the
turn. If Ollama repeatedly disappears during a tool response, restart Ollama and
check `nvidia-smi` for VRAM pressure.

**CUDA or cuDNN error:** update the NVIDIA driver or use the CPU override above.

**First response is slower after idle:** this is expected after Ollama or Faster
Whisper unloads. Increase `keep_alive` or `stt_idle_unload_minutes` if the latency is
more important than idle VRAM.

**Piper model missing:** rerun the Piper download command and confirm both the
`.onnx` and `.onnx.json` files exist under `data/voices/`.

**Wake-word startup failed:** confirm the configured ONNX path exists, then rerun
`tools/setup_wake_word.py`.

**Weather unavailable:** verify the two coordinates, internet access, and
timezone. Time and date still work when weather fails.

**Google credentials missing:** place the downloaded Desktop OAuth JSON at
`secrets/google/client_secret.json`, then run `tools/authenticate_google.py`.

**Google authorization expired or revoked:** rerun the authorization command. Delete
only the affected ignored token file first if Google reports a scope/token mismatch.
Weather, time, and local reminders continue working when Google is unavailable.

## Repository and Privacy

`.gitignore` excludes:

- `.venv`, Python caches, build output, coverage, and editor files
- Downloaded/trained model files (`.onnx`, `.tflite`, `.gguf`, `.safetensors`,
  `.pt`, and `.pth`)
- Generated WAV files, logs, and temporary work
- `config/*.local.json`, local `.env` files, tokens, private keys, credentials,
  and secret directories
- Local SQLite databases and their journal/WAL files under `data/`

Do not use `git add -f` for local config, credentials, recordings, or model files.
Before committing, inspect both normal and ignored files:

```powershell
git status --short
git status --short --ignored
```

Model files are intentionally downloaded separately to keep the repository small
and avoid redistributing files with unknown licenses.

## Project Layout

```text
config/                  Shared defaults plus ignored local overrides
data/voices/             Downloaded Piper voices (ignored)
data/wake_words/         Downloaded or trained wake models (ignored)
data/reminders.sqlite3   Persistent local reminders (ignored)
docs/                    Training notes
outputs/                 Generated recordings and speech (ignored)
tests/                   Deterministic automated tests
tools/                   Setup, diagnostics, and runnable entry points
voice_assistant/audio/   Audio devices, STT, TTS, wake word, end-of-speech
voice_assistant/assistant/ Conversation and tool-calling loop
voice_assistant/integrations/ External services such as Open-Meteo
voice_assistant/tools/   Local assistant tools such as time and date
```
