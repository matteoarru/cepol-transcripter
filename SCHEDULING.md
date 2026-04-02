# Daily Scheduling

This guide explains how to run CEPOL Transcripter automatically every day on
Ubuntu.

The application is safe to schedule regularly because it already skips any media
file that has both sibling output files:

- `file.txt`
- `file.srt`

That means each scheduled run will normally process only new or incomplete
media.

## Recommended approach

On Ubuntu, the best option is a `systemd --user` timer.

Benefits:

- survives terminal closure
- easier log inspection than `cron`
- supports missed-run catch-up with `Persistent=true`
- can be configured to run even when you are not logged in

## Example target folder

This repository has already been used successfully with:

```text
/home/matteo/Insync/matteo@arru.it/OneDrive/WORK/CEPOL/CEPOL Training
```

You can replace that path with any other root folder if needed.

## Step 1: Create the folders and empty files

Create the required directories first:

```bash
mkdir -p ~/bin ~/.config/systemd/user
```

Then create the files you will edit:

```bash
touch ~/bin/cepol-transcripter-daily.sh
touch ~/.config/systemd/user/cepol-transcripter.service
touch ~/.config/systemd/user/cepol-transcripter.timer
```

This gives you a clean starting point before you begin editing.

## Step 2: Create the wrapper script

Create:

[`~/bin/cepol-transcripter-daily.sh`](/home/matteo/bin/cepol-transcripter-daily.sh)

with this content:

```bash
#!/usr/bin/env bash
set -euo pipefail

LOCK_FILE="/home/matteo/.cache/cepol-transcripter.lock"
mkdir -p "$(dirname "$LOCK_FILE")"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another transcription run is already in progress. Exiting."
    exit 0
fi

exec /home/matteo/projects/cepol-transcripter/run.sh \
    "/home/matteo/Insync/matteo@arru.it/OneDrive/WORK/CEPOL/CEPOL Training" \
    --log-level INFO
```

Make it executable:

```bash
mkdir -p ~/bin
chmod +x ~/bin/cepol-transcripter-daily.sh
```

Why this wrapper helps:

- keeps the scheduled command short
- prevents overlapping runs with `flock`
- makes it easy to change the target folder later

## Step 3: Create the `systemd` service

Create:

[`~/.config/systemd/user/cepol-transcripter.service`](/home/matteo/.config/systemd/user/cepol-transcripter.service)

with this content:

```ini
[Unit]
Description=Daily CEPOL transcription batch

[Service]
Type=oneshot
WorkingDirectory=/home/matteo/projects/cepol-transcripter
ExecStart=/home/matteo/bin/cepol-transcripter-daily.sh
Nice=10
```

## Step 4: Create the daily timer

Create:

[`~/.config/systemd/user/cepol-transcripter.timer`](/home/matteo/.config/systemd/user/cepol-transcripter.timer)

with this content:

```ini
[Unit]
Description=Run CEPOL transcripter every day

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true
Unit=cepol-transcripter.service

[Install]
WantedBy=timers.target
```

This example runs every day at `02:00`.

You can change that line if you prefer another time, for example:

- `OnCalendar=*-*-* 01:00:00`
- `OnCalendar=Mon..Fri 21:30:00`

## Step 5: Enable the timer

Run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now cepol-transcripter.timer
systemctl --user list-timers | grep cepol
```

## Step 6: Allow the timer to run without an active login

If you want the timer to run even when you are not logged in graphically, run:

```bash
sudo loginctl enable-linger matteo
```

Without linger, `systemd --user` timers may stop when your user session ends.

## Step 7: Test it manually

Run the service once:

```bash
systemctl --user start cepol-transcripter.service
```

Watch logs:

```bash
journalctl --user -u cepol-transcripter.service -n 100 -f
```

Useful checks:

```bash
systemctl --user status cepol-transcripter.timer
systemctl --user status cepol-transcripter.service
systemctl --user list-timers | grep cepol
```

## What happens on each run

The scheduled process will:

1. scan the configured root folder recursively
2. skip files that already have both `.txt` and `.srt`
3. transcribe only new or unfinished compatible media files
4. save outputs next to the source file
5. remove any generated sibling `*.audio.wav` cache after a successful run

## Recommended timing

Good choices for the schedule:

- after OneDrive/Insync has usually synced the latest files
- during low GPU usage hours
- outside working hours if large webinar batches are expected

If another long run is still active when the timer fires, the lock file prevents
starting a second overlapping batch.

## Cron alternative

If you prefer `cron`, you can use the same wrapper script:

```bash
crontab -e
```

Add:

```cron
0 2 * * * /home/matteo/bin/cepol-transcripter-daily.sh >> /home/matteo/.local/state/cepol-transcripter.log 2>&1
```

`systemd` is still the recommended option on Ubuntu because it gives better log
handling and better recovery behavior.

## Updating the schedule later

If you change the service or timer files, reload `systemd`:

```bash
systemctl --user daemon-reload
systemctl --user restart cepol-transcripter.timer
```

## Troubleshooting

If the timer never runs:

- check `systemctl --user list-timers`
- check `journalctl --user -u cepol-transcripter.timer`
- verify linger is enabled if you expect background runs while logged out

If the service starts but exits immediately:

- check the wrapper path is correct
- confirm [run.sh](/home/matteo/projects/cepol-transcripter/run.sh) is executable
- confirm the virtual environment exists at [`venv/`](/home/matteo/projects/cepol-transcripter/venv)

If nothing seems to happen:

- check that the target root path still exists
- remember that fully processed files are intentionally skipped
- run the command manually once to confirm there is pending media
