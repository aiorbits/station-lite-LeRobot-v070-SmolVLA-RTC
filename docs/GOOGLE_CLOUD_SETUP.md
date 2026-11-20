# Google Cloud setup: from zero to a robot-ready SmolVLA server

How I set up Google Cloud (17–19 Nov 2026) so the Station Lite robot can use the teacher's SmolVLA model on a cloud GPU. This file covers the cloud side only. Day-to-day use is in `docs/MODAL_RUNBOOK_AND_TROUBLESHOOTING.md` (Option B: Google Cloud).

> **VM specs (what we use)**
>
> | | |
> |---|---|
> | **GPU** | **1x NVIDIA L4, 32 GB GPU memory** |
> | **Machine type** | **`g2-standard-8`: 8 vCPUs, 32 GB RAM** |
> | **Boot disk** | **100 GB** |
> | **Zone** | `asia-southeast1-b` (Singapore) |
> | **Software** | Ubuntu, conda env `smolvla`: Python 3.12, LeRobot 0.7.0, torch 2.15.0 |
> | **Speed** | about 360 ms per prediction (warm) |
>
> We started on `g2-standard-4` (4 vCPUs, **16 GB RAM**). 16 GB RAM is too little for training, because loading the demo videos and images fills it. `g2-standard-8` has the same L4 but double the RAM and CPUs. The 24 GB of GPU memory is enough for both serving and fine-tuning SmolVLA. Changing the machine type is in section 9.

**Result:**
- VM `vla-instance` (1x NVIDIA L4, `g2-standard-8`) in `asia-southeast1-b`
- server on port 8011
- fixed IP `<VM_IP>`
- only the robot's network may reach it
- the server starts by itself when the VM boots

**Why the cloud:** my laptop has no NVIDIA GPU, and the 4080 PC is the teacher's personal one. The robot's network has internet (dblink's SIM), so the robot can reach a cloud server directly.

> Steps 1–2 (account, project, VM creation, Cloud NAT) were done before the chat notes started. They're written as the usual console steps, with what we ended up with.

---

## 1. Account, project, billing
1. Sign in at <https://console.cloud.google.com> with the work Google account.
2. Create a project → name `vla-project` (its ID became `<PROJECT_ID>`).
3. Billing: link a billing account. New accounts usually get free-trial credits, and charges come out of those first.
4. Budget alert: Billing → **Budgets & alerts** → Create budget (e.g. 50, with alerts at 50 / 90 / 100%).
5. If creating a GPU VM says "quota exceeded": IAM & Admin → **Quotas** → search the GPU (e.g. "NVIDIA L4 GPUs") for the region → request an increase. This can take hours to days.

## 2. The VM and outbound internet
1. Compute Engine → VM instances → **Create instance**:

   | Setting | Value |
   |---|---|
   | Name | `vla-instance` |
   | Region / zone | `asia-southeast1` / `asia-southeast1-b` (Singapore, close to the robot) |
   | Machine | GPUs → **NVIDIA L4** x1 → machine type **`g2-standard-8`** (8 vCPUs, 32 GB RAM) |
   | Boot disk | Ubuntu (or a Deep Learning VM image), **at least 100 GB** |

2. The organisation blocks external IPs on VMs by default, so the VM first had **no external IP**.
3. To give it outgoing internet (pip, git, Hugging Face): VPC network → **Cloud NAT** → create a gateway for the VM's region/network.

## 3. Install the software (VM → SSH button → browser terminal)
1. Check the GPU:
   ```bash
   nvidia-smi
   ```
   If it says `command not found`, install the driver, then reconnect:
   ```bash
   sudo apt update && sudo apt install -y ubuntu-drivers-common && sudo ubuntu-drivers install && sudo reboot
   ```
2. Tools and the teacher's code:
   ```bash
   sudo apt update && sudo apt install -y git tmux unzip wget curl
   ```
   ```bash
   git clone https://github.com/aiorbits/station-lite-LeRobot-v070-SmolVLA-RTC.git ~/smolvla
   ```
3. Conda (Miniforge):
   ```bash
   wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash Miniforge3-Linux-x86_64.sh -b && ~/miniforge3/bin/conda init bash && exec bash
   ```
4. Environment with the teacher's LeRobot version:
   ```bash
   conda create -n smolvla python=3.12 ffmpeg -c conda-forge -y
   ```
   ```bash
   conda activate smolvla && pip install "lerobot[smolvla]==0.7.0" scipy
   ```
   `scipy` isn't installed by LeRobot, and one of the teacher's tests needs it.
5. Check:
   ```bash
   python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), lerobot.__version__)"
   ```
   → `... True NVIDIA L4 0.7.0`. We got the same versions as the teacher: torch 2.15.0, transformers 5.22.2.
6. The vision-language part, at the exact version the teacher pinned:
   ```bash
   export HF_HOME=~/smolvla/.hf_cache && hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467
   ```
   ```bash
   D=$HF_HOME/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct && mkdir -p $D/refs && printf 7b375e1b73b11138ff12fe22c8f2822d8fe03467 > $D/refs/main
   ```
   The second line matters: offline, the loader asks for `main`. Without that file the server fails with "We couldn't connect to 'https://huggingface.co'".
7. The teacher's tests:
   ```bash
   cd ~/smolvla && python -m unittest discover -s tests
   ```
   → `Ran 133 tests ... OK`.

## 4. Upload the checkpoint
The checkpoint is `C:\intern\Lumos Stationlite\docs\010000`. Only `pretrained_model` is needed (about 0.9 GB).
1. On the laptop (PowerShell):
   ```powershell
   Compress-Archive -Path "C:\intern\Lumos Stationlite\docs\010000\pretrained_model" -DestinationPath "C:\intern\Lumos Stationlite\pretrained_model_010000.zip"
   ```
2. VM SSH window → **UPLOAD FILE** (top right) → pick the zip. Done when `ls -lh ~/pretrained_model_010000.zip` shows about 870M.
3. Unzip **with Python**. Windows zips store `pretrained_model\config.json` with backslashes, and plain `unzip` makes broken names:
   ```bash
   mkdir -p ~/smolvla/checkpoints/010000 && cd ~/smolvla/checkpoints/010000 && python - <<'EOF'
   import zipfile, os
   z = zipfile.ZipFile(os.path.expanduser("~/pretrained_model_010000.zip"))
   for info in z.infolist():
       name = info.filename.replace("\\", "/")
       if name.endswith("/"):
           continue
       os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
       with open(name, "wb") as f:
           f.write(z.read(info))
   EOF
   ls ~/smolvla/checkpoints/010000/pretrained_model
   ```

## 5. First server start and test (on the VM)
```bash
tmux new -A -s server
```
```bash
conda activate smolvla && cd ~/smolvla && export HF_HOME=~/smolvla/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && python src/station_lite_smolvla_right7_server.py --checkpoint ~/smolvla/checkpoints/010000/pretrained_model --host 0.0.0.0 --port 8011 --seed 42 --seed-mode fixed --offline --no-verify-env
```
1. You see `[ready] ... gpu=NVIDIA L4 bind=0.0.0.0:8011`. Press **Ctrl+B**, then **D**.
2. From a second VM window, run `curl http://127.0.0.1:8011/health`. It returns JSON.
3. Run the teacher's warm-up (two fake `/predict` calls, in his `docs/DEPLOYMENT_GUIDE.md`). We got 2073 ms, then 348 ms.

## 6. Let the robot reach the VM

**6a. Allow an external IP for this one VM** (the organisation blocks them):
1. IAM & Admin → **Organization policies** → "Define allowed external IPs for VM instances" (`constraints/compute.vmExternalIpAccess`).
2. **Override parent's policy**, enforcement **Replace** (with "Merge", the parent's deny still wins).
3. Add rule → Custom → **Allow** → `projects/<PROJECT_ID>/zones/asia-southeast1-b/instances/vla-instance` → Set policy.

**6b. Fixed external IP:**
1. VPC network → IP addresses → **Reserve external static IP address**.
2. Name `smolvla-ip`, region `asia-southeast1`, attached to `vla-instance`.
3. Result: **<VM_IP>**.

**6c. Firewall: only the robot's network.** The server has no password.
1. On the robot, run `curl -s ifconfig.me` to get the robot network's public IP. It changes now and then: we've seen `210.10.2.180` and `210.10.1.235`.
2. VPC network → Firewall → **Create firewall rule**:
   - name `allow-robot-8011`
   - Ingress, Allow
   - target tag `smolvla-server`
   - source `210.10.0.0/22`
   - TCP `8011`
3. VM → Edit → **Network tags** → add `smolvla-server` → Save.
4. Browser SSH already works through the existing rule `default-allow-ssh` (tcp:22). Don't use "Retry without Cloud Identity-Aware Proxy".

**6d. Test from the robot:**
```bash
curl -s -o /dev/null --max-time 5 -w "%{http_code} %{time_total}s\n" http://<VM_IP>:8011/health
```
→ `200 0.04s`.
- If it hangs, the robot's IP probably moved outside `210.10.0.0/22`. Update the rule's source.
- A longer test with real-size requests showed the SIM sometimes spikes or drops a connection. The run settings in `my_smolvla_setup_notes.md` allow for that.

## 7. Server starts itself on boot
So the browser SSH window isn't needed every day:
1. VM → **Edit** → **Metadata** → Add item.
2. Key `startup-script`, value:
   ```bash
   #!/bin/bash
   sudo -u <USER> bash -lc 'tmux has-session -t server 2>/dev/null || tmux new -d -s server "source ~/miniforge3/bin/activate smolvla && cd ~/smolvla && export HF_HOME=~/smolvla/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && python src/station_lite_smolvla_right7_server.py --checkpoint ~/smolvla/checkpoints/010000/pretrained_model --host 0.0.0.0 --port 8011 --seed 42 --seed-mode fixed --offline --no-verify-env 2>&1 | tee ~/server.log; bash"'
   ```
3. Save. From the next START, the server comes up by itself in about 2 min. Its output is in `~/server.log`, and you can see it with `tmux attach -t server`.

## 8. Costs and switching off

| Thing | When it costs |
|---|---|
| L4 GPU + machine | only while the VM is **running** (the big cost) |
| Boot disk (100 GB) | always, a few $/month |
| Static IP `smolvla-ip` | about $3.60/month in use, about $7/month while the VM is stopped |
| Cloud NAT | small hourly charge |

- **Every session ends with:** Compute Engine → tick `vla-instance` → **STOP**.
- Billing reports lag by hours to a day, and trial credits are used first, so the money shown may not drop right away.
- **When the project ends:**
  1. Copy off anything you need.
  2. VPC network → IP addresses → `smolvla-ip` → **Release**.
  3. Delete the VM (this deletes its disk).
  4. Delete the Cloud NAT and the firewall rule.

## 9. Ideas for later
- **Faster answers / smoother motion:** the L4 takes about 360 ms, and the teacher's 4080 about 120 ms. An A100 (`a2-highgpu-1g`) would be faster but costs several times more and needs GPU quota. Changing it: stop VM → Edit → machine type → Start; nothing else changes.
- **Changing the machine type** (how we went from `g2-standard-4` to `g2-standard-8`):
  1. Compute Engine → tick `vla-instance` → **STOP**, and wait until it shows stopped.
  2. Click `vla-instance` → **EDIT** → Machine configuration → machine type → **Save**.
  3. **START**. The disk, IP, firewall and startup script stay the same.
  4. Check on the VM: `nvidia-smi --query-gpu=name,memory.total --format=csv && free -h` should show `NVIDIA L4, 23034 MiB` and about 31 Gi of RAM.
- **Training on the same VM:** upload the raw demos plus a manifest, then use the teacher's `src/training/convert_right_only_dataset.py` and `train_right_only_woodblock.py` (needs `hf download lerobot/smolvla_base` too).
