# RUNBOOK — Build the NeMo-RL image on an Azure VM and run it on Azure AI Runtime

Copy-paste, top to bottom. Replace the values in the **`### 0. Variables`** block
once; everything else uses them.

> **Important — where the final image lives:** AI Runtime can only pull custom
> images from **Docker Hub** (ACR / ECR / GCR / GHCR are **not** supported yet —
> confirmed with the AIR team; ACR support is on the roadmap but not shipped).
> So even though you *build* on an Azure VM, you must **push the final image to
> Docker Hub**, not ACR.
>
> **Build needs no GPU.** nvcc compiles for sm90 without a physical GPU. Use a
> cheap CPU VM to build; do the GPU smoke-test on AIR (1×H100) at the end.

---

### 0. Variables (edit these once, then paste the rest as-is)

```bash
# --- Azure ---
export AZ_RG="rg-nemo-rl-build"
export AZ_LOCATION="japaneast"          # pick a region you have quota in
export AZ_VM="nemo-rl-builder"
export AZ_VM_SIZE="Standard_D16as_v5"   # 16 vCPU / 64 GB, no GPU (build only)
export AZ_ADMIN="azureuser"

# --- Docker Hub (the registry AIR will pull from) ---
export DH_USER="<your-dockerhub-user>"
export DH_IMAGE="air-nemo-rl"
export DH_TAG="v0.4.0"                   # -> <your-dockerhub-user>/air-nemo-rl:v0.4.0

# --- Databricks (Azure workspace with AI Runtime enabled) ---
export DBX_PROFILE="<your-azure-databricks-profile>"
```

---

### 0.5. Clone this repo (on your laptop)

The repo is **private**, so use the GitHub CLI (already authenticated) or an SSH
key. Pick one:

```bash
# Option A — GitHub CLI (simplest if `gh auth status` is logged in)
gh repo clone hiouchiy/llm-finetuning-on-databricks-air
cd llm-finetuning-on-databricks-air

# Option B — HTTPS (will prompt for a GitHub Personal Access Token)
git clone https://github.com/hiouchiy/llm-finetuning-on-databricks-air.git
cd llm-finetuning-on-databricks-air

# Option C — SSH (if you have an SSH key on GitHub)
git clone git@github.com:hiouchiy/llm-finetuning-on-databricks-air.git
cd llm-finetuning-on-databricks-air
```

From here on, run commands **from the repo root** (the folder you just `cd`'d
into). The file you build is `cli/06_custom_docker_nemo_rl/Dockerfile`.

---

### 1. Create the Azure VM (Ubuntu 22.04, big OS disk)

Run on your laptop (needs `az` CLI logged in: `az login`).

```bash
az group create -n "$AZ_RG" -l "$AZ_LOCATION"

az vm create \
  -g "$AZ_RG" -n "$AZ_VM" \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size "$AZ_VM_SIZE" \
  --admin-username "$AZ_ADMIN" \
  --generate-ssh-keys \
  --os-disk-size-gb 256 \
  --storage-sku Premium_LRS

# Grab the public IP
export VM_IP=$(az vm show -g "$AZ_RG" -n "$AZ_VM" -d --query publicIps -o tsv)
echo "VM IP: $VM_IP"
```

> A 256 GB OS disk is intentional: the 20 GB limit is for the *final* image;
> build intermediates (vLLM/TE/flash-attn compiles) are far larger.

---

### 2. Install Docker + buildx on the VM

```bash
ssh "$AZ_ADMIN@$VM_IP" 'bash -s' <<'REMOTE'
set -euo pipefail
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo usermod -aG docker $USER
docker --version && docker buildx version
REMOTE

# Re-login so the docker group applies
ssh "$AZ_ADMIN@$VM_IP" 'docker run --rm hello-world >/dev/null 2>&1 && echo "docker OK" || echo "re-login needed"'
```

If it says "re-login needed", just `ssh` in again (new shell picks up the group).

---

### 3. Copy the Dockerfile to the VM

Run from the repo root you cloned in step 0.5:

```bash
scp cli/06_custom_docker_nemo_rl/Dockerfile "$AZ_ADMIN@$VM_IP:~/Dockerfile"
```

---

> **AWS target?** Use `Dockerfile.aws` instead of `Dockerfile` (identical except
> the base is `dcs-base-aws-devel` with EFA networking). Build it on the SAME VM
> — the build host's cloud is irrelevant; only `--platform linux/amd64` matters.
> Just point `-f` at `~/Dockerfile.aws` and use a distinct tag, e.g.
> `<registry-user>/air-nemo-rl:v0.4.0-aws`.

### 4. Build and push to Docker Hub (on the VM)

```bash
ssh "$AZ_ADMIN@$VM_IP" "bash -s" <<REMOTE
set -euo pipefail
# Log in to Docker Hub (you'll be prompted for a Docker Hub Personal Access Token)
docker login -u "$DH_USER"

# Build for linux/amd64 (AIR workers are x86_64) and push.
docker buildx create --use --name airbuilder 2>/dev/null || docker buildx use airbuilder
docker buildx build \
  --platform linux/amd64 \
  -t "$DH_USER/$DH_IMAGE:$DH_TAG" \
  --push \
  -f ~/Dockerfile ~
REMOTE
```

> The Dockerfile's final stage imports mamba/TE/vLLM/flash-attn, so if any
> version is mismatched **the build fails here** — you won't discover it later
> on AIR. Expect this build to take a while (vLLM/TE compile from source).
> Keep the final image < 20 GB (check with `docker images`); if it's over,
> tell me and we'll add a multi-stage slim step.

---

### 5. Register the image with AI Runtime and run

Run on your laptop (needs the `air` CLI + Databricks Azure profile).

```bash
# Register (Docker Hub only). You'll be prompted for the Docker Hub PAT.
air register image "$DH_USER/$DH_IMAGE:$DH_TAG" \
  --interactive-authenticate -p "$DBX_PROFILE"

# Run using train.yaml in this folder. Edit train.yaml first:
#   - environment.docker_image.url = <your-dockerhub-user>/air-nemo-rl:v0.4.0
#   - command = your real NeMo-RL entry point (absolute path under /opt/nemo-rl)
air run -f cli/06_custom_docker_nemo_rl/train.yaml -p "$DBX_PROFILE" --watch
```

---

### 6. Tear down the build VM (stop paying for it)

```bash
az group delete -n "$AZ_RG" --yes --no-wait
```

---

## Cheatsheet: what must match (so versions don't drift)

| Layer | Value | Set by |
|---|---|---|
| AIR base image | `dcs-base-azure-devel` (CUDA 12.9.1, /opt/venv) | Dockerfile `FROM` |
| torch | **2.7.1** | Dockerfile (NeMo-RL v0.4.0 pin) |
| vLLM / TE / mamba / flash-attn | resolved by NeMo-RL pyproject @ `v0.4.0` | `uv pip install -e /opt/nemo-rl` |
| GPU arch | `TORCH_CUDA_ARCH_LIST=9.0` (H100) | Dockerfile ENV |
| Execution | Azure AI Runtime (1×/8× H100) | `air run` |

The rule: **don't hand-pick wheels.** torch is pinned once; NeMo-RL resolves the
rest consistently against it; the build-time import gate proves they agree.

## Known Azure AIR caveats (from earlier verification)

- **Multi-node H100 is not available on Azure AIR** yet (single-node 8×H100 only,
  and subject to capacity). Keep `compute.num_accelerators` ≤ 8 on Azure.
- Custom images are a **Beta** feature and are **not compatible with Serverless
  Egress Control (SEG)** workspaces.
