# MatClaw on vepfs + Docker: Troubleshooting & Environment Guide

This document records the full debugging process of getting MatClaw running on a vepfs-based server with SSH proxy forwarding and Docker containers. It serves as a reference for future developers and AI assistants working in similar environments.

## Server Environment Summary

| Item | Detail |
|------|--------|
| **OS** | Linux 5.4.250 (Debian-based) |
| **Filesystem** | vepfs (network filesystem, mounted at `/vepfs-mlp2/`) |
| **Docker** | Client 28.2.2, Server 24.0.9 (version mismatch, API downgraded to 1.43) |
| **Docker Root** | `/ebs/docker/165536.165536` (NOT on vepfs) |
| **GPU** | NVIDIA (driver 535.129.03, CUDA 12.2), accessible via `--gpus all` |
| **Proxy** | SSH port forwarding → `127.0.0.1:7890`, exposed via `http_proxy`/`https_proxy` in `~/.bashrc` |
| **Auth** | Claude OAuth via `/root/.claude/.credentials.json` (no API key in `.env`) |
| **Node.js** | Installed on vepfs at `/vepfs-mlp2/mlp-public/250266/` |

## Architecture Overview

```
Feishu/Web → MatClaw host process (Node.js)
                ↓ spawns Docker container per chat group
            Docker container (matclaw-agent:cuda)
                ↓ runs agent-runner → Claude Agent SDK
                ↓ output via stdout (primary) + IPC files (fallback)
            MatClaw host process
                ↓ sends reply back to Feishu/Web
```

Key files involved:
- `src/container-runner.ts` — spawns containers, reads output, manages mounts
- `container/agent-runner/src/index.ts` — runs inside container, calls Claude SDK
- `container/Dockerfile` — container image build (entrypoint compiles TypeScript on startup)

---

## Problem 1: Container Crashes Immediately (ENOENT)

### Symptom
Container exits with code 1 in ~4 seconds. stdout and stderr are both empty.

### Root Cause
Claude Agent SDK calls `appendFileSync('~/.claude/debug/<uuid>.txt')` at startup without creating the `debug/` directory first. MatClaw creates the `.claude/` mount directory with only `settings.json` and `skills/`, not `debug/`.

### Diagnosis Steps
1. Container logs showed empty stdout/stderr — no clue from outside
2. Ran container manually with `--entrypoint /bin/bash` and step-by-step debugging
3. Redirected node stderr to a mounted volume file to capture the actual error:
   ```
   Error: ENOENT: no such file or directory, open '/home/node/.claude/debug/xxx.txt'
   ```

### Fix
In `src/container-runner.ts`, after creating `groupSessionsDir` (the `.claude/` mount):

```typescript
mkdirWorld(path.join(groupSessionsDir, 'debug'));
```

---

## Problem 2: Permission Denied on Network Filesystem (EACCES)

### Symptom
After fixing Problem 1, the error changed to `EACCES: permission denied` on the same path.

### Root Cause
vepfs maps UIDs differently than local filesystems. Directories created by root (uid 0) on the host appear as `nobody:nogroup` inside the container. The container's `node` user (uid 1000) has no write permission with default `755` mode.

```
Host (root, uid=0) creates dir with mode 755
    ↓ mounted into container via -v
Container sees owner as nobody:nogroup (vepfs UID mapping)
    ↓
Container's node user (uid=1000) → cannot write (only owner has write)
```

This affects ALL directories that the container needs to write to:
- `~/.claude/` and `~/.claude/debug/`
- `/workspace/group/` (group workspace)
- `/workspace/ipc/` and all subdirectories
- Any other bind-mounted writable paths

### Fix
Created a helper function in `src/container-runner.ts`:

```typescript
/**
 * Create a directory with world-writable permissions.
 * On network filesystems (vepfs, NFS) the host creates dirs as root but the
 * container runs as uid 1000 (node). Ownership may be mapped to nobody:nogroup
 * so we need mode 0o777 to ensure the container user can write.
 */
function mkdirWorld(dirPath: string): void {
  fs.mkdirSync(dirPath, { recursive: true });
  try { fs.chmodSync(dirPath, 0o777); } catch { /* best-effort */ }
}
```

Replaced all `fs.mkdirSync(...)` calls for container-writable directories with `mkdirWorld(...)`. This includes:
- `.claude/` and `.claude/debug/`
- `.codex/`
- IPC directories (`messages/`, `tasks/`, `input/`, `output/`)
- Group workspace directory
- Logs directory

For existing installations, also run once:
```bash
chmod -R 777 /path/to/matclaw/data/ /path/to/matclaw/groups/
```

---

## Problem 3: API Returns 403 Forbidden

### Symptom
Container no longer crashes, but Claude API returns `403 {"error":{"type":"forbidden","message":"Request not allowed"}}`.

### Root Cause
The host machine accesses the internet through an SSH port-forwarding proxy (`http_proxy=http://127.0.0.1:7890`). This environment variable exists in the host shell but is NOT passed to Docker containers. Without the proxy, the container tries to connect directly to `api.anthropic.com`, which fails (network restricted or geo-blocked).

### Fix
In `src/container-runner.ts`, in the `buildContainerArgs()` function, automatically forward proxy environment variables to the container:

```typescript
for (const proxyVar of [
  'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
  'no_proxy', 'NO_PROXY',
]) {
  const val = process.env[proxyVar];
  if (val) {
    const containerVal = val.replace(
      /127\.0\.0\.1|localhost/g,
      'host.docker.internal',
    );
    args.push('-e', `${proxyVar}=${containerVal}`);
  }
}
// Linux requires explicit --add-host for host.docker.internal
if (process.platform === 'linux' && hasProxy) {
  args.push('--add-host', 'host.docker.internal:host-gateway');
}
```

The `127.0.0.1` → `host.docker.internal` replacement is critical: inside a container, `127.0.0.1` refers to the container itself, not the host.

---

## Problem 4: ECONNREFUSED on Docker Bridge IP

### Symptom
Proxy env vars are passed, but container gets `connect ECONNREFUSED 172.17.0.1:7890`.

### Root Cause
SSH port forwarding binds to `127.0.0.1` only by default. Docker's `host.docker.internal` resolves to the Docker bridge IP (`172.17.0.1`), which is NOT `127.0.0.1`.

```
SSH tunnel listens on:     127.0.0.1:7890 ✓
Container connects to:     172.17.0.1:7890 ✗ (nobody listening here)
```

### Fix
Use `socat` to bridge the Docker bridge IP to localhost:

```bash
socat TCP-LISTEN:7890,fork,reuseaddr,bind=172.17.0.1 TCP:127.0.0.1:7890 &
```

This must run whenever SSH is connected. Added to `~/.bashrc` with idempotency check:

```bash
# Bridge SSH proxy to Docker bridge so containers can use the proxy.
if ! ss -tln 2>/dev/null | grep -q '172.17.0.1:7890'; then
  socat TCP-LISTEN:7890,fork,reuseaddr,bind=172.17.0.1 TCP:127.0.0.1:7890 &>/dev/null &
fi
```

**Security note:** This only listens on the Docker bridge IP, not on `0.0.0.0`. Only local Docker containers can reach it.

### Alternative
If you control the SSH client config, you can make SSH listen on all interfaces:
```
ssh -R 0.0.0.0:7890:proxy-target:port ...
```
But this may expose the proxy to other users on the same machine.

---

## Problem 5: Docker stdout Pipe Data Lost

### Symptom
API works, agent generates correct responses (visible in session files), but MatClaw never receives the output. The `container-live.log` is empty. Feishu gets no reply.

### Root Cause
Docker's stdout/stderr pipe does not deliver data to the host process in this vepfs environment. This was confirmed by multiple tests:

```bash
# This produces NO output on the host, even though the container runs fine:
docker run --rm ubuntu:22.04 echo "hello"
# (empty)

# But writing to a mounted volume works:
docker run --rm -v /vepfs-path:/mnt --entrypoint bash image -c "echo hello > /mnt/test.txt"
# /mnt/test.txt contains "hello"
```

This appears to be a compatibility issue between Docker daemon (running on `/ebs/`) and the vepfs filesystem where the MatClaw process runs. The pipe file descriptors may not work correctly across filesystem boundaries.

### Fix
Added an **IPC file fallback** mechanism. The agent-runner writes output to both stdout (for normal environments) and to JSON files in the IPC directory (for environments where stdout is broken).

**In `container/agent-runner/src/index.ts`:**

```typescript
const IPC_OUTPUT_DIR = '/workspace/ipc/output';
let ipcOutputSeq = 0;

function writeOutput(output: ContainerOutput): void {
  // Primary: stdout markers
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);

  // Fallback: IPC file (for environments where Docker stdout is broken)
  try {
    fs.mkdirSync(IPC_OUTPUT_DIR, { recursive: true });
    const seq = String(ipcOutputSeq++).padStart(6, '0');
    fs.writeFileSync(
      path.join(IPC_OUTPUT_DIR, `${seq}-${Date.now()}.json`),
      JSON.stringify(output),
    );
  } catch { /* best-effort */ }
}
```

**In `src/container-runner.ts`:**

Added a polling loop that checks `ipc/output/` every 500ms for JSON files, parses them, and feeds them into the same output chain as stdout:

```typescript
const pollIpcOutput = () => {
  if (!ipcPolling) return;
  try {
    const files = fs.readdirSync(ipcOutputDir)
      .filter(f => f.endsWith('.json')).sort();
    for (const file of files) {
      const parsed = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      fs.unlinkSync(filePath); // consume the file
      // ... feed into output chain (same as stdout path)
    }
  } catch { /* dir may not exist yet */ }
  setTimeout(pollIpcOutput, 500);
};
```

Also does a final poll on container close to catch any last-moment output.

---

## Problem 6: `docker exec` and `docker run` stdout Completely Silent

### Symptom
When running interactive commands against a running container or launching one-off containers, **all stdout is lost** — no output reaches the host terminal:

```bash
# All of these produce ZERO output on the host:
docker run --rm ubuntu:24.04 echo "hello"
docker exec <running-container> pip list
docker exec <running-container> /bin/bash -c "echo test"
docker exec <running-container> python -c "print('hi')"

# Even piping, tee, variable capture — all empty:
result=$(docker exec <container> echo hello); echo "got: $result"
# got:
```

stderr is also affected. Exit codes return correctly (0), but no data arrives.

### Root Cause
Same underlying issue as Problem 5. Docker's stdout/stderr pipe mechanism does not deliver data to the host process when the Docker daemon's storage backend (`/ebs/`) and the calling process's working directory (vepfs) are on different filesystem types.

This affects:
- `docker run` — stdout/stderr of the container's main process
- `docker exec` — stdout/stderr of commands run inside an existing container
- `docker logs` — may also be empty or incomplete

This does **NOT** affect:
- File I/O inside the container (reading/writing files works normally)
- File I/O on bind-mounted volumes (container can write, host can read)
- `docker cp` — copying files between container and host
- `docker inspect`, `docker ps`, `docker history` — metadata commands work fine
- Network operations inside the container (API calls, downloads, etc.)

### Workarounds

**Workaround 1: Write to file inside container, then `docker cp` out**
```bash
# Run command with output redirected to a file inside the container
docker exec <container> /bin/bash -c "pip list > /tmp/output.txt 2>&1"

# Copy the file out to the host
docker cp <container>:/tmp/output.txt ./output.txt

# Read it on the host
cat ./output.txt
```

**Workaround 2: Write to a bind-mounted volume**
```bash
# If the container already has a mounted volume:
docker exec <container> /bin/bash -c "pip list > /workspace/group/output.txt 2>&1"

# Read directly from the host mount path
cat groups/<group>/output.txt
```

**Workaround 3: Use `docker history` for image inspection**
When you need to verify what was installed in an image (e.g., which pip packages), `docker history` reads image metadata — not container stdout — so it works reliably:

```bash
# Check all pip install layers in the image:
docker history matclaw-agent:cuda --no-trunc | grep "pip install"
```

### Impact on MatClaw Operations
- **Agent execution:** Not affected. MatClaw uses IPC file fallback (Problem 5 fix) for all agent output.
- **Data downloads (e.g., matbench):** Not affected. Downloads and file operations happen inside the container filesystem or on mounted volumes.
- **Model training:** Not affected. Training runs entirely inside the container.
- **Manual debugging:** Affected. Developers must use file-based workarounds (above) instead of relying on terminal output from `docker exec`.

### Lesson
- In environments where Docker stdout is broken, **never assume command output will reach the host terminal**
- Always have a file-based fallback for any cross-container communication
- `docker history` and `docker inspect` are reliable alternatives for image introspection since they read daemon metadata, not container stdout

---

## Debugging Techniques Learned

### 1. When Docker stdout is invisible
Docker stdout/stderr may not work in certain environments. Use **mounted volumes** to extract debug info:

```bash
docker run --rm \
  -v /host/path:/tmp/debug \
  --entrypoint bash image \
  -c "your-command > /tmp/debug/stdout.txt 2>/tmp/debug/stderr.txt"
```

### 2. Claude SDK debug logs
The SDK writes detailed logs to `~/.claude/debug/`. Check the newest `.txt` file:

```bash
ls -lt ~/.claude/debug/*.txt | head -1
# then read it, grep for ERROR
```

### 3. Checking container processes from outside
```bash
docker top <container-name>    # see what's running inside
docker exec <name> bash -c "..." # run commands inside (but stdout may be lost!)
```

### 4. Session files contain the actual response
Even if output delivery fails, the SDK saves conversation history:
```
data/sessions/<group>/.claude/projects/-workspace-group/*.jsonl
```
Parse these to see if the agent actually generated a response.

### 5. Port/process cleanup
```bash
ss -tlnp | grep <port>       # find what's using a port
kill -9 <pid>                 # force kill (use when normal kill doesn't work)
pkill -9 -f "tsx src/index"   # kill by command pattern
```

---

## Files Modified

| File | Changes |
|------|---------|
| `src/container-runner.ts` | Added `mkdirWorld()` helper; created `.claude/debug/` dir; forwarded proxy env vars to containers with localhost→host.docker.internal rewrite; added `--add-host` for Linux; added IPC output file polling as stdout fallback; used `mkdirWorld()` for all container-writable dirs |
| `container/agent-runner/src/index.ts` | Added IPC file fallback in `writeOutput()` — writes JSON to `/workspace/ipc/output/` alongside stdout |
| `~/.bashrc` | Added socat bridge auto-start for Docker proxy access |

---

## Checklist for Similar Environments

If you're deploying MatClaw on a server with network filesystem + Docker + SSH proxy:

- [ ] Ensure all container-writable directories use `chmod 777` (or use `mkdirWorld()`)
- [ ] Set `http_proxy`/`https_proxy` in `.env` OR ensure they're in the shell environment
- [ ] Run socat bridge if proxy listens on localhost only
- [ ] Verify Docker stdout works: `docker run --rm ubuntu echo test` — if empty, IPC fallback is needed
- [ ] Check `data/sessions/<group>/.claude/debug/` for SDK-level errors
